# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import byte_v2_attn as byte_v2_attn_module
from vllm.v1.attention.backends.byte_v2_layout import (
    DEFAULT_BYTE_V2_TILE_POLICY,
    ByteV2RawStagingLayout,
)
from vllm.v1.attention.backends.byte_v2_static_w16 import (
    STATIC_W16_CANONICAL_METADATA_STATUS,
    STATIC_W16_CANONICAL_RAW_FALLBACK_STATUS,
    STATIC_W16_PAGE_BYTES,
    STATIC_W16_STATUS_OFFSET_BYTES,
)

_REQUIRED_WRITER_OPS = (
    "byte_v2_prepare_raw_staging",
    "byte_v2_append_raw_staging",
    "byte_v2_release_raw_staging",
    "byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache",
    "byte_v2_static_w16_commit_raw_staging_to_hybrid_cache",
    "byte_v2_static_w16_update_hybrid_cache_raw_tail_q1",
)
_K_BASE = 115
_V_BASE = 110


def _writer_ops_are_available() -> bool:
    namespace = getattr(torch.ops, "_C_cache_ops", None)
    if namespace is None:
        return False
    try:
        return all(
            getattr(namespace, name) is not None for name in _REQUIRED_WRITER_OPS
        )
    except AttributeError:
        return False


def _make_bf16_tokens(
    num_tokens: int,
    *,
    exponent_base: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(
        num_tokens * 8 * 128,
        dtype=torch.int32,
        device=device,
    )
    sign = ((indices + offset) & 1) << 15
    exponent = exponent_base + ((indices // 17 + offset) % 16)
    mantissa = (indices * 29 + offset * 11) % 128
    bits = sign | (exponent << 7) | mantissa
    return bits.to(torch.int16).view(torch.bfloat16).reshape(num_tokens, 8, 128)


def _make_slot_mapping(
    request_lengths: tuple[int, ...],
    physical_pages: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, int]]:
    slots: list[int] = []
    expected_rows: dict[int, int] = {}
    page_index = 0
    for request_len in request_lengths:
        request_pages = (request_len + 15) // 16
        for token_index in range(request_len):
            physical_page = physical_pages[page_index + token_index // 16]
            slots.append(physical_page * 16 + token_index % 16)
            expected_rows[physical_page] = token_index % 16 + 1
        page_index += request_pages
    assert page_index == len(physical_pages)
    return torch.tensor(slots, dtype=torch.int64, device=device), expected_rows


def _make_manager_case(
    *,
    num_blocks: int,
    num_staging_slots: int,
    num_raw_slots: int,
    device: torch.device,
    retain_cascade_q16: bool = False,
):
    raw_layout = ByteV2RawStagingLayout()
    kv_cache = torch.zeros(
        (num_blocks, STATIC_W16_PAGE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    kv_cache[
        :,
        STATIC_W16_STATUS_OFFSET_BYTES : STATIC_W16_STATUS_OFFSET_BYTES + 4,
    ].view(torch.int32).fill_(STATIC_W16_CANONICAL_METADATA_STATUS)
    hybrid_state = SimpleNamespace(
        raw_pages=torch.full(
            (num_raw_slots, raw_layout.slot_size_bytes),
            0xD3,
            dtype=torch.uint8,
            device=device,
        ),
        page_to_raw_slot=torch.full(
            (num_blocks,), -1, dtype=torch.int32, device=device
        ),
        free_slots=torch.arange(num_raw_slots, dtype=torch.int32, device=device),
        free_count=torch.tensor([num_raw_slots], dtype=torch.int32, device=device),
        fatal=torch.zeros(1, dtype=torch.int32, device=device),
    )
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=num_blocks,
        num_staging_slots=num_staging_slots,
        slot_size_bytes=raw_layout.slot_size_bytes,
        device=device,
    ).allocate()
    # Hydrate intentionally leaves invalid partial-page rows untouched. Use a
    # fixed sentinel so the test can also prove the skip preserves those bytes.
    workspace.raw_staging.fill_(0xA5)
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
        hybrid_raw_mutable_tail_q1=True,
        static_w16_retain_cascade_q16=retain_cascade_q16,
    )
    manager.configure_static_w16_bases((_K_BASE, _V_BASE))
    manager.bind_shared_workspace(workspace)
    return manager, workspace, hybrid_state, kv_cache


def _active_staging_pages(staged):
    raw_staging, block_to_staging_slot, _, valid_rows = staged
    result = {}
    for physical_page, staging_slot in enumerate(
        block_to_staging_slot.detach().cpu().tolist()
    ):
        if staging_slot >= 0:
            result[physical_page] = (
                raw_staging[staging_slot],
                int(valid_rows[staging_slot].item()),
            )
    return result


def _persistent_raw_pages(hybrid_state):
    result = {}
    for physical_page, raw_slot in enumerate(
        hybrid_state.page_to_raw_slot.detach().cpu().tolist()
    ):
        if raw_slot >= 0:
            result[physical_page] = hybrid_state.raw_pages[raw_slot]
    return result


def _assert_staged_tokens_match_input(
    staged,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    raw_staging, block_to_staging_slot, _, _ = staged
    staged_kv = raw_staging.view(torch.bfloat16).view(-1, 2, 8, 16, 128)
    physical_pages = torch.div(slot_mapping, 16, rounding_mode="floor")
    rows = slot_mapping % 16
    staging_slots = block_to_staging_slot.index_select(0, physical_pages)
    token_indices = torch.arange(key.shape[0], device=key.device)
    staged_key = staged_kv[staging_slots, 0, :, rows, :]
    staged_value = staged_kv[staging_slots, 1, :, rows, :]
    assert torch.equal(staged_key[token_indices], key)
    assert torch.equal(staged_value[token_indices], value)


def _hydrate_static_w16_pages(
    *,
    kv_cache: torch.Tensor,
    hybrid_state,
    physical_pages: tuple[int, ...],
    valid_rows: int,
) -> torch.Tensor:
    device = kv_cache.device
    hydrated = torch.full(
        (len(physical_pages), ByteV2RawStagingLayout().slot_size_bytes),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    byte_v2_attn_module.byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache(
        hydrated,
        kv_cache,
        hybrid_state.raw_pages,
        hybrid_state.page_to_raw_slot,
        torch.tensor(physical_pages, dtype=torch.int32, device=device),
        torch.full(
            (len(physical_pages),),
            valid_rows,
            dtype=torch.int32,
            device=device,
        ),
        hybrid_state.fatal,
    )
    return hydrated.view(torch.bfloat16).view(-1, 2, 8, 16, 128)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_static_w16_batched_raw_tail_q1_matches_staging_writer(monkeypatch):
    if not _writer_ops_are_available():
        pytest.skip("Static-W16 writer schemas are not registered")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    physical_pages = (7, 1, 6, 2)
    num_requests = len(physical_pages)
    metadata = SimpleNamespace(
        num_actual_tokens=num_requests,
        max_query_len=1,
        query_start_loc_cpu=torch.arange(num_requests + 1, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.ones(num_requests, dtype=torch.int32),
        is_prefilling=torch.zeros(num_requests, dtype=torch.bool),
    )
    runs = []
    for batched_q1 in (False, True):
        monkeypatch.setenv(
            "BYTE_V2_STATIC_W16_BATCHED_RAW_TAIL_Q1",
            "1" if batched_q1 else "0",
        )
        runs.append(
            _make_manager_case(
                num_blocks=10,
                num_staging_slots=num_requests,
                num_raw_slots=num_requests,
                device=device,
            )
        )

    expected_keys = []
    expected_values = []
    for row in range(16):
        key = _make_bf16_tokens(
            num_requests,
            exponent_base=_K_BASE,
            offset=41 + row,
            device=device,
        )
        value = _make_bf16_tokens(
            num_requests,
            exponent_base=_V_BASE,
            offset=83 + row,
            device=device,
        )
        if row == 0:
            # Make one request exceed the 128-entry page escape capacity. The
            # batched seal must retain RAW authority for this page while the
            # other three pages demote to compact authority.
            key_bits = key[0].view(torch.int16).reshape(-1)
            old_bits = key_bits[:129].to(torch.int32) & 0xFFFF
            key_bits[:129] = ((old_bits & 0x807F) | ((_K_BASE + 20) << 7)).to(
                torch.int16
            )
        expected_keys.append(key)
        expected_values.append(value)
        slot_mapping = torch.tensor(
            [physical_page * 16 + row for physical_page in physical_pages],
            dtype=torch.int64,
            device=device,
        )
        metadata.seq_lens_cpu_upper_bound.fill_(row + 1)

        for manager, _, _, kv_cache in runs:
            assert manager.update(
                key=key,
                value=value,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                attn_metadata=metadata,
            ) == (True, False)

        hydrated = [
            _hydrate_static_w16_pages(
                kv_cache=kv_cache,
                hybrid_state=hybrid_state,
                physical_pages=physical_pages,
                valid_rows=row + 1,
            )
            for _, _, hybrid_state, kv_cache in runs
        ]
        torch.accelerator.synchronize(device)
        assert torch.equal(hydrated[1], hydrated[0])
        for previous_row in range(row + 1):
            assert torch.equal(
                hydrated[1][:, 0, :, previous_row, :],
                expected_keys[previous_row],
            )
            assert torch.equal(
                hydrated[1][:, 1, :, previous_row, :],
                expected_values[previous_row],
            )

        expected_free_count = num_requests - 1 if row == 15 else 0
        for _, workspace, hybrid_state, _ in runs:
            assert hybrid_state.fatal.item() == 0
            assert hybrid_state.free_count.item() == expected_free_count
            assert bool((workspace.block_to_staging_slot == -1).all())
            assert bool((workspace.staging_to_physical_block == -1).all())
            assert bool((workspace.valid_rows == 0).all())
            assert workspace.next_staging_slot.item() == 0
            assert workspace.overflow.item() == 0

    reference_state = runs[0][2]
    candidate_state = runs[1][2]
    for state in (reference_state, candidate_state):
        mapping = state.page_to_raw_slot.detach().cpu().tolist()
        assert mapping[physical_pages[0]] >= 0
        assert all(mapping[page] == -1 for page in physical_pages[1:])
        free_count = int(state.free_count.item())
        live_slots = [slot for slot in mapping if slot >= 0]
        free_slots = state.free_slots[:free_count].detach().cpu().tolist()
        assert sorted(live_slots + free_slots) == list(range(num_requests))

    reference_cache = runs[0][3]
    candidate_cache = runs[1][3]
    for physical_page in physical_pages[1:]:
        assert torch.equal(
            candidate_cache[physical_page],
            reference_cache[physical_page],
        )
    metadata_start = STATIC_W16_STATUS_OFFSET_BYTES
    metadata_end = metadata_start + 5 * 4
    assert torch.equal(
        candidate_cache[physical_pages[0], metadata_start:metadata_end],
        reference_cache[physical_pages[0], metadata_start:metadata_end],
    )


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_static_w16_retains_and_resets_first_cascade_q16_page():
    if not _writer_ops_are_available():
        pytest.skip("Static-W16 writer schemas are not registered")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    physical_pages = (7, 1, 6, 2)
    num_requests = len(physical_pages)
    manager, workspace, hybrid_state, kv_cache = _make_manager_case(
        num_blocks=10,
        num_staging_slots=num_requests,
        num_raw_slots=num_requests,
        device=device,
        retain_cascade_q16=True,
    )
    key = _make_bf16_tokens(
        num_requests * 16,
        exponent_base=_K_BASE,
        offset=131,
        device=device,
    )
    value = _make_bf16_tokens(
        num_requests * 16,
        exponent_base=_V_BASE,
        offset=173,
        device=device,
    )
    slot_mapping, _ = _make_slot_mapping(
        (16,) * num_requests,
        physical_pages,
        device,
    )
    common_prefix_len = 4_080
    metadata = SimpleNamespace(
        num_actual_tokens=num_requests * 16,
        max_query_len=16,
        query_start_loc_cpu=torch.arange(
            0,
            (num_requests + 1) * 16,
            16,
            dtype=torch.int32,
        ),
        seq_lens_cpu_upper_bound=torch.full(
            (num_requests,),
            common_prefix_len + 16,
            dtype=torch.int32,
        ),
        is_prefilling=torch.ones(num_requests, dtype=torch.bool),
        use_cascade=True,
        common_prefix_len=common_prefix_len,
    )

    assert manager.update(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        attn_metadata=metadata,
    ) == (True, False)
    hydrated = _hydrate_static_w16_pages(
        kv_cache=kv_cache,
        hybrid_state=hybrid_state,
        physical_pages=physical_pages,
        valid_rows=16,
    )
    torch.accelerator.synchronize(device)

    assert torch.equal(hydrated[:, 0].permute(0, 2, 1, 3).reshape_as(key), key)
    assert torch.equal(
        hydrated[:, 1].permute(0, 2, 1, 3).reshape_as(value),
        value,
    )
    mapping = hybrid_state.page_to_raw_slot.detach().cpu().tolist()
    assert all(mapping[page] >= 0 for page in physical_pages)
    assert hybrid_state.free_count.item() == 0
    status = kv_cache[
        list(physical_pages),
        STATIC_W16_STATUS_OFFSET_BYTES : STATIC_W16_STATUS_OFFSET_BYTES + 4,
    ].view(torch.int32)
    assert bool((status == STATIC_W16_CANONICAL_RAW_FALLBACK_STATUS).all())
    assert hybrid_state.fatal.item() == 0
    assert bool((workspace.block_to_staging_slot == -1).all())

    byte_v2_attn_module.byte_v2_reset_raw_fallback_pages(
        hybrid_state.page_to_raw_slot,
        hybrid_state.free_slots,
        hybrid_state.free_count,
        hybrid_state.fatal,
        torch.tensor(physical_pages, dtype=torch.int32, device=device),
    )
    torch.accelerator.synchronize(device)
    assert all(
        hybrid_state.page_to_raw_slot[page].item() == -1 for page in physical_pages
    )
    assert hybrid_state.free_count.item() == num_requests
    assert hybrid_state.fatal.item() == 0


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_static_w16_initial_prefill_skip_hydrate_matches_hydrated_state(
    monkeypatch,
):
    if not _writer_ops_are_available():
        pytest.skip("Static-W16 writer schemas are not registered")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    request_lengths = (16, 17, 73)
    query_starts = (0, 16, 33, 106)
    physical_pages = (6, 1, 7, 0, 5, 2, 4, 3)
    num_tokens = query_starts[-1]
    slot_mapping, expected_rows = _make_slot_mapping(
        request_lengths,
        physical_pages,
        device,
    )
    key = _make_bf16_tokens(
        num_tokens,
        exponent_base=_K_BASE,
        offset=3,
        device=device,
    )
    value = _make_bf16_tokens(
        num_tokens,
        exponent_base=_V_BASE,
        offset=7,
        device=device,
    )
    # The first request is one full page. Force 129 K escapes across its first
    # two head chunks so this test covers a full RAW fallback page in addition
    # to compact full pages and two partial RAW pages.
    key_bits = key[:16].view(torch.int16).reshape(-1)
    old_bits = key_bits[:129].to(torch.int32) & 0xFFFF
    key_bits[:129] = ((old_bits & 0x807F) | ((_K_BASE + 20) << 7)).to(torch.int16)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(request_lengths, dtype=torch.int32),
    )

    runs = []
    for skip_hydrate in (False, True):
        monkeypatch.setenv(
            "BYTE_V2_STATIC_W16_INITIAL_PREFILL_SKIP_HYDRATE",
            "1" if skip_hydrate else "0",
        )
        manager, workspace, hybrid_state, kv_cache = _make_manager_case(
            num_blocks=10,
            num_staging_slots=len(physical_pages),
            num_raw_slots=len(physical_pages),
            device=device,
        )
        assert manager.update(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            attn_metadata=metadata,
            retain_initial_prefill=True,
        ) == (True, False)
        staged = manager.stage_initial_prefill(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            attn_metadata=metadata,
        )
        assert staged is not None
        torch.accelerator.synchronize(device)
        _assert_staged_tokens_match_input(staged, key, value, slot_mapping)
        runs.append((manager, workspace, hybrid_state, kv_cache, staged))

    reference = runs[0]
    candidate = runs[1]
    assert candidate[2].fatal.item() == reference[2].fatal.item() == 0
    assert candidate[2].free_count.item() == reference[2].free_count.item()
    free_count = int(reference[2].free_count.item())
    assert torch.equal(
        candidate[2].free_slots[:free_count].sort().values,
        reference[2].free_slots[:free_count].sort().values,
    )

    reference_staging = _active_staging_pages(reference[4])
    candidate_staging = _active_staging_pages(candidate[4])
    assert reference_staging.keys() == candidate_staging.keys() == expected_rows.keys()
    for physical_page, expected_valid_rows in expected_rows.items():
        reference_page, reference_rows = reference_staging[physical_page]
        candidate_page, candidate_rows = candidate_staging[physical_page]
        assert reference_rows == candidate_rows == expected_valid_rows
        assert torch.equal(candidate_page, reference_page)

    reference_raw = _persistent_raw_pages(reference[2])
    candidate_raw = _persistent_raw_pages(candidate[2])
    assert reference_raw.keys() == candidate_raw.keys()
    assert len(reference_raw) == 3
    for physical_page in reference_raw:
        assert torch.equal(candidate_raw[physical_page], reference_raw[physical_page])

    # Compact pages are the authoritative reader source and must match in full.
    # On RAW-authoritative pages, concurrent generic packing may assign unused
    # escape-directory offsets in a different order; compare only the live
    # status/base/valid-row words there, then compare the authoritative RAW page
    # above. This keeps the check strict on every byte that a reader can consume.
    for physical_page in range(reference[3].shape[0]):
        if physical_page not in reference_raw:
            assert torch.equal(candidate[3][physical_page], reference[3][physical_page])
            continue
        metadata_start = STATIC_W16_STATUS_OFFSET_BYTES
        metadata_end = metadata_start + 5 * 4
        assert torch.equal(
            candidate[3][physical_page, metadata_start:metadata_end],
            reference[3][physical_page, metadata_start:metadata_end],
        )

    for manager, workspace, _, _, staged in runs:
        manager.release_initial_prefill(staged[2], staged[3])
        torch.accelerator.synchronize(device)
        assert bool((workspace.block_to_staging_slot == -1).all())
        assert bool((workspace.staging_to_physical_block == -1).all())
        assert bool((workspace.valid_rows == 0).all())
        assert workspace.next_staging_slot.item() == 0
        assert workspace.overflow.item() == 0


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_static_w16_page_aligned_cached_prefill_skip_matches_hydrated_state(
    monkeypatch,
):
    if not _writer_ops_are_available():
        pytest.skip("Static-W16 writer schemas are not registered")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    query_len = 33
    context_len = 16
    physical_pages = (6, 1, 7)
    slot_mapping, expected_rows = _make_slot_mapping(
        (query_len,),
        physical_pages,
        device,
    )
    key = _make_bf16_tokens(
        query_len,
        exponent_base=_K_BASE,
        offset=13,
        device=device,
    )
    value = _make_bf16_tokens(
        query_len,
        exponent_base=_V_BASE,
        offset=17,
        device=device,
    )
    # Keep the first full page compact, force the second full page to RAW, and
    # leave the final page partial. This exercises every authoritative reader
    # source whose valid rows the aligned append replaces.
    key_bits = key[16:32].view(torch.int16).reshape(-1)
    old_bits = key_bits[:129].to(torch.int32) & 0xFFFF
    key_bits[:129] = ((old_bits & 0x807F) | ((_K_BASE + 20) << 7)).to(torch.int16)
    metadata = SimpleNamespace(
        num_actual_tokens=query_len,
        query_start_loc_cpu=torch.tensor((0, query_len), dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(
            (context_len + query_len,), dtype=torch.int32
        ),
        is_prefilling=torch.tensor((True,), dtype=torch.bool),
    )
    real_hydrate = (
        byte_v2_attn_module.byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache
    )

    runs = []
    hydrate_call_counts = []
    for skip_hydrate in (False, True):
        monkeypatch.setenv(
            "BYTE_V2_STATIC_W16_PAGE_ALIGNED_PREFILL_SKIP_HYDRATE",
            "1" if skip_hydrate else "0",
        )
        manager, workspace, hybrid_state, kv_cache = _make_manager_case(
            num_blocks=10,
            num_staging_slots=len(physical_pages),
            num_raw_slots=len(physical_pages),
            device=device,
        )
        hydrate_calls: list[None] = []

        def counted_hydrate(*args, _calls=hydrate_calls, **kwargs):
            _calls.append(None)
            return real_hydrate(*args, **kwargs)

        monkeypatch.setattr(
            byte_v2_attn_module,
            "byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache",
            counted_hydrate,
        )
        assert manager.update(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            attn_metadata=metadata,
        ) == (True, False)
        torch.accelerator.synchronize(device)
        hydrate_call_counts.append(len(hydrate_calls))

        physical_page_tensor = torch.tensor(
            physical_pages,
            dtype=torch.int32,
            device=device,
        )
        valid_rows = torch.tensor(
            [expected_rows[page] for page in physical_pages],
            dtype=torch.int32,
            device=device,
        )
        hydrated = torch.full(
            (len(physical_pages), ByteV2RawStagingLayout().slot_size_bytes),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        real_hydrate(
            hydrated,
            kv_cache,
            hybrid_state.raw_pages,
            hybrid_state.page_to_raw_slot,
            physical_page_tensor,
            valid_rows,
            hybrid_state.fatal,
        )
        torch.accelerator.synchronize(device)
        hydrated_kv = hydrated.view(torch.bfloat16).view(-1, 2, 8, 16, 128)
        for token_index in range(query_len):
            page_index, row = divmod(token_index, 16)
            assert torch.equal(hydrated_kv[page_index, 0, :, row, :], key[token_index])
            assert torch.equal(
                hydrated_kv[page_index, 1, :, row, :], value[token_index]
            )

        assert bool((workspace.block_to_staging_slot == -1).all())
        assert bool((workspace.staging_to_physical_block == -1).all())
        assert bool((workspace.valid_rows == 0).all())
        assert workspace.next_staging_slot.item() == 0
        assert workspace.overflow.item() == 0
        runs.append((hybrid_state, kv_cache, hydrated))

    assert hydrate_call_counts == [1, 0]
    reference_state, reference_cache, reference_hydrated = runs[0]
    candidate_state, candidate_cache, candidate_hydrated = runs[1]
    assert reference_state.fatal.item() == candidate_state.fatal.item() == 0
    assert torch.equal(candidate_hydrated, reference_hydrated)

    reference_raw = _persistent_raw_pages(reference_state)
    candidate_raw = _persistent_raw_pages(candidate_state)
    assert reference_raw.keys() == candidate_raw.keys()
    assert len(reference_raw) == 2
    for physical_page in reference_raw:
        assert torch.equal(candidate_raw[physical_page], reference_raw[physical_page])

    for physical_page in range(reference_cache.shape[0]):
        if physical_page not in reference_raw:
            assert torch.equal(
                candidate_cache[physical_page], reference_cache[physical_page]
            )
            continue
        metadata_start = STATIC_W16_STATUS_OFFSET_BYTES
        metadata_end = metadata_start + 5 * 4
        assert torch.equal(
            candidate_cache[physical_page, metadata_start:metadata_end],
            reference_cache[physical_page, metadata_start:metadata_end],
        )


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_static_w16_shared_prefix_cascade_matches_regular_reader(monkeypatch):
    if not _writer_ops_are_available():
        pytest.skip("Static-W16 writer schemas are not registered")
    fa2_namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    if fa2_namespace is None or not hasattr(
        fa2_namespace,
        "static_w16_canonical_varlen_fwd",
    ):
        pytest.skip("Static-W16 canonical FA2 reader is not registered")

    monkeypatch.delenv(
        "BYTE_V2_STATIC_W16_FA2_PROFILE_SEQ_RANGE",
        raising=False,
    )
    device = torch.device("cuda", torch.accelerator.current_device_index())
    num_requests = 8
    common_prefix_len = 256
    common_pages = tuple(range(16))
    private_pages = tuple(range(16, 16 + num_requests))
    manager, workspace, hybrid_state, kv_cache = _make_manager_case(
        num_blocks=32,
        num_staging_slots=24,
        num_raw_slots=24,
        device=device,
    )

    common_slots, _ = _make_slot_mapping(
        (common_prefix_len,),
        common_pages,
        device,
    )
    common_key = _make_bf16_tokens(
        common_prefix_len,
        exponent_base=_K_BASE,
        offset=23,
        device=device,
    )
    common_value = _make_bf16_tokens(
        common_prefix_len,
        exponent_base=_V_BASE,
        offset=29,
        device=device,
    )
    # The shared prefix itself includes one authoritative RAW page, so cascade
    # must preserve both compact decoding and raw-sidecar routing.
    raw_key_bits = common_key[:16].view(torch.int16).reshape(-1)
    old_bits = raw_key_bits[:129].to(torch.int32) & 0xFFFF
    raw_key_bits[:129] = ((old_bits & 0x807F) | ((_K_BASE + 20) << 7)).to(torch.int16)
    common_metadata = SimpleNamespace(
        num_actual_tokens=common_prefix_len,
        query_start_loc_cpu=torch.tensor((0, common_prefix_len), dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor((common_prefix_len,), dtype=torch.int32),
        is_prefilling=torch.tensor((True,), dtype=torch.bool),
    )
    assert manager.update(
        key=common_key,
        value=common_value,
        kv_cache=kv_cache,
        slot_mapping=common_slots,
        attn_metadata=common_metadata,
    ) == (True, False)

    suffix_slots = torch.tensor(
        [page * 16 for page in private_pages],
        dtype=torch.int64,
        device=device,
    )
    suffix_key = _make_bf16_tokens(
        num_requests,
        exponent_base=_K_BASE,
        offset=31,
        device=device,
    )
    suffix_value = _make_bf16_tokens(
        num_requests,
        exponent_base=_V_BASE,
        offset=37,
        device=device,
    )
    query_start_locs = torch.arange(
        num_requests + 1,
        dtype=torch.int32,
        device=device,
    )
    suffix_metadata = SimpleNamespace(
        num_actual_tokens=num_requests,
        query_start_loc_cpu=query_start_locs.cpu(),
        seq_lens_cpu_upper_bound=torch.full(
            (num_requests,),
            common_prefix_len + 1,
            dtype=torch.int32,
        ),
        is_prefilling=torch.zeros(num_requests, dtype=torch.bool),
    )
    assert manager.update(
        key=suffix_key,
        value=suffix_value,
        kv_cache=kv_cache,
        slot_mapping=suffix_slots,
        attn_metadata=suffix_metadata,
    ) == (True, False)

    query = _make_bf16_tokens(
        num_requests,
        exponent_base=120,
        offset=41,
        device=device,
    ).repeat(1, 4, 1)
    common_block_table = torch.tensor(
        common_pages,
        dtype=torch.int32,
        device=device,
    )
    block_table = torch.stack(
        [
            torch.cat(
                (
                    common_block_table,
                    torch.tensor([page], dtype=torch.int32, device=device),
                )
            )
            for page in private_pages
        ]
    )
    seq_lens = torch.full(
        (num_requests,),
        common_prefix_len + 1,
        dtype=torch.int32,
        device=device,
    )
    reference = torch.empty_like(query)
    byte_v2_attn_module.byte_v2_static_w16_fa2_paged_attention(
        reference,
        query,
        kv_cache,
        hybrid_state.raw_pages,
        hybrid_state.page_to_raw_slot,
        query_start_locs,
        block_table,
        seq_lens,
        scale=0.125,
        max_query_len=1,
        max_seq_len=common_prefix_len + 1,
        causal=True,
    )

    prefix_output, prefix_lse = (
        byte_v2_attn_module.byte_v2_static_w16_fa2_paged_attention_with_lse(
            query,
            kv_cache,
            hybrid_state.raw_pages,
            hybrid_state.page_to_raw_slot,
            torch.tensor([0, num_requests], dtype=torch.int32, device=device),
            block_table[:1, :16],
            torch.tensor([common_prefix_len], dtype=torch.int32, device=device),
            scale=0.125,
            max_query_len=num_requests,
            max_seq_len=common_prefix_len,
            causal=False,
        )
    )
    suffix_output, suffix_lse = (
        byte_v2_attn_module.byte_v2_static_w16_fa2_paged_attention_with_lse(
            query,
            kv_cache,
            hybrid_state.raw_pages,
            hybrid_state.page_to_raw_slot,
            query_start_locs,
            block_table[:, 16:],
            torch.ones(num_requests, dtype=torch.int32, device=device),
            scale=0.125,
            max_query_len=1,
            max_seq_len=1,
            causal=True,
        )
    )
    candidate = torch.empty_like(query)
    byte_v2_attn_module.merge_attn_states(
        candidate,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    )

    staged_prefix = manager.stage_cached_prefill(
        kv_cache=kv_cache,
        block_table=block_table[:1, :16],
        seq_len=common_prefix_len,
    )
    assert staged_prefix is not None
    raw_staging, page_map, local_block_table, valid_rows = staged_prefix
    try:
        hydrated_prefix_output, hydrated_prefix_lse = (
            byte_v2_attn_module.byte_v2_fa2_raw_staging_attention_with_lse(
                query,
                raw_staging,
                page_map,
                torch.tensor(
                    [0, num_requests],
                    dtype=torch.int32,
                    device=device,
                ),
                local_block_table,
                torch.tensor(
                    [common_prefix_len],
                    dtype=torch.int32,
                    device=device,
                ),
                scale=0.125,
                num_kv_heads=8,
                block_size=16,
                head_dim=128,
                max_query_len=num_requests,
                max_seq_len=common_prefix_len,
                causal=False,
                block_tables_are_staging_slots=True,
            )
        )
    finally:
        manager.release_cached_prefill(local_block_table, valid_rows)
    hydrated_candidate = torch.empty_like(query)
    byte_v2_attn_module.merge_attn_states(
        hydrated_candidate,
        hydrated_prefix_output,
        hydrated_prefix_lse,
        suffix_output,
        suffix_lse,
    )
    torch.accelerator.synchronize(device)

    torch.testing.assert_close(candidate, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        hydrated_candidate,
        reference,
        atol=1e-2,
        rtol=1e-2,
    )
    assert hybrid_state.fatal.item() == 0
    assert bool((workspace.block_to_staging_slot == -1).all())
    assert workspace.next_staging_slot.item() == 0
    assert workspace.overflow.item() == 0
