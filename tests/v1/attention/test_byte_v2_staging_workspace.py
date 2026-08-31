# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.attention import (
    bind_byte_v2_raw_staging_workspace,
)
from vllm.v1.attention.backends.byte_v2_attn import (
    ByteV2RawFallbackStore,
    ByteV2RawStagingWorkspaceSpec,
    plan_byte_v2_raw_staging_waves,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    DEFAULT_BYTE_V2_TILE_POLICY,
    ByteV2RawStagingLayout,
)

pytestmark = pytest.mark.cpu_test


def _metadata(
    query_start_locs,
    num_actual_tokens=None,
    *,
    seq_lens_cpu_upper_bound=None,
    is_prefilling=None,
):
    if num_actual_tokens is None:
        num_actual_tokens = query_start_locs[-1]
    return SimpleNamespace(
        num_actual_tokens=num_actual_tokens,
        query_start_loc_cpu=torch.tensor(query_start_locs, dtype=torch.int32),
        seq_lens_cpu_upper_bound=(
            None
            if seq_lens_cpu_upper_bound is None
            else torch.tensor(seq_lens_cpu_upper_bound, dtype=torch.int32)
        ),
        is_prefilling=(
            None
            if is_prefilling is None
            else torch.tensor(is_prefilling, dtype=torch.bool)
        ),
    )


def test_byte_v2_raw_staging_workspace_exact_size_and_initial_state():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=512,
        num_staging_slots=128,
        slot_size_bytes=65_536,
        device=torch.device("cpu"),
    )

    workspace = spec.allocate()

    assert spec.nbytes == 65_536 * 128 + 4 * 512 + 8 * 128 + 8
    assert workspace.raw_staging.shape == (128, 65_536)
    assert workspace.block_to_staging_slot.tolist() == [-1] * 512
    assert workspace.staging_to_physical_block.tolist() == [-1] * 128
    assert not workspace.valid_rows.any()
    assert workspace.next_staging_slot.item() == 0
    assert workspace.overflow.item() == 0


def test_byte_v2_raw_fallback_uses_plan_and_rejects_cache_rebind():
    store = ByteV2RawFallbackStore(
        raw_layout=ByteV2RawStagingLayout(
            tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
            num_kv_heads=8,
        )
    )
    store.bind_plan(num_blocks=2, num_raw_slots=3)
    first_cache = torch.empty((2, 1), dtype=torch.uint8)
    second_cache = torch.empty((2, 1), dtype=torch.uint8)

    state = store.state(first_cache)

    assert state.raw_pages.shape[0] == 3
    with pytest.raises(RuntimeError, match="binding changed"):
        store.state(second_cache)


def test_byte_v2_page_aware_wave_packs_one_long_request():
    waves = plan_byte_v2_raw_staging_waves(
        1024,
        128,
        _metadata([0, 1024]),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 1024, 65)
    ]


def test_byte_v2_page_aware_wave_splits_16k_into_nine_waves():
    waves = plan_byte_v2_raw_staging_waves(
        16_384,
        128,
        _metadata([0, 16_384]),
    )

    assert len(waves) == 9
    assert waves[0].start == 0
    assert waves[-1].end == 16_384
    assert all(wave.max_unique_pages <= 128 for wave in waves)
    assert all(left.end == right.start for left, right in zip(waves, waves[1:]))


def test_byte_v2_initial_prefill_uses_exact_page_aligned_waves():
    waves = plan_byte_v2_raw_staging_waves(
        4096,
        128,
        _metadata(
            [0, 4096],
            seq_lens_cpu_upper_bound=[4096],
        ),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 2048, 128),
        (2048, 4096, 128),
    ]


def test_byte_v2_initial_prefill_production_shape_uses_eight_waves():
    query_start_locs = [0, 4096, 8192, 12_288, 16_384]
    waves = plan_byte_v2_raw_staging_waves(
        16_384,
        128,
        _metadata(
            query_start_locs,
            seq_lens_cpu_upper_bound=[4096] * 4,
        ),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 2048, 128),
        (2048, 4096, 128),
        (4096, 6144, 128),
        (6144, 8192, 128),
        (8192, 10_240, 128),
        (10_240, 12_288, 128),
        (12_288, 14_336, 128),
        (14_336, 16_384, 128),
    ]


def test_byte_v2_mixed_cached_and_initial_rows_keep_individual_bounds():
    waves = plan_byte_v2_raw_staging_waves(
        4112,
        128,
        _metadata(
            [0, 16, 4112],
            seq_lens_cpu_upper_bound=[32, 4096],
        ),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 16, 2),
        (16, 2064, 128),
        (2064, 4112, 128),
    ]


def test_byte_v2_zero_length_padding_does_not_add_a_wave():
    waves = plan_byte_v2_raw_staging_waves(
        4096,
        128,
        _metadata(
            [0, 4096, 4096],
            seq_lens_cpu_upper_bound=[4096, 0],
        ),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 2048, 128),
        (2048, 4096, 128),
    ]


def test_byte_v2_cached_row_keeps_conservative_page_bound():
    waves = plan_byte_v2_raw_staging_waves(
        4096,
        128,
        _metadata(
            [0, 4096],
            seq_lens_cpu_upper_bound=[4112],
        ),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 2033, 128),
        (2033, 4066, 128),
        (4066, 4096, 3),
    ]


def test_byte_v2_marks_only_page_aligned_prefill_waves_as_overwriting():
    waves = plan_byte_v2_raw_staging_waves(
        34,
        2,
        _metadata(
            [0, 34],
            seq_lens_cpu_upper_bound=[50],
            is_prefilling=[True],
        ),
    )

    assert [
        (
            wave.start,
            wave.end,
            wave.max_unique_pages,
            wave.overwrites_all_valid_rows,
        )
        for wave in waves
    ] == [
        (0, 17, 2, True),
        (17, 34, 2, False),
    ]


@pytest.mark.parametrize(
    ("is_prefilling", "seq_len", "expected"),
    [
        ([True], 32, True),
        ([True], 17, False),
        ([True], 15, False),
        ([False], 32, False),
        (None, 32, False),
    ],
)
def test_byte_v2_page_aligned_overwrite_proof_fails_closed(
    is_prefilling,
    seq_len,
    expected,
):
    waves = plan_byte_v2_raw_staging_waves(
        16,
        2,
        _metadata(
            [0, 16],
            seq_lens_cpu_upper_bound=[seq_len],
            is_prefilling=is_prefilling,
        ),
    )

    assert len(waves) == 1
    assert waves[0].overwrites_all_valid_rows is expected


@pytest.mark.parametrize(
    ("seq_lens", "expected"),
    [
        ([20, 20], True),
        ([20, 5], False),
    ],
)
def test_byte_v2_merged_wave_requires_every_request_segment_to_be_aligned(
    seq_lens,
    expected,
):
    waves = plan_byte_v2_raw_staging_waves(
        8,
        4,
        _metadata(
            [0, 4, 8],
            seq_lens_cpu_upper_bound=seq_lens,
            is_prefilling=[True, True],
        ),
    )

    assert len(waves) == 1
    assert (waves[0].start, waves[0].end) == (0, 8)
    assert waves[0].overwrites_all_valid_rows is expected


def test_byte_v2_page_aware_wave_bounds_many_one_token_requests():
    query_start_locs = list(range(130))

    waves = plan_byte_v2_raw_staging_waves(
        129,
        128,
        _metadata(query_start_locs),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 128, 128),
        (128, 129, 1),
    ]


def test_byte_v2_untrusted_metadata_uses_one_page_per_token_bound():
    waves = plan_byte_v2_raw_staging_waves(
        300,
        128,
        _metadata([1, 300]),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 128, 128),
        (128, 256, 128),
        (256, 300, 44),
    ]


class _WorkspaceImpl:
    def __init__(self, spec):
        self.spec = spec
        self.workspace = None
        self.raw_plan = None
        self.raw_state_cache = None
        self.decoded_prefix_cache_plan = None
        self.decoded_prefix_cache_bytes = 0
        self.hybrid_raw_mutable_tail_q1 = False
        self.fa2_hybrid_raw_fallback = True

    def raw_staging_workspace_spec(self, kv_cache, num_staging_slots):
        assert kv_cache.device == self.spec.device
        assert num_staging_slots == self.spec.num_staging_slots
        return self.spec

    def bind_raw_staging_workspace(self, workspace):
        self.workspace = workspace

    def bind_raw_fallback_plan(self, *, num_blocks, num_raw_slots):
        self.raw_plan = (num_blocks, num_raw_slots)

    def initialize_raw_fallback_state(self, kv_cache):
        self.raw_state_cache = kv_cache

    def bind_decoded_prefix_cache_plan(self, kv_cache, num_cache_slots):
        self.decoded_prefix_cache_plan = (kv_cache, num_cache_slots)
        self.decoded_prefix_cache_bytes = num_cache_slots * (
            self.spec.slot_size_bytes + 8
        )

    def decoded_prefix_cache_nbytes(self):
        return self.decoded_prefix_cache_bytes


def test_bind_byte_v2_raw_staging_workspace_shares_one_allocation():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=4,
        num_staging_slots=2,
        slot_size_bytes=64,
        device=torch.device("cpu"),
    )
    impls = [_WorkspaceImpl(spec), _WorkspaceImpl(spec)]
    context = {
        str(index): SimpleNamespace(
            impl=impl,
            kv_cache=torch.empty((4, 1), dtype=torch.uint8),
        )
        for index, impl in enumerate(impls)
    }
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=2,
        byte_v2_raw_staging_workspace_bytes=spec.nbytes,
    )

    workspace = bind_byte_v2_raw_staging_workspace(
        context,
        config,
        use_ubatching=False,
    )

    assert workspace is not None
    assert impls[0].workspace is workspace
    assert impls[1].workspace is workspace
    assert impls[0].raw_plan == (4, 1)
    assert impls[1].raw_plan == (4, 1)
    assert impls[0].raw_state_cache is context["0"].kv_cache
    assert impls[1].raw_state_cache is context["1"].kv_cache


def test_bind_byte_v2_raw_staging_workspace_accounts_decoded_prefix_cache():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=4,
        num_staging_slots=2,
        slot_size_bytes=64,
        device=torch.device("cpu"),
    )
    impls = [_WorkspaceImpl(spec), _WorkspaceImpl(spec)]
    context = {
        str(index): SimpleNamespace(
            impl=impl,
            kv_cache=torch.empty((4, 1), dtype=torch.uint8),
        )
        for index, impl in enumerate(impls)
    }
    cache_slots = 3
    cache_bytes = len(impls) * cache_slots * (spec.slot_size_bytes + 8)
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=2,
        byte_v2_raw_staging_workspace_bytes=spec.nbytes,
        byte_v2_decoded_prefix_cache_slots=cache_slots,
        byte_v2_decoded_prefix_cache_bytes=cache_bytes,
    )

    workspace = bind_byte_v2_raw_staging_workspace(
        context,
        config,
        use_ubatching=False,
    )

    assert workspace is not None
    plans = [impl.decoded_prefix_cache_plan for impl in impls]
    assert all(plan is not None for plan in plans)
    assert [plan[1] for plan in plans if plan is not None] == [3, 3]
    assert sum(impl.decoded_prefix_cache_bytes for impl in impls) == cache_bytes


def test_bind_byte_v2_raw_staging_workspace_rejects_capability_mismatch():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=4,
        num_staging_slots=2,
        slot_size_bytes=64,
        device=torch.device("cpu"),
    )
    impl = _WorkspaceImpl(spec)
    impl.hybrid_raw_mutable_tail_q1 = True
    context = {
        "layer": SimpleNamespace(
            impl=impl,
            kv_cache=torch.empty((4, 1), dtype=torch.uint8),
        )
    }
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=2,
        byte_v2_raw_staging_workspace_bytes=spec.nbytes,
        byte_v2_raw_mutable_tail_q1=False,
    )

    with pytest.raises(RuntimeError, match="capability mismatch"):
        bind_byte_v2_raw_staging_workspace(
            context,
            config,
            use_ubatching=False,
        )


def test_bind_byte_v2_raw_staging_workspace_rejects_compact_only_v6():
    impl = _WorkspaceImpl(spec=None)
    impl.fa2_hybrid_raw_fallback = False
    context = {
        "layer": SimpleNamespace(
            impl=impl,
            kv_cache=torch.empty((4, 50_560), dtype=torch.uint8),
        )
    }
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=0,
        byte_v2_raw_staging_slots=0,
        byte_v2_raw_staging_workspace_bytes=0,
    )

    with pytest.raises(RuntimeError, match="cannot bind a compact-only engine"):
        bind_byte_v2_raw_staging_workspace(
            context,
            config,
            use_ubatching=False,
        )


def test_bind_byte_v2_raw_staging_workspace_rejects_ubatching():
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=128,
        byte_v2_raw_staging_workspace_bytes=1,
    )

    with pytest.raises(RuntimeError, match="single-lane"):
        bind_byte_v2_raw_staging_workspace(
            {},
            config,
            use_ubatching=True,
        )
