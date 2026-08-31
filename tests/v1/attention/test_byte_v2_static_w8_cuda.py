# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.byte_v2_layout import (
    ByteV2PageLayoutV6,
    ByteV2RawStagingLayout,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_hydrate_raw_staging_from_hybrid_cache,
    byte_v2_static_w8_writer_is_available,
    byte_v2_update_hybrid_cache_raw_staging_multi_token,
)

_POLICY = (16, 16, 16, 64, 128, 128)


def _state(*, pages: int, raw_slots: int, device: torch.device):
    raw_layout = ByteV2RawStagingLayout()
    return SimpleNamespace(
        transient=torch.zeros(
            (pages, raw_layout.slot_size_bytes),
            dtype=torch.uint8,
            device=device,
        ),
        persistent=torch.zeros(
            (raw_slots, raw_layout.slot_size_bytes),
            dtype=torch.uint8,
            device=device,
        ),
        page_to_raw=torch.full((pages,), -1, dtype=torch.int32, device=device),
        block_to_staging=torch.full((pages,), -1, dtype=torch.int32, device=device),
        staging_to_block=torch.full((pages,), -1, dtype=torch.int32, device=device),
        valid_rows=torch.zeros((pages,), dtype=torch.int32, device=device),
        next_slot=torch.zeros((1,), dtype=torch.int32, device=device),
        staging_overflow=torch.zeros((1,), dtype=torch.int32, device=device),
        free_slots=torch.arange(
            raw_slots - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=device,
        ),
        free_count=torch.full((1,), raw_slots, dtype=torch.int32, device=device),
        fatal=torch.zeros((1,), dtype=torch.int32, device=device),
    )


def _write(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    static_bases: tuple[int, int],
):
    page_layout = ByteV2PageLayoutV6()
    pages = key.shape[0] // 16
    state = _state(pages=pages, raw_slots=pages, device=key.device)
    cache = torch.zeros(
        (pages, page_layout.page_size_bytes),
        dtype=torch.uint8,
        device=key.device,
    )
    slots = torch.arange(key.shape[0], dtype=torch.int64, device=key.device)
    byte_v2_update_hybrid_cache_raw_staging_multi_token(
        key,
        value,
        state.transient,
        cache,
        state.persistent,
        slots,
        state.block_to_staging,
        state.staging_to_block,
        state.valid_rows,
        state.next_slot,
        state.staging_overflow,
        state.page_to_raw,
        state.free_slots,
        state.free_count,
        state.fatal,
        tile_policy=_POLICY,
        static_k_high7_base=static_bases[0],
        static_v_high7_base=static_bases[1],
    )
    return cache, state


def _hydrate(cache: torch.Tensor, state) -> tuple[torch.Tensor, torch.Tensor]:
    raw_layout = ByteV2RawStagingLayout()
    pages = cache.shape[0]
    decoded = torch.zeros(
        (pages, raw_layout.slot_size_bytes),
        dtype=torch.uint8,
        device=cache.device,
    )
    staging_to_block = torch.arange(pages, dtype=torch.int32, device=cache.device)
    valid_rows = torch.full((pages,), 16, dtype=torch.int32, device=cache.device)
    byte_v2_hydrate_raw_staging_from_hybrid_cache(
        decoded,
        cache,
        state.persistent,
        state.page_to_raw,
        staging_to_block,
        valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    decoded = decoded.view(torch.bfloat16).reshape(pages, 2, 8, 16, 128)
    key = decoded[:, 0].permute(0, 2, 1, 3).reshape(-1, 8, 128)
    value = decoded[:, 1].permute(0, 2, 1, 3).reshape(-1, 8, 128)
    return key, value


def _assert_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    assert torch.equal(left.view(torch.int16), right.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_w8_writer_roundtrip_matches_dynamic_v6():
    if not byte_v2_static_w8_writer_is_available():
        pytest.skip("Static-W8 writer schema is not registered")
    torch.manual_seed(17)
    key = torch.randn((16, 8, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key) * 0.25

    dynamic_cache, dynamic_state = _write(key, value, static_bases=(-1, -1))
    static_cache, static_state = _write(key, value, static_bases=(58, 56))
    dynamic_key, dynamic_value = _hydrate(dynamic_cache, dynamic_state)
    static_key, static_value = _hydrate(static_cache, static_state)

    _assert_bitwise_equal(dynamic_key, key)
    _assert_bitwise_equal(dynamic_value, value)
    _assert_bitwise_equal(static_key, key)
    _assert_bitwise_equal(static_value, value)
    _assert_bitwise_equal(static_key, dynamic_key)
    _assert_bitwise_equal(static_value, dynamic_value)
    assert int(static_state.page_to_raw.cpu()[0]) == -1
    assert int(static_state.fatal.cpu()[0]) == 0

    layout = ByteV2PageLayoutV6()
    cache_cpu = static_cache.cpu()
    overflow = int.from_bytes(
        cache_cpu[
            0,
            layout.outlier_pool_overflow_offset : (
                layout.outlier_pool_overflow_offset + 4
            ),
        ].numpy(),
        byteorder="little",
    )
    assert overflow == 0
    for head in range(8):
        for dim_tile in range(8):
            assert (
                cache_cpu[
                    0,
                    layout.k_base_offset(
                        kv_head=head,
                        dim_tile=dim_tile,
                        token_tile=0,
                    ),
                ]
                == 58
            )
            assert (
                cache_cpu[
                    0,
                    layout.v_base_offset(
                        kv_head=head,
                        dim_tile=dim_tile,
                        token_tile=0,
                    ),
                ]
                == 56
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_w8_pool_overflow_uses_exact_raw_sidecar():
    if not byte_v2_static_w8_writer_is_available():
        pytest.skip("Static-W8 writer schema is not registered")
    indices = torch.arange(16 * 8 * 128, dtype=torch.int32, device="cuda")
    low = indices.reshape(16, 8, 128) % 251
    key_high7 = torch.full_like(low, 58)
    value_high7 = torch.full_like(low, 56)
    token_in_tile = torch.arange(16, device="cuda")[:, None, None]
    dim_in_tile = torch.arange(128, device="cuda")[None, None, :] % 16
    # Seventeen outliers per V tile stay below the 128-entry tile limit,
    # while 8 heads * 8 dim tiles * 17 = 1088 exceeds the V6 page pool.
    value_high7 = torch.where(
        token_in_tile * 16 + dim_in_tile < 17,
        64,
        value_high7,
    )
    key = ((key_high7 << 8) | low).to(torch.int16).view(torch.bfloat16)
    value = ((value_high7 << 8) | ((low + 17) % 251)).to(torch.int16)
    value = value.view(torch.bfloat16)

    cache, state = _write(key, value, static_bases=(58, 56))
    decoded_key, decoded_value = _hydrate(cache, state)

    _assert_bitwise_equal(decoded_key, key)
    _assert_bitwise_equal(decoded_value, value)
    assert int(state.page_to_raw.cpu()[0]) == 0
    assert int(state.free_count.cpu()[0]) == 0
    assert int(state.fatal.cpu()[0]) == 0
    layout = ByteV2PageLayoutV6()
    cache_cpu = cache.cpu()
    overflow = int.from_bytes(
        cache_cpu[
            0,
            layout.outlier_pool_overflow_offset : (
                layout.outlier_pool_overflow_offset + 4
            ),
        ].numpy(),
        byteorder="little",
    )
    assert overflow != 0
