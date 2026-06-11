# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.byte_v2_codec import (
    BYTE_V2_FAST_TILE_PAYLOAD_BYTES,
    BYTE_V2_TILE_ELEMS,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    ByteV2PageLayout,
    byte_v2_reshape_and_cache_ref,
    count_byte_v2_page_statuses,
    pack_byte_v2_kv_block_to_page,
    pack_byte_v2_raw_kv_block_to_page,
    unpack_byte_v2_kv_block_from_page,
    unpack_byte_v2_raw_kv_block_from_page,
)
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec


def _assert_bf16_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def _bf16_from_u16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def test_byte_v2_page_layout_offsets_match_spec():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
    )
    layout = ByteV2PageLayout.from_spec(spec)

    assert layout.page_size_bytes == spec.page_size_bytes
    compressed_bytes = 2 * (2 + 1) * BYTE_V2_FAST_TILE_PAYLOAD_BYTES
    raw_bytes = 16 * 2 * (32 + 16) * 2
    assert layout.page_size_bytes == 16 + max(compressed_bytes, raw_bytes)

    first = layout.tile_offsets("k", kv_head=0, dim_tile=0)
    assert first.base == 16
    assert first.fallback == 17
    assert first.low_bytes == 18
    assert first.code_packed == 18 + BYTE_V2_TILE_ELEMS

    last = layout.tile_offsets("v", kv_head=1, dim_tile=0)
    assert last.base == 16 + 5 * BYTE_V2_FAST_TILE_PAYLOAD_BYTES
    assert last.code_packed + 128 == 16 + compressed_bytes


def test_byte_v2_page_layout_raw_fallback_round_trip():
    torch.manual_seed(5)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(8, 2, 32)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(8, 2, 16)).to(torch.bfloat16)

    page = pack_byte_v2_raw_kv_block_to_page(key, value, 8, layout)
    key_out, value_out = unpack_byte_v2_raw_kv_block_from_page(page, layout)

    assert int(page[BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8
    _assert_bf16_bits_equal(key_out[:8], key)
    _assert_bf16_bits_equal(value_out[:8], value)
    assert torch.equal(key_out[8:], torch.zeros_like(key_out[8:]))
    assert torch.equal(value_out[8:], torch.zeros_like(value_out[8:]))


def test_byte_v2_page_layout_pack_unpack_fast_path():
    torch.manual_seed(0)
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
    )
    layout = ByteV2PageLayout.from_spec(spec)
    key = (1.0 + 0.01 * torch.randn(16, 2, 32)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 2, 16)).to(torch.bfloat16)

    page = pack_byte_v2_kv_block_to_page(key, value, layout)
    key_out, value_out = unpack_byte_v2_kv_block_from_page(page, layout)

    assert page.dtype == torch.uint8
    assert tuple(page.shape) == (spec.page_size_bytes,)
    assert (
        int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
        == BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    _assert_bf16_bits_equal(key_out, key)
    _assert_bf16_bits_equal(value_out, value)


def test_byte_v2_page_status_counts_distinguish_page_states():
    torch.manual_seed(11)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    kv_cache = torch.zeros(5, layout.page_size_bytes, dtype=torch.uint8)
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)

    pack_byte_v2_kv_block_to_page(key, value, layout, page=kv_cache[0])
    pack_byte_v2_raw_kv_block_to_page(
        key[:8], value[:8], 8, layout, page=kv_cache[1]
    )
    pack_byte_v2_raw_kv_block_to_page(key, value, 16, layout, page=kv_cache[2])
    kv_cache[3, BYTE_V2_PAGE_STATUS_OFFSET] = 99

    counts = count_byte_v2_page_statuses(kv_cache, layout)

    assert counts.compressed == 1
    assert counts.partial_raw == 1
    assert counts.full_raw_fallback == 1
    assert counts.empty == 1
    assert counts.invalid == 1
    assert counts.raw_pages == 2


def test_byte_v2_page_status_counts_accept_partial_compressed_page():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)
    kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_COMPRESSED
    kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET] = 8

    counts = count_byte_v2_page_statuses(kv_cache, layout)

    assert counts.partial_compressed == 1
    assert counts.compressed == 0
    assert counts.invalid == 0


def test_byte_v2_reshape_and_cache_ref_packs_full_blocks():
    torch.manual_seed(1)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(32, 2, 32)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(32, 2, 16)).to(torch.bfloat16)
    kv_cache = torch.zeros(2, layout.page_size_bytes, dtype=torch.uint8)
    slot_mapping = torch.arange(32, dtype=torch.int64)

    packed = byte_v2_reshape_and_cache_ref(
        key, value, kv_cache, slot_mapping, layout
    )

    assert packed == [0, 1]
    key_0, value_0 = unpack_byte_v2_kv_block_from_page(kv_cache[0], layout)
    key_1, value_1 = unpack_byte_v2_kv_block_from_page(kv_cache[1], layout)
    _assert_bf16_bits_equal(key_0, key[:16])
    _assert_bf16_bits_equal(value_0, value[:16])
    _assert_bf16_bits_equal(key_1, key[16:])
    _assert_bf16_bits_equal(value_1, value[16:])


def test_byte_v2_reshape_and_cache_ref_reorders_by_slot_offset():
    torch.manual_seed(2)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_sorted = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value_sorted = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    order = torch.tensor([7, 2, 15, 0, 4, 12, 1, 11, 6, 14, 9, 3, 5, 8, 10, 13])
    key = key_sorted[order]
    value = value_sorted[order]
    slot_mapping = 16 + order
    kv_cache = torch.zeros(2, layout.page_size_bytes, dtype=torch.uint8)

    packed = byte_v2_reshape_and_cache_ref(
        key, value, kv_cache, slot_mapping, layout
    )

    assert packed == [1]
    key_out, value_out = unpack_byte_v2_kv_block_from_page(kv_cache[1], layout)
    _assert_bf16_bits_equal(key_out, key_sorted)
    _assert_bf16_bits_equal(value_out, value_sorted)


def test_byte_v2_reshape_and_cache_ref_writes_partial_raw_blocks():
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    value = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)

    packed = byte_v2_reshape_and_cache_ref(
        key, value, kv_cache, torch.arange(8), layout
    )

    assert packed == []
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8
    key_out, value_out = unpack_byte_v2_raw_kv_block_from_page(kv_cache[0], layout)
    _assert_bf16_bits_equal(key_out[:8], key)
    _assert_bf16_bits_equal(value_out[:8], value)


def test_byte_v2_reshape_and_cache_ref_finalizes_existing_raw_block():
    torch.manual_seed(6)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)

    first = byte_v2_reshape_and_cache_ref(
        key[:8], value[:8], kv_cache, torch.arange(8), layout
    )
    second = byte_v2_reshape_and_cache_ref(
        key[8:], value[8:], kv_cache, torch.arange(8, 16), layout
    )

    assert first == []
    assert second == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    key_out, value_out = unpack_byte_v2_kv_block_from_page(kv_cache[0], layout)
    _assert_bf16_bits_equal(key_out, key)
    _assert_bf16_bits_equal(value_out, value)


def test_byte_v2_page_layout_uses_raw_fallback_for_uncompressible_tile():
    exp = (torch.arange(BYTE_V2_TILE_ELEMS, dtype=torch.int32) % 32) + 40
    low = torch.arange(BYTE_V2_TILE_ELEMS, dtype=torch.int32) & 0x7F
    key = _bf16_from_u16((exp << 7) | low, (16, 1, 16))
    value = torch.ones(16, 1, 16, dtype=torch.bfloat16)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )

    page = pack_byte_v2_kv_block_to_page(key, value, layout)
    key_out, value_out = unpack_byte_v2_raw_kv_block_from_page(page, layout)

    assert int(page[BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    _assert_bf16_bits_equal(key_out, key)
    _assert_bf16_bits_equal(value_out, value)


def test_byte_v2_page_layout_validates_shapes():
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = torch.ones(16, 1, 16, dtype=torch.bfloat16)
    value = torch.ones(16, 1, 16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="key_block"):
        pack_byte_v2_kv_block_to_page(key[:, :, :8], value, layout)

    with pytest.raises(ValueError, match="value_block"):
        pack_byte_v2_kv_block_to_page(key, value.to(torch.float32), layout)

    with pytest.raises(ValueError, match="uint8"):
        unpack_byte_v2_kv_block_from_page(torch.zeros(layout.page_size_bytes), layout)
