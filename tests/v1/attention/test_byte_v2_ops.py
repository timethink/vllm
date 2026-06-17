# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET,
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3,
    ByteV2PageLayout,
    ByteV2PageLayoutV3,
    unpack_byte_v2_kv_block_from_page,
)
from vllm.v1.attention.backends.byte_v2_ops import byte_v2_reshape_and_cache
from vllm.v1.attention.backends.byte_v2_torch import (
    count_byte_v2_page_statuses_torch,
)


def _assert_bf16_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def _raw_decode_single_block(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_decode_tokens, num_heads, _ = query.shape
    q_per_kv = num_heads // key.shape[1]
    output = torch.empty(
        num_decode_tokens, num_heads, value.shape[2], dtype=query.dtype
    )
    for req_idx in range(num_decode_tokens):
        for q_head in range(num_heads):
            kv_head = q_head // q_per_kv
            scores = torch.matmul(
                key[:, kv_head].to(torch.float32),
                query[req_idx, q_head].to(torch.float32),
            )
            probs = torch.softmax(scores * scale, dim=0)
            output[req_idx, q_head] = torch.matmul(
                probs, value[:, kv_head].to(torch.float32)
            ).to(query.dtype)
    return output


def _raw_decode_batched_blocks(
    query: torch.Tensor,
    key_by_request: torch.Tensor,
    value_by_request: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_decode_tokens, num_heads, _ = query.shape
    q_per_kv = num_heads // key_by_request.shape[2]
    output = torch.empty(
        num_decode_tokens, num_heads, value_by_request.shape[3], dtype=query.dtype
    )
    for req_idx in range(num_decode_tokens):
        for q_head in range(num_heads):
            kv_head = q_head // q_per_kv
            scores = torch.matmul(
                key_by_request[req_idx, :, kv_head].to(torch.float32),
                query[req_idx, q_head].to(torch.float32),
            )
            probs = torch.softmax(scores * scale, dim=0)
            output[req_idx, q_head] = torch.matmul(
                probs,
                value_by_request[req_idx, :, kv_head].to(torch.float32),
            ).to(query.dtype)
    return output


def test_byte_v2_reshape_and_cache_op_packs_full_block():
    torch.manual_seed(3)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)
    slot_mapping = torch.arange(16, dtype=torch.int64)

    packed = byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed.dtype == torch.int64
    assert packed.tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    key_out, value_out = unpack_byte_v2_kv_block_from_page(kv_cache[0], layout)
    _assert_bf16_bits_equal(key_out, key)
    _assert_bf16_bits_equal(value_out, value)


def test_byte_v2_reshape_and_cache_op_writes_partial_raw_block():
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    value = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)

    packed = byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(8, dtype=torch.int64),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed.dtype == torch.int64
    assert packed.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )


def test_byte_v2_custom_ops_wrapper_uses_registered_reference_op():
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    value = torch.ones(8, 1, 16, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(8, dtype=torch.int64),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed.dtype == torch.int64
    assert packed.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_reshape_and_cache_op_packs_full_block_cuda():
    torch.manual_seed(13)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed.dtype == torch.int64
    assert packed.tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.compressed == 1
    assert counts.raw_pages == 0
    assert counts.invalid == 0
    key_out, value_out = unpack_byte_v2_kv_block_from_page(
        kv_cache[0].cpu(), layout
    )
    _assert_bf16_bits_equal(key_out, key.cpu())
    _assert_bf16_bits_equal(value_out, value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_reshape_and_cache_op_falls_back_for_split_direct_group_cuda(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC", "1")
    torch.manual_seed(17)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (2, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    slot_mapping = torch.cat(
        [
            torch.arange(8, dtype=torch.int64),
            torch.arange(16, 24, dtype=torch.int64),
        ]
    ).cuda()

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )
    torch.cuda.synchronize()

    assert packed.numel() == 0
    assert fallback_block_ids.cpu().tolist() == [-1, -1]
    assert fallback_next_slot.cpu().tolist() == [0]
    assert fallback_tile_next_slot.cpu().tolist() == [0]
    kv_cache_cpu = kv_cache.cpu()
    assert int(kv_cache_cpu[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache_cpu[1, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache_cpu[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8
    assert int(kv_cache_cpu[1, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8

    key_0, value_0 = unpack_byte_v2_kv_block_from_page(kv_cache_cpu[0], layout)
    key_1, value_1 = unpack_byte_v2_kv_block_from_page(kv_cache_cpu[1], layout)
    _assert_bf16_bits_equal(key_0[:8], key[:8].cpu())
    _assert_bf16_bits_equal(value_0[:8], value[:8].cpu())
    _assert_bf16_bits_equal(key_1[:8], key[8:].cpu())
    _assert_bf16_bits_equal(value_1[:8], value[8:].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_decompress_cache_to_bf16_op_cuda():
    torch.manual_seed(23)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(16, 2, 32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 2, 32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )
    assert packed.tolist() == [0]

    key_cache = torch.empty(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )
    torch.cuda.synchronize()

    _assert_bf16_bits_equal(key_cache[0].cpu(), key.cpu())
    _assert_bf16_bits_equal(value_cache[0].cpu(), value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_v3_native_cache_update_decompress_and_cute_decode_cuda(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1", "1")
    monkeypatch.delenv("VLLM_BYTE_V2_DECODE_FAST_STAGE1", raising=False)
    monkeypatch.delenv("VLLM_BYTE_V2_DECODE_FLASH_STAGE1", raising=False)
    torch.manual_seed(31)
    layout = ByteV2PageLayoutV3(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
    )
    num_blocks = 16
    num_tokens = num_blocks * 16
    key = (1.0 + 0.001 * torch.randn(num_tokens, 8, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.001 * torch.randn(num_tokens, 8, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        num_blocks, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        4, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full(
        (num_blocks,), -1, dtype=torch.int32, device="cuda"
    )
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (num_blocks, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(num_tokens, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )
    torch.cuda.synchronize()

    assert packed.cpu().tolist() == list(range(num_blocks))
    assert fallback_block_ids.cpu().tolist() == [-1] * num_blocks
    assert fallback_next_slot.cpu().tolist() == [0]
    assert fallback_tile_next_slot.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    assert int(kv_cache[0, BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET].item()) == (
        BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3
    )

    key_cache = torch.empty(
        num_blocks, 16, 8, 128, device="cuda", dtype=torch.bfloat16
    )
    value_cache = torch.empty(
        num_blocks, 16, 8, 128, device="cuda", dtype=torch.bfloat16
    )
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
    )
    torch.cuda.synchronize()
    _assert_bf16_bits_equal(
        key_cache.reshape(num_tokens, 8, 128).cpu(), key.cpu()
    )
    _assert_bf16_bits_equal(
        value_cache.reshape(num_tokens, 8, 128).cpu(), value.cpu()
    )

    query = (0.5 + 0.001 * torch.randn(1, 32, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(1, -1),
        torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
    )
    expected = _raw_decode_batched_blocks(
        query.cpu(), key.cpu().unsqueeze(0), value.cpu().unsqueeze(0), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_v3_decode_append_finalizes_partial_block_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC", "1")
    torch.manual_seed(37)
    layout = ByteV2PageLayoutV3(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
    )
    key = (1.0 + 0.001 * torch.randn(16, 8, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.001 * torch.randn(16, 8, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        4, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    first = ops.byte_v2_reshape_and_cache(
        key[:15],
        value[:15],
        kv_cache,
        torch.arange(15, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )
    assert first.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 15

    second = ops.byte_v2_reshape_and_cache(
        key[15:],
        value[15:],
        kv_cache,
        torch.tensor([15], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )
    assert second.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16

    key_cache = torch.empty(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
    )
    torch.cuda.synchronize()
    _assert_bf16_bits_equal(key_cache[0].cpu(), key.cpu())
    _assert_bf16_bits_equal(value_cache[0].cpu(), value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_v3_decode_append_writes_outlier_arena_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE", "1")
    torch.manual_seed(41)
    layout = ByteV2PageLayoutV3(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
    )
    key = torch.ones(16, 1, 128, dtype=torch.float32)
    key[0, 0, 0] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = (2.0 + 0.001 * torch.randn(16, 1, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_arena = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    outlier_block_flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_tile_bitmap = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    outlier_tile_meta = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    outlier_next_entry = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = None
    for row in range(16):
        packed = ops.byte_v2_reshape_and_cache(
            key[row:row + 1],
            value[row:row + 1],
            kv_cache,
            torch.tensor([row], dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            head_size_v=128,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
            fallback_tile_ids=fallback_tile_ids,
            fallback_tile_next_slot=fallback_tile_next_slot,
            outlier_arena=outlier_arena,
            outlier_block_flags=outlier_block_flags,
            outlier_tile_bitmap=outlier_tile_bitmap,
            outlier_tile_meta=outlier_tile_meta,
            outlier_next_entry=outlier_next_entry,
        )

    assert packed is not None
    assert packed.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    assert int(kv_cache[0, BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET].item()) == (
        BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3
    )
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [1]
    assert fallback_tile_next_slot.cpu().tolist() == [0]
    assert fallback_tile_ids.cpu().tolist() == [[-1] * layout.total_tiles]
    assert outlier_block_flags.cpu().tolist() == [1]
    assert outlier_tile_bitmap.cpu().tolist() == [[1]]
    assert int(outlier_next_entry.cpu().item()) == 1

    meta = int(outlier_tile_meta[0, 0].cpu().item())
    assert meta >= 0
    offset = meta >> 8
    count = meta & 0xFF
    assert (offset, count) == (0, 1)
    entry = int(outlier_arena[0].cpu().item())
    raw_bits = int(
        key.cpu().contiguous().view(torch.int16).reshape(-1)[0].item()
    ) & 0xFFFF
    assert entry & 0xFF == 0
    assert (entry >> 8) & 0xFFFF == raw_bits

    key_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
        outlier_arena=outlier_arena,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    torch.cuda.synchronize()
    _assert_bf16_bits_equal(key_cache[0].cpu(), key.cpu())
    _assert_bf16_bits_equal(value_cache[0].cpu(), value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_v3_outlier_only_no_fallback_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1", "1")
    monkeypatch.delenv("VLLM_BYTE_V2_DECODE_FAST_STAGE1", raising=False)
    monkeypatch.delenv("VLLM_BYTE_V2_DECODE_FLASH_STAGE1", raising=False)
    layout = ByteV2PageLayoutV3(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
    )
    key = torch.ones(16, 1, 128, dtype=torch.float32)
    value = 2.0 * torch.ones(16, 1, 128, dtype=torch.float32)
    key[0, 0, 0] = 1.0e20
    value[0, 0, 5] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = value.to(device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    outlier_arena = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    outlier_block_flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_tile_bitmap = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    outlier_tile_meta = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    outlier_next_entry = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
        outlier_next_entry=outlier_next_entry,
    )
    torch.cuda.synchronize()

    assert packed.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    assert int(kv_cache[0, BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET].item()) == (
        BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3
    )
    assert outlier_block_flags.cpu().tolist() == [1]
    assert int(outlier_next_entry.cpu().item()) == 2
    assert int((outlier_tile_meta >= 0).sum().cpu().item()) == 2

    key_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        outlier_arena=outlier_arena,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    torch.cuda.synchronize()
    _assert_bf16_bits_equal(key_cache[0].cpu(), key.cpu())
    _assert_bf16_bits_equal(value_cache[0].cpu(), value.cpu())

    query = torch.zeros(1, 4, 128, device="cuda", dtype=torch.bfloat16)
    query[:, :, 0] = 1.0
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        1.0,
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    expected = _raw_decode_single_block(query.cpu(), key.cpu(), value.cpu(), 1.0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_v3_decode_append_preserves_partial_outliers_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE", "1")
    torch.manual_seed(43)
    layout = ByteV2PageLayoutV3(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
    )
    old_key = torch.ones(16, 1, 128, dtype=torch.float32)
    old_key[0, 0, 0] = 1.0e20
    old_key = old_key.to(device="cuda", dtype=torch.bfloat16)
    old_value = (2.0 + 0.001 * torch.randn(16, 1, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    new_key = (3.0 + 0.001 * torch.randn(1, 1, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    new_value = (4.0 + 0.001 * torch.randn(1, 1, 128)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_arena = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    outlier_block_flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_tile_bitmap = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    outlier_tile_meta = torch.full(
        (1, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    outlier_next_entry = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed_old = ops.byte_v2_reshape_and_cache(
        old_key,
        old_value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
        outlier_next_entry=outlier_next_entry,
    )
    assert packed_old.cpu().tolist() == [0]
    assert int(outlier_next_entry.cpu().item()) == 1

    kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET] = 15
    packed_new = ops.byte_v2_reshape_and_cache(
        new_key,
        new_value,
        kv_cache,
        torch.tensor([15], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
        outlier_next_entry=outlier_next_entry,
    )

    assert packed_new.cpu().tolist() == [0]
    assert int(outlier_next_entry.cpu().item()) == 2
    meta = int(outlier_tile_meta[0, 0].cpu().item())
    assert meta >> 8 == 1
    assert meta & 0xFF == 1

    expected_key = torch.cat([old_key[:15], new_key], dim=0)
    expected_value = torch.cat([old_value[:15], new_value], dim=0)
    key_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    ops.byte_v2_decompress_cache_to_bf16(
        kv_cache,
        key_cache,
        value_cache,
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
        outlier_arena=outlier_arena,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    torch.cuda.synchronize()
    _assert_bf16_bits_equal(key_cache[0].cpu(), expected_key.cpu())
    _assert_bf16_bits_equal(value_cache[0].cpu(), expected_value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_reshape_and_cache_op_overwrites_reused_full_block_cuda():
    torch.manual_seed(19)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    old_key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    old_value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    new_key = (3.0 + 0.01 * torch.randn(8, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    new_value = (4.0 + 0.01 * torch.randn(8, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    packed_old = ops.byte_v2_reshape_and_cache(
        old_key,
        old_value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )
    assert packed_old.tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16

    packed_new = ops.byte_v2_reshape_and_cache(
        new_key,
        new_value,
        kv_cache,
        torch.arange(8, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed_new.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.partial_compressed == 1
    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([8], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )
    expected = _raw_decode_single_block(
        query.cpu(), new_key.cpu(), new_value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_reshape_and_cache_op_finalizes_existing_raw_block_cuda():
    torch.manual_seed(17)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    first = ops.byte_v2_reshape_and_cache(
        key[:8],
        value[:8],
        kv_cache,
        torch.arange(8, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )
    assert first.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8

    second = ops.byte_v2_reshape_and_cache(
        key[8:],
        value[8:],
        kv_cache,
        torch.arange(8, 16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert second.dtype == torch.int64
    assert second.tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    key_out, value_out = unpack_byte_v2_kv_block_from_page(
        kv_cache[0].cpu(), layout
    )
    _assert_bf16_bits_equal(key_out, key.cpu())
    _assert_bf16_bits_equal(value_out, value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_reshape_and_cache_op_accepts_strided_kv_cuda():
    torch.manual_seed(19)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_base = (1.0 + 0.01 * torch.randn(16, 1, 32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value_base = (2.0 + 0.01 * torch.randn(16, 1, 32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    key = key_base[:, :, ::2]
    value = value_base[:, :, ::2]
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert packed.tolist() == [0]
    key_out, value_out = unpack_byte_v2_kv_block_from_page(
        kv_cache[0].cpu(), layout
    )
    _assert_bf16_bits_equal(key_out, key.cpu())
    _assert_bf16_bits_equal(value_out, value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_compressed_only_cache_update_and_decode_cuda():
    torch.manual_seed(23)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )

    first = ops.byte_v2_reshape_and_cache(
        key[:8],
        value[:8],
        kv_cache,
        torch.arange(8, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert first.numel() == 0
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.partial_compressed == 1
    assert counts.raw_pages == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 8

    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual_partial = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([8], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )
    expected_partial = _raw_decode_single_block(
        query.cpu(), key[:8].cpu(), value[:8].cpu(), 0.125
    )
    torch.testing.assert_close(actual_partial.cpu(), expected_partial)

    second = ops.byte_v2_reshape_and_cache(
        key[8:],
        value[8:],
        kv_cache,
        torch.arange(8, 16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    assert second.tolist() == [0]
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.compressed == 1
    assert counts.raw_pages == 0
    assert counts.invalid == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_prefill_direct_encode_full_blocks_cuda():
    torch.manual_seed(37)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(32, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(32, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(32, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.cpu().tolist() == [0, 1]
    assert fallback_block_ids.cpu().tolist() == [-1, -1]
    assert fallback_next_slot.cpu().tolist() == [0]
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.compressed == 2
    assert counts.raw_pages == 0
    assert counts.invalid == 0
    for block_id in range(2):
        key_out, value_out = unpack_byte_v2_kv_block_from_page(
            kv_cache[block_id].cpu(), layout
        )
        token_start = block_id * 16
        token_end = token_start + 16
        _assert_bf16_bits_equal(key_out, key[token_start:token_end].cpu())
        _assert_bf16_bits_equal(value_out, value[token_start:token_end].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_prefill_direct_skip_validation_sync_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC", "1")
    torch.manual_seed(41)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(32, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(32, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(32, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.cpu().tolist() == [0, 1]
    assert fallback_block_ids.cpu().tolist() == [-1, -1]
    assert fallback_next_slot.cpu().tolist() == [0]
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.compressed == 2
    assert counts.raw_pages == 0
    assert counts.invalid == 0
    for block_id in range(2):
        key_out, value_out = unpack_byte_v2_kv_block_from_page(
            kv_cache[block_id].cpu(), layout
        )
        token_start = block_id * 16
        token_end = token_start + 16
        _assert_bf16_bits_equal(key_out, key[token_start:token_end].cpu())
        _assert_bf16_bits_equal(value_out, value[token_start:token_end].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_sparse_fallback_pool_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(16, 1, 16, dtype=torch.float32)
    key.reshape(-1)[1::2] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    assert fallback_block_ids.cpu().tolist() == [0]
    assert fallback_next_slot.cpu().tolist() == [1]
    counts = count_byte_v2_page_statuses_torch(kv_cache, layout)
    assert counts.full_raw_fallback == 1
    assert counts.invalid == 0

    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), key.cpu(), value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_tile_fallback_pool_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(16, 1, 16, dtype=torch.float32)
    key.reshape(-1)[1::2] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )

    assert packed.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [0]
    assert fallback_tile_next_slot.cpu().tolist() == [1]
    assert fallback_tile_ids.cpu().tolist()[0][0] >= 0
    assert fallback_tile_ids.cpu().tolist()[0][1] == -1
    assert int(kv_cache[0, layout.tile_offsets("k", 0, 0).fallback].item()) == 1
    assert int(kv_cache[0, layout.tile_offsets("v", 0, 0).fallback].item()) == 0

    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), key.cpu(), value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_outlier_arena_prefill_direct_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE", "1")
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(16, 1, 16, dtype=torch.float32)
    key[0, 0, 0] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = torch.ones(16, 1, 16, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_arena = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    outlier_block_flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_tile_bitmap = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    outlier_tile_meta = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    outlier_next_entry = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
        outlier_next_entry=outlier_next_entry,
    )

    assert packed.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [0]
    assert fallback_tile_next_slot.cpu().tolist() == [0]
    assert fallback_tile_ids.cpu().tolist() == [[-1, -1]]
    assert outlier_block_flags.cpu().tolist() == [1]
    assert outlier_tile_bitmap.cpu().tolist() == [[1]]

    page = kv_cache[0].cpu()
    k_offsets = layout.tile_offsets("k", 0, 0)
    assert int(page[k_offsets.fallback].item()) == 0

    meta = int(outlier_tile_meta[0, 0].cpu().item())
    assert meta >= 0
    assert int(outlier_tile_meta[0, 1].cpu().item()) == -1
    assert int(outlier_next_entry.cpu().item()) == 1
    offset = meta >> 8
    count = meta & 0xFF
    assert (offset, count) == (0, 1)

    entry = int(outlier_arena[0].cpu().item())
    raw_bits = int(
        key.cpu().contiguous().view(torch.int16).reshape(-1)[0].item()
    ) & 0xFFFF
    assert entry & 0xFF == 0
    assert (entry >> 8) & 0xFFFF == raw_bits

    outlier_tile_meta[0, 1] = meta
    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        1.0,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    expected = _raw_decode_single_block(
        query.cpu(),
        key.cpu(),
        value.cpu(),
        1.0,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_outlier_arena_decode_overlay_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_TILE_FASTPATH", "1")
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    num_blocks = 20
    outlier_block = 7
    outlier_row = outlier_block * layout.block_size
    key = torch.ones(num_blocks * 16, 1, 16, dtype=torch.float32)
    value = (2.0 * torch.ones(num_blocks * 16, 1, 16)).to(torch.float32)
    key[outlier_row, 0, 0] = 1.0e20
    value[outlier_row, 0, 3] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = value.to(device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        num_blocks, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full(
        (num_blocks,), -1, dtype=torch.int32, device="cuda"
    )
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full(
        (num_blocks, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    outlier_arena = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    outlier_block_flags = torch.zeros(
        num_blocks, dtype=torch.int32, device="cuda"
    )
    outlier_tile_bitmap = torch.zeros(
        (num_blocks, 1), dtype=torch.int32, device="cuda"
    )
    outlier_tile_meta = torch.full(
        (num_blocks, layout.total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    outlier_next_entry = torch.zeros(1, dtype=torch.int32, device="cuda")

    ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(num_blocks * 16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
        outlier_next_entry=outlier_next_entry,
    )

    assert int(outlier_next_entry.cpu().item()) == 2
    assert int(fallback_next_slot.cpu().item()) == 0
    assert int(fallback_tile_next_slot.cpu().item()) == 0
    assert int((outlier_tile_meta >= 0).sum().cpu().item()) == 2
    expected_flags = [0] * num_blocks
    expected_flags[outlier_block] = 1
    assert outlier_block_flags.cpu().tolist() == expected_flags
    expected_bitmap = [[0] for _ in range(num_blocks)]
    expected_bitmap[outlier_block] = [3]
    assert outlier_tile_bitmap.cpu().tolist() == expected_bitmap

    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    query[:, :, 0] = 1.0
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor([num_blocks * 16], dtype=torch.int32, device="cuda"),
        1.0,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
        outlier_arena=outlier_arena,
        outlier_block_flags=outlier_block_flags,
        outlier_tile_bitmap=outlier_tile_bitmap,
        outlier_tile_meta=outlier_tile_meta,
    )
    expected = _raw_decode_single_block(query.cpu(), key.cpu(), value.cpu(), 1.0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_tile_and_block_fallback_pool_do_not_overlap_cuda():
    torch.manual_seed(63)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    full_key = torch.ones(16, 1, 16, dtype=torch.float32)
    full_key.reshape(-1)[1::2] = 1.0e20
    full_key = full_key.to(device="cuda", dtype=torch.bfloat16)
    full_value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    partial_key = (3.0 + 0.01 * torch.randn(1, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    partial_value = (4.0 + 0.01 * torch.randn(1, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    fallback_tile_ids = torch.full((2, 2), -1, dtype=torch.int32, device="cuda")
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed_full = ops.byte_v2_reshape_and_cache(
        full_key,
        full_value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )
    packed_partial = ops.byte_v2_reshape_and_cache(
        partial_key,
        partial_value,
        kv_cache,
        torch.tensor([16], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
    )

    assert packed_full.cpu().tolist() == [0]
    assert packed_partial.numel() == 0
    assert fallback_block_ids.cpu().tolist() == [-1, 0]
    assert fallback_next_slot.cpu().tolist() == [1]
    assert fallback_tile_next_slot.cpu().tolist() == [1]
    assert fallback_tile_ids.cpu().tolist()[0][0] == 3

    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_tile_ids=fallback_tile_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), full_key.cpu(), full_value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_lossy_outlier_threshold_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE", "1")
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(16, 1, 16, dtype=torch.float32)
    key[0, 0, 0] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = torch.ones(16, 1, 16, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.arange(16, dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.cpu().tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [0]

    key_out, value_out = unpack_byte_v2_kv_block_from_page(
        kv_cache[0].cpu(), layout
    )
    _assert_bf16_bits_equal(value_out, value.cpu())
    _assert_bf16_bits_equal(key_out[1:], key.cpu()[1:])
    _assert_bf16_bits_equal(key_out[0, :, 1:], key.cpu()[0, :, 1:])

    page = kv_cache[0].cpu()
    offsets = layout.tile_offsets("k", 0, 0)
    base = int(page[offsets.base].item())
    assert int(page[offsets.fallback].item()) == 0

    original_outlier_exp = (
        int(key.cpu()[0, 0, 0].contiguous().view(torch.int16).item()) >> 7
    ) & 0xFF
    stored_exp = (
        int(key_out[0, 0, 0].contiguous().view(torch.int16).item()) >> 7
    ) & 0xFF
    assert stored_exp != original_outlier_exp
    assert stored_exp == min(max(original_outlier_exp, base), base + 15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_decode_append_uses_partial_raw_fallback_cuda():
    torch.manual_seed(29)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(1, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(1, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.tensor([0], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 1
    assert fallback_block_ids.cpu().tolist() == [0]
    assert fallback_next_slot.cpu().tolist() == [1]

    query = torch.zeros(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), key.cpu(), value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_batched_decode_append_uses_partial_raw_fallback_cuda():
    torch.manual_seed(32)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(2, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(2, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.tensor([0, 16], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.numel() == 0
    assert torch.equal(
        kv_cache[:, BYTE_V2_PAGE_STATUS_OFFSET].cpu(),
        torch.full((2,), BYTE_V2_PAGE_STATUS_RAW_FALLBACK, dtype=torch.uint8),
    )
    assert kv_cache[:, BYTE_V2_PAGE_VALID_ROWS_OFFSET].cpu().tolist() == [1, 1]
    fallback_ids = fallback_block_ids.cpu().tolist()
    assert sorted(fallback_ids) == [0, 1]
    assert fallback_next_slot.cpu().tolist() == [2]

    query = torch.randn(2, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0], [1]], dtype=torch.int32, device="cuda"),
        torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = value.cpu()[:, 0].unsqueeze(1).expand(2, 2, 16).contiguous()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_batched_same_block_update_falls_back_cuda():
    torch.manual_seed(34)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(2, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(2, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
    )

    assert packed.numel() == 0
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 2
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [0]

    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), key.cpu(), value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda():
    torch.manual_seed(31)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = None
    for row in range(16):
        packed = ops.byte_v2_reshape_and_cache(
            key[row:row + 1],
            value[row:row + 1],
            kv_cache,
            torch.tensor([row], dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )

    assert packed is not None
    assert packed.tolist() == [0]
    assert int(kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
        BYTE_V2_PAGE_STATUS_COMPRESSED
    )
    assert int(kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET].item()) == 16
    assert fallback_block_ids.cpu().tolist() == [-1]
    assert fallback_next_slot.cpu().tolist() == [1]

    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_decode_single_block(
        query.cpu(), key.cpu(), value.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_batched_decode_append_finalizes_blocks_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH", "1")
    torch.manual_seed(33)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key_by_request = (1.0 + 0.01 * torch.randn(2, 16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    value_by_request = (2.0 + 0.01 * torch.randn(2, 16, 1, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        2, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        2, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    packed = None
    for row in range(16):
        packed = ops.byte_v2_reshape_and_cache(
            key_by_request[:, row].contiguous(),
            value_by_request[:, row].contiguous(),
            kv_cache,
            torch.tensor([row, 16 + row], dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )

    assert packed is not None
    assert packed.cpu().tolist() == [0, 1]
    assert torch.equal(
        kv_cache[:, BYTE_V2_PAGE_STATUS_OFFSET].cpu(),
        torch.full((2,), BYTE_V2_PAGE_STATUS_COMPRESSED, dtype=torch.uint8),
    )
    assert kv_cache[:, BYTE_V2_PAGE_VALID_ROWS_OFFSET].cpu().tolist() == [16, 16]
    assert fallback_block_ids.cpu().tolist() == [-1, -1]
    assert fallback_next_slot.cpu().tolist() == [2]

    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(
        device="cuda", dtype=torch.bfloat16
    )
    actual = ops.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        torch.tensor([[0], [1]], dtype=torch.int32, device="cuda"),
        torch.tensor([16, 16], dtype=torch.int32, device="cuda"),
        0.125,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_decode_batched_blocks(
        query.cpu(), key_by_request.cpu(), value_by_request.cpu(), 0.125
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_batched_decode_append_duplicate_block_fails_cuda(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH", "1")
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(2, 1, 16, device="cuda", dtype=torch.bfloat16)
    value = torch.ones(2, 1, 16, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        1, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="duplicate token slot"):
        ops.byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(1, 1, 16, device="cuda", dtype=torch.bfloat16)
    value = torch.ones(1, 1, 16, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        0, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="sparse fallback pool exhausted"):
        ops.byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_sparse_fallback_pool_exhaustion_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.ones(16, 1, 16, dtype=torch.float32)
    key.reshape(-1)[1::2] = 1.0e20
    key = key.to(device="cuda", dtype=torch.bfloat16)
    value = torch.ones(16, 1, 16, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        1, layout.page_size_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_pool = torch.empty(
        0, layout.raw_block_bytes, dtype=torch.uint8, device="cuda"
    )
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="sparse fallback pool exhausted"):
        ops.byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            torch.arange(16, dtype=torch.int64, device="cuda"),
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )
