# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends import byte_v2_attn as byte_v2_attn_module
from vllm.v1.attention.backends.byte_v2_attn import (
    ByteV2AttentionImpl,
    ByteV2Metadata,
)
from vllm.v1.attention.backends.byte_v2_decode import (
    byte_v2_paged_decode_attention_ref,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    ByteV2PageLayout,
    pack_byte_v2_kv_block_to_page,
    pack_byte_v2_raw_kv_block_to_page,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_paged_decode_attention,
)
from vllm.v1.attention.backends.byte_v2_torch import (
    byte_v2_paged_prefill_attention_torch,
    byte_v2_raw_prefill_attention_torch,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)


def _make_blocks(
    num_blocks: int,
    layout: ByteV2PageLayout,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    key_blocks = [
        (1.0 + 0.01 * torch.randn(
            layout.block_size, layout.num_kv_heads, layout.head_size
        )).to(torch.bfloat16)
        for _ in range(num_blocks)
    ]
    value_blocks = [
        (2.0 + 0.01 * torch.randn(
            layout.block_size, layout.num_kv_heads, layout.head_size_v
        )).to(torch.bfloat16)
        for _ in range(num_blocks)
    ]
    kv_cache = torch.zeros(num_blocks, layout.page_size_bytes, dtype=torch.uint8)
    for block_id, (key_block, value_block) in enumerate(
        zip(key_blocks, value_blocks, strict=True)
    ):
        pack_byte_v2_kv_block_to_page(
            key_block, value_block, layout, page=kv_cache[block_id]
        )
    return key_blocks, value_blocks, kv_cache


def _raw_paged_decode(
    query: torch.Tensor,
    key_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
) -> torch.Tensor:
    num_decode_tokens, num_heads, _ = query.shape
    q_per_kv = num_heads // layout.num_kv_heads
    output = torch.empty(
        num_decode_tokens, num_heads, layout.head_size_v, dtype=query.dtype
    )

    for req_idx in range(num_decode_tokens):
        remaining = int(seq_lens[req_idx].item())
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        logical_block = 0
        while remaining:
            physical_block = int(block_table[req_idx, logical_block].item())
            rows = min(remaining, layout.block_size)
            key_parts.append(key_blocks[physical_block][:rows])
            value_parts.append(value_blocks[physical_block][:rows])
            remaining -= rows
            logical_block += 1

        key = torch.cat(key_parts, dim=0).to(torch.float32)
        value = torch.cat(value_parts, dim=0).to(torch.float32)
        for q_head in range(num_heads):
            kv_head = q_head // q_per_kv
            scores = torch.matmul(
                key[:, kv_head, :], query[req_idx, q_head].to(torch.float32)
            )
            probs = torch.softmax(scores * scale, dim=0)
            output[req_idx, q_head] = torch.matmul(
                probs, value[:, kv_head, :]
            ).to(query.dtype)

    return output


def _raw_request_kv(
    key_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
    block_table_row: torch.Tensor,
    seq_len: int,
    layout: ByteV2PageLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = seq_len
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    logical_block = 0
    while remaining:
        physical_block = int(block_table_row[logical_block].item())
        rows = min(remaining, layout.block_size)
        key_parts.append(key_blocks[physical_block][:rows])
        value_parts.append(value_blocks[physical_block][:rows])
        remaining -= rows
        logical_block += 1
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def _raw_paged_prefill(
    query: torch.Tensor,
    key_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
    start_req_idx: int,
) -> torch.Tensor:
    num_heads = query.shape[1]
    q_per_kv = num_heads // layout.num_kv_heads
    output_parts: list[torch.Tensor] = []

    for req_idx in range(start_req_idx, query_start_loc.numel() - 1):
        q_start = int(query_start_loc[req_idx].item())
        q_end = int(query_start_loc[req_idx + 1].item())
        query_len = q_end - q_start
        seq_len = int(seq_lens[req_idx].item())
        context_len = seq_len - query_len
        key, value = _raw_request_kv(
            key_blocks,
            value_blocks,
            block_table[req_idx],
            seq_len,
            layout,
        )
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        req_output = torch.empty(
            query_len, num_heads, layout.head_size_v, dtype=query.dtype
        )
        for local_idx, token_idx in enumerate(range(q_start, q_end)):
            attend_len = context_len + local_idx + 1
            for q_head in range(num_heads):
                kv_head = q_head // q_per_kv
                scores = torch.matmul(
                    key[:attend_len, kv_head, :],
                    query[token_idx, q_head].to(torch.float32),
                )
                probs = torch.softmax(scores * scale, dim=0)
                req_output[local_idx, q_head] = torch.matmul(
                    probs, value[:attend_len, kv_head, :]
                ).to(query.dtype)
        output_parts.append(req_output)

    return torch.cat(output_parts, dim=0)


def test_byte_v2_paged_decode_reference_matches_raw_attention():
    torch.manual_seed(4)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 16], dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual_ref = byte_v2_paged_decode_attention_ref(
        query, kv_cache, block_table, seq_lens, layout, scale
    )
    actual_op = byte_v2_paged_decode_attention(
        query,
        kv_cache,
        block_table,
        seq_lens,
        scale,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual_ref, expected)
    torch.testing.assert_close(actual_op, expected)


def test_byte_v2_paged_decode_reference_reads_raw_tail_block():
    torch.manual_seed(7)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    pack_byte_v2_raw_kv_block_to_page(
        key_blocks[1][:4], value_blocks[1][:4], 4, layout, page=kv_cache[1]
    )
    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([20], dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention_ref(
        query, kv_cache, block_table, seq_lens, layout, scale
    )

    torch.testing.assert_close(actual, expected)


def test_byte_v2_attention_impl_forward_cpu_reference_decode():
    torch.manual_seed(5)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 16], dtype=torch.int32)
    scale = 0.125
    metadata = ByteV2Metadata(
        seq_lens=seq_lens,
        slot_mapping=torch.arange(2, dtype=torch.int64),
        block_table=block_table,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=20,
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
    )
    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=scale,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((3, 32), 99, dtype=torch.bfloat16)

    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.reshape(2, -1),
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    assert result is output
    torch.testing.assert_close(output[:2], expected.reshape(2, -1))
    assert torch.equal(output[2], torch.zeros_like(output[2]))


def test_byte_v2_attention_impl_forward_cpu_reference_mixed_prefill_decode():
    torch.manual_seed(12)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(4, layout)
    query = (0.5 + 0.01 * torch.randn(5, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([17, 20], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)
    scale = 0.125
    metadata = ByteV2Metadata(
        seq_lens=seq_lens,
        slot_mapping=torch.arange(5, dtype=torch.int64),
        block_table=block_table,
        query_start_loc=query_start_loc,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=20,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        num_prefill_tokens=4,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
        query_start_loc_cpu=query_start_loc,
    )
    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=scale,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((6, 32), 99, dtype=torch.bfloat16)

    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.reshape(5, -1),
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    decode_expected = _raw_paged_decode(
        query[:1],
        key_blocks,
        value_blocks,
        block_table[:1],
        seq_lens[:1],
        layout,
        scale,
    )
    prefill_expected = _raw_paged_prefill(
        query,
        key_blocks,
        value_blocks,
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        scale,
        start_req_idx=1,
    )

    assert result is output
    torch.testing.assert_close(output[:1], decode_expected.reshape(1, -1))
    torch.testing.assert_close(output[1:5], prefill_expected.reshape(4, -1))
    assert torch.equal(output[5], torch.zeros_like(output[5]))


def test_byte_v2_attention_impl_continuation_prefill_reads_paged_cache():
    torch.manual_seed(21)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(4, layout)
    query = (0.5 + 0.01 * torch.randn(5, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([17, 20], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)
    scale = 0.125
    metadata = ByteV2Metadata(
        seq_lens=seq_lens,
        slot_mapping=torch.arange(5, dtype=torch.int64),
        block_table=block_table,
        query_start_loc=query_start_loc,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=20,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        num_prefill_tokens=4,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
        query_start_loc_cpu=query_start_loc,
    )
    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=scale,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((6, 32), 99, dtype=torch.bfloat16)

    fake_key = torch.zeros_like(query[:, :1, :])
    fake_value = torch.zeros_like(query[:, :1, :])
    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.reshape(5, -1),
        key=fake_key,
        value=fake_value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    decode_expected = _raw_paged_decode(
        query[:1],
        key_blocks,
        value_blocks,
        block_table[:1],
        seq_lens[:1],
        layout,
        scale,
    )
    prefill_expected = _raw_paged_prefill(
        query,
        key_blocks,
        value_blocks,
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        scale,
        start_req_idx=1,
    )

    assert result is output
    torch.testing.assert_close(output[:1], decode_expected.reshape(1, -1))
    torch.testing.assert_close(output[1:5], prefill_expected.reshape(4, -1))
    assert torch.equal(output[5], torch.zeros_like(output[5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_attention_impl_continuation_prefill_cuda_decode_like():
    torch.manual_seed(22)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(4, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([20], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    scale = 0.125
    metadata = ByteV2Metadata(
        seq_lens=seq_lens.cuda(),
        slot_mapping=torch.arange(4, dtype=torch.int64, device="cuda"),
        block_table=block_table.cuda(),
        query_start_loc=query_start_loc.cuda(),
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=20,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=4,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu=seq_lens,
    )
    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=scale,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((4, 32), 99, dtype=torch.bfloat16, device="cuda")

    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.cuda().reshape(4, -1),
        key=torch.zeros(4, 1, 16, dtype=torch.bfloat16, device="cuda"),
        value=torch.zeros(4, 1, 16, dtype=torch.bfloat16, device="cuda"),
        kv_cache=kv_cache.cuda(),
        attn_metadata=metadata,
        output=output,
    )

    expected = _raw_paged_prefill(
        query,
        key_blocks,
        value_blocks,
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        scale,
        start_req_idx=0,
    )
    assert result is output
    torch.testing.assert_close(output.cpu(), expected.reshape(4, -1))


def test_byte_v2_paged_prefill_torch_reads_sparse_fallback_pool():
    torch.manual_seed(18)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key_block = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value_block = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)
    kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size
    fallback_pool = torch.empty(1, layout.raw_block_bytes, dtype=torch.uint8)
    fallback_pool[0, : layout.raw_key_bytes] = key_block.contiguous().view(
        torch.uint8
    ).flatten()
    fallback_pool[0, layout.raw_key_bytes :] = value_block.contiguous().view(
        torch.uint8
    ).flatten()
    fallback_block_ids = torch.tensor([0], dtype=torch.int32)

    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    scale = 0.125

    actual = byte_v2_paged_prefill_attention_torch(
        query,
        kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        scale,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
    )
    expected = _raw_paged_prefill(
        query,
        [key_block],
        [value_block],
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        scale,
        start_req_idx=0,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_matches_raw_attention_cuda():
    torch.manual_seed(14)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 16], dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda():
    torch.manual_seed(24)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(2, 8, 32)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([24, 16], dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    torch.manual_seed(25)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 20
    key_template = torch.arange(
        layout.block_size * layout.num_kv_heads * layout.head_size,
        dtype=torch.float32,
    ).reshape(layout.block_size, layout.num_kv_heads, layout.head_size)
    value_template = torch.arange(
        layout.block_size * layout.num_kv_heads * layout.head_size_v,
        dtype=torch.float32,
    ).reshape(layout.block_size, layout.num_kv_heads, layout.head_size_v)
    key_blocks = [
        (0.75 + ((key_template + block_id) % 127) / 512).to(torch.bfloat16)
        for block_id in range(num_blocks)
    ]
    value_blocks = [
        (1.25 + ((value_template + 3 * block_id) % 127) / 512).to(
            torch.bfloat16
        )
        for block_id in range(num_blocks)
    ]
    kv_cache = torch.zeros(num_blocks, layout.page_size_bytes, dtype=torch.uint8)
    for block_id, (key_block, value_block) in enumerate(
        zip(key_blocks, value_blocks, strict=True)
    ):
        pack_byte_v2_kv_block_to_page(
            key_block, value_block, layout, page=kv_cache[block_id]
        )
    query = (0.5 + torch.arange(8 * 32, dtype=torch.float32).reshape(1, 8, 32)
             / 1024).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 3],
                            dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    monkeypatch.delenv("VLLM_BYTE_V2_DECODE_SPLIT_K", raising=False)
    actual_default = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(
        actual_default.cpu(), expected, rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_paged_decode_external_partial_workspace_cuda(
    monkeypatch,
):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    torch.manual_seed(125)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 20
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)
    query = (0.5 + 0.01 * torch.randn(1, 8, 32)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 3],
                            dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    query_cuda = query.cuda()
    kv_cache_cuda = kv_cache.cuda()
    block_table_cuda = block_table.cuda()
    seq_lens_cuda = seq_lens.cuda()
    workspace = torch.empty(1 * 8 * 4 * (layout.head_size_v + 1),
                            dtype=torch.float32,
                            device="cuda")

    actual = ops.byte_v2_paged_decode_attention(
        query_cuda,
        kv_cache_cuda,
        block_table_cuda,
        seq_lens_cuda,
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
        partial_workspace=workspace,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    with pytest.raises(RuntimeError, match="partial_workspace is too small"):
        ops.byte_v2_paged_decode_attention(
            query_cuda,
            kv_cache_cuda,
            block_table_cuda,
            seq_lens_cuda,
            scale,
            block_size=16,
            num_kv_heads=2,
            head_size=32,
            head_size_v=32,
            page_size_bytes=layout.page_size_bytes,
            partial_workspace=workspace[:-1],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_native_paged_decode_parallel_reduce_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE", "1")
    torch.manual_seed(126)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 20
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)
    query = (0.5 + 0.01 * torch.randn(1, 8, 32)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor(
        [num_blocks * layout.block_size - 3], dtype=torch.int32
    )
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = ops.byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda(
    monkeypatch,
):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    torch.manual_seed(28)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 20
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)
    fallback_block = 7
    kv_cache[fallback_block, BYTE_V2_PAGE_STATUS_OFFSET] = (
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    )
    kv_cache[fallback_block, BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size
    fallback_pool = torch.empty(1, layout.raw_block_bytes, dtype=torch.uint8)
    fallback_pool[0, : layout.raw_key_bytes] = key_blocks[
        fallback_block
    ].contiguous().view(torch.uint8).flatten()
    fallback_pool[0, layout.raw_key_bytes :] = value_blocks[
        fallback_block
    ].contiguous().view(torch.uint8).flatten()
    fallback_block_ids = torch.full((num_blocks,), -1, dtype=torch.int32)
    fallback_block_ids[fallback_block] = 0

    query = (0.5 + 0.01 * torch.randn(1, 8, 32)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 3],
                            dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = ops.byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool.cuda(),
        fallback_block_ids=fallback_block_ids.cuda(),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_tile_fastpath_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_TILE_FASTPATH", "1")
    torch.manual_seed(29)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 20
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)

    fallback_block = 9
    fallback_kv_head = 1
    fallback_k_dim_tile = 1
    fallback_v_dim_tile = 0
    raw_tile_bytes = layout.block_size * 16 * 2
    fallback_pool = torch.zeros(1, layout.raw_block_bytes, dtype=torch.uint8)
    fallback_pool_flat = fallback_pool.flatten()
    key_tile = key_blocks[fallback_block][
        :, fallback_kv_head, fallback_k_dim_tile * 16 : (fallback_k_dim_tile + 1) * 16
    ]
    value_tile = value_blocks[fallback_block][
        :, fallback_kv_head, fallback_v_dim_tile * 16 : (fallback_v_dim_tile + 1) * 16
    ]
    fallback_pool_flat[0:raw_tile_bytes] = (
        key_tile.contiguous().view(torch.uint8).flatten()
    )
    fallback_pool_flat[raw_tile_bytes : 2 * raw_tile_bytes] = (
        value_tile.contiguous().view(torch.uint8).flatten()
    )

    fallback_tile_ids = torch.full(
        (num_blocks, layout.total_tiles), -1, dtype=torch.int32
    )
    key_tile_idx = layout.tile_index(
        "k", fallback_kv_head, fallback_k_dim_tile
    )
    value_tile_idx = layout.tile_index(
        "v", fallback_kv_head, fallback_v_dim_tile
    )
    fallback_tile_ids[fallback_block, key_tile_idx] = 0
    fallback_tile_ids[fallback_block, value_tile_idx] = 1
    kv_cache[
        fallback_block,
        layout.tile_offsets("k", fallback_kv_head, fallback_k_dim_tile).fallback,
    ] = 1
    kv_cache[
        fallback_block,
        layout.tile_offsets("v", fallback_kv_head, fallback_v_dim_tile).fallback,
    ] = 1
    fallback_block_ids = torch.full((num_blocks,), -1, dtype=torch.int32)

    query = (0.5 + 0.01 * torch.randn(1, 8, 32)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 3],
                            dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = ops.byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool.cuda(),
        fallback_block_ids=fallback_block_ids.cuda(),
        fallback_tile_ids=fallback_tile_ids.cuda(),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_cute_stage1_cuda(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1", "1")
    torch.manual_seed(29)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=8, head_size=128, head_size_v=128
    )
    num_blocks = 20
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)
    query = (0.5 + 0.01 * torch.randn(1, 32, 128)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 5],
                            dtype=torch.int32)
    scale = layout.head_size**-0.5

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = ops.byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size,
        head_size_v=layout.head_size_v,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda(
    monkeypatch,
):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA split-K path requires Ampere or newer")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "0")
    torch.manual_seed(27)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_blocks = 24
    key_blocks, value_blocks, kv_cache = _make_blocks(num_blocks, layout)
    query = (0.5 + 0.01 * torch.randn(1, 8, 32)).to(torch.bfloat16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([num_blocks * layout.block_size - 1],
                            dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_paged_decode_attention_op_reads_raw_tail_cuda():
    torch.manual_seed(16)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    pack_byte_v2_raw_kv_block_to_page(
        key_blocks[1][:4], value_blocks[1][:4], 4, layout, page=kv_cache[1]
    )
    query = (0.5 + 0.01 * torch.randn(1, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([20], dtype=torch.int32)
    scale = 0.125

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    actual = byte_v2_paged_decode_attention(
        query.cuda(),
        kv_cache.cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        scale,
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
    )

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_attention_impl_forward_cuda_decode_correctness_fallback():
    torch.manual_seed(15)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key_blocks, value_blocks, kv_cache = _make_blocks(2, layout)
    query = (0.5 + 0.01 * torch.randn(2, 2, 16)).to(torch.bfloat16)
    block_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 16], dtype=torch.int32)
    scale = 0.125
    metadata = ByteV2Metadata(
        seq_lens=seq_lens.cuda(),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cuda"),
        block_table=block_table.cuda(),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=20,
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
    )
    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=scale,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((3, 32), 99, dtype=torch.bfloat16, device="cuda")

    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.cuda().reshape(2, -1),
        key=torch.empty(0, device="cuda"),
        value=torch.empty(0, device="cuda"),
        kv_cache=kv_cache.cuda(),
        attn_metadata=metadata,
        output=output,
    )

    expected = _raw_paged_decode(
        query, key_blocks, value_blocks, block_table, seq_lens, layout, scale
    )
    assert result is output
    torch.testing.assert_close(output[:2].cpu(), expected.reshape(2, -1))
    assert torch.equal(output[2], torch.zeros_like(output[2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_raw_prefill_uses_flash_attention_fast_path(monkeypatch):
    if not is_flash_attn_varlen_func_available():
        pytest.skip("FlashAttention varlen op is not available")

    torch.manual_seed(31)
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=2, head_size=32, head_size_v=32
    )
    num_tokens = 17
    scale = 0.125
    query = (0.5 + 0.01 * torch.randn(num_tokens, 8, 32)).to(
        torch.bfloat16
    )
    key = (1.0 + 0.01 * torch.randn(num_tokens, 2, 32)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(num_tokens, 2, 32)).to(torch.bfloat16)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32)
    expected = byte_v2_raw_prefill_attention_torch(
        query.cuda(),
        key.cuda(),
        value.cuda(),
        seq_lens.cuda(),
        query_start_loc.cuda(),
        layout,
        scale,
    )
    assert expected is not None

    def fail_torch_prefill(*args, **kwargs):
        raise AssertionError("PyTorch raw prefill fallback should not run")

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_raw_prefill_attention_torch",
        fail_torch_prefill,
    )

    metadata = ByteV2Metadata(
        seq_lens=seq_lens.cuda(),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device="cuda"),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        query_start_loc=query_start_loc.cuda(),
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        max_seq_len=num_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=num_tokens,
        block_size=16,
        page_size_bytes=layout.page_size_bytes,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu=seq_lens,
    )
    impl = ByteV2AttentionImpl(
        num_heads=8,
        head_size=32,
        scale=scale,
        num_kv_heads=2,
        kv_cache_dtype="byte_v2",
    )
    output = torch.full((num_tokens, 8 * 32), 99, dtype=torch.bfloat16, device="cuda")
    result = impl.forward(
        layer=torch.nn.Module(),
        query=query.cuda().reshape(num_tokens, -1),
        key=key.cuda(),
        value=value.cuda(),
        kv_cache=torch.zeros(
            2,
            layout.page_size_bytes,
            dtype=torch.uint8,
            device="cuda",
        ),
        attn_metadata=metadata,
        output=output,
    )

    assert result is output
    torch.testing.assert_close(
        output.cpu(),
        expected.cpu().reshape(num_tokens, -1),
        rtol=2e-2,
        atol=2e-2,
    )
