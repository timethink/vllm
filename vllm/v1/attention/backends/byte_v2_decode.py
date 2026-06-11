# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference Byte-v2 paged decode attention.

This CPU-only implementation is a correctness fallback for wiring tests. It
uses vLLM-style block-table addressing, unpacks compressed pages, and computes
ordinary scaled dot-product attention. The production path will replace this
with a CUDA kernel that decodes tiles inside the attention kernel.
"""

import math

import torch

from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    ByteV2PageLayout,
    unpack_byte_v2_kv_block_from_page,
)


def _validate_decode_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: ByteV2PageLayout,
) -> None:
    if query.device.type != "cpu" or kv_cache.device.type != "cpu":
        raise ValueError("Byte-v2 reference decode currently requires CPU tensors")
    if block_table.device.type != "cpu" or seq_lens.device.type != "cpu":
        raise ValueError("Byte-v2 reference metadata tensors must be on CPU")
    if query.dtype != torch.bfloat16:
        raise ValueError("Byte-v2 reference decode expects BF16 query")
    if query.ndim != 3:
        raise ValueError("query must have shape [num_decode_tokens, num_heads, D]")
    if kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must use uint8 dtype")
    if kv_cache.ndim != 2 or kv_cache.shape[1] != layout.page_size_bytes:
        raise ValueError(
            "kv_cache must have shape "
            f"[num_blocks, {layout.page_size_bytes}], got {tuple(kv_cache.shape)}"
        )
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [num_reqs, max_blocks]")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1-D tensor")

    num_decode_tokens, num_heads, head_size = query.shape
    if block_table.shape[0] < num_decode_tokens:
        raise ValueError("block_table must contain one row per decode token")
    if seq_lens.numel() < num_decode_tokens:
        raise ValueError("seq_lens must contain one entry per decode token")
    if head_size != layout.head_size:
        raise ValueError(
            f"query head size must be {layout.head_size}, got {head_size}"
        )
    if num_heads % layout.num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")


def _validate_prefill_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    layout: ByteV2PageLayout,
    start_req_idx: int,
) -> None:
    if query.device.type != "cpu" or kv_cache.device.type != "cpu":
        raise ValueError("Byte-v2 reference prefill currently requires CPU tensors")
    if block_table.device.type != "cpu" or seq_lens.device.type != "cpu":
        raise ValueError("Byte-v2 reference metadata tensors must be on CPU")
    if query_start_loc.device.type != "cpu":
        raise ValueError("Byte-v2 query_start_loc must be on CPU")
    if query.dtype != torch.bfloat16:
        raise ValueError("Byte-v2 reference prefill expects BF16 query")
    if query.ndim != 3:
        raise ValueError("query must have shape [num_tokens, num_heads, D]")
    if kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must use uint8 dtype")
    if kv_cache.ndim != 2 or kv_cache.shape[1] != layout.page_size_bytes:
        raise ValueError(
            "kv_cache must have shape "
            f"[num_blocks, {layout.page_size_bytes}], got {tuple(kv_cache.shape)}"
        )
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [num_reqs, max_blocks]")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1-D tensor")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be a 1-D tensor")
    if start_req_idx < 0:
        raise ValueError("start_req_idx must be non-negative")
    if query_start_loc.numel() < start_req_idx + 1:
        raise ValueError("query_start_loc is too short for start_req_idx")
    if int(query_start_loc[-1].item()) > query.shape[0]:
        raise ValueError("query_start_loc points past the query tensor")

    _, num_heads, head_size = query.shape
    if head_size != layout.head_size:
        raise ValueError(
            f"query head size must be {layout.head_size}, got {head_size}"
        )
    if num_heads % layout.num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")


def _gather_request_kv(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    layout: ByteV2PageLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if seq_len == 0:
        key = torch.empty(0, layout.num_kv_heads, layout.head_size)
        value = torch.empty(0, layout.num_kv_heads, layout.head_size_v)
        return key.to(torch.bfloat16), value.to(torch.bfloat16)

    num_logical_blocks = math.ceil(seq_len / layout.block_size)
    if block_table_row.numel() < num_logical_blocks:
        raise ValueError("block_table row is too short for seq_len")

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    remaining = seq_len
    for logical_block in range(num_logical_blocks):
        physical_block = int(block_table_row[logical_block].item())
        if physical_block < 0 or physical_block >= kv_cache.shape[0]:
            raise ValueError(f"block_table points to invalid block {physical_block}")

        page = kv_cache[physical_block]
        status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
        if status not in (
            BYTE_V2_PAGE_STATUS_COMPRESSED,
            BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
        ):
            raise NotImplementedError(
                "Byte-v2 reference decode supports compressed or raw pages, "
                f"got status {status}"
            )

        valid_rows = int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item())
        key_block, value_block = unpack_byte_v2_kv_block_from_page(page, layout)
        rows = min(remaining, layout.block_size, valid_rows)
        key_parts.append(key_block[:rows])
        value_parts.append(value_block[:rows])
        remaining -= rows

    if remaining:
        raise ValueError("seq_len exceeds valid rows in Byte-v2 pages")
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def byte_v2_paged_decode_attention_ref(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
) -> torch.Tensor:
    """Reference paged decode attention over Byte-v2 compressed pages.

    Args:
        query: BF16 tensor `[num_decode_tokens, num_heads, head_size]`.
        kv_cache: CPU uint8 Byte-v2 pages `[num_blocks, page_size_bytes]`.
        block_table: Physical block ids `[num_decode_tokens, max_blocks]`.
        seq_lens: Context lengths `[num_decode_tokens]`.
        layout: Byte-v2 page layout.
        scale: Attention scale.

    Returns:
        BF16 tensor `[num_decode_tokens, num_heads, head_size_v]`.
    """
    _validate_decode_inputs(query, kv_cache, block_table, seq_lens, layout)

    query = query.detach().cpu().contiguous()
    block_table = block_table.detach().cpu().contiguous()
    seq_lens = seq_lens.detach().cpu().to(torch.int64)

    num_decode_tokens, num_heads, _ = query.shape
    q_per_kv = num_heads // layout.num_kv_heads
    output = torch.empty(
        num_decode_tokens, num_heads, layout.head_size_v, dtype=query.dtype
    )

    for req_idx in range(num_decode_tokens):
        seq_len = int(seq_lens[req_idx].item())
        key, value = _gather_request_kv(
            kv_cache, block_table[req_idx], seq_len, layout
        )
        if seq_len == 0:
            output[req_idx].zero_()
            continue

        key_f32 = key.to(torch.float32)
        value_f32 = value.to(torch.float32)
        for q_head in range(num_heads):
            kv_head = q_head // q_per_kv
            q_vec = query[req_idx, q_head].to(torch.float32)
            scores = torch.matmul(key_f32[:, kv_head, :], q_vec) * scale
            probs = torch.softmax(scores, dim=0)
            out = torch.matmul(probs, value_f32[:, kv_head, :])
            output[req_idx, q_head] = out.to(query.dtype)

    return output


def byte_v2_paged_prefill_attention_ref(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
    *,
    start_req_idx: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Reference prefill attention over Byte-v2 paged K/V cache.

    Args:
        query: BF16 tensor `[num_tokens, num_heads, head_size]` containing all
            query rows in the batch.
        kv_cache: CPU uint8 Byte-v2 pages `[num_blocks, page_size_bytes]`.
        block_table: Physical block ids `[num_reqs, max_blocks]`.
        seq_lens: Final context lengths `[num_reqs]` after cache update.
        query_start_loc: Prefix sum of query rows `[num_reqs + 1]`.
        layout: Byte-v2 page layout.
        scale: Attention scale.
        start_req_idx: First request index to treat as prefill.
        causal: Whether each prefill token should mask future tokens.

    Returns:
        BF16 tensor containing only the prefill output rows.
    """
    _validate_prefill_inputs(
        query,
        kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        layout,
        start_req_idx,
    )

    query = query.detach().cpu().contiguous()
    block_table = block_table.detach().cpu().contiguous()
    seq_lens = seq_lens.detach().cpu().to(torch.int64)
    query_start_loc = query_start_loc.detach().cpu().to(torch.int64)

    num_reqs = query_start_loc.numel() - 1
    if block_table.shape[0] < num_reqs:
        raise ValueError("block_table must contain one row per request")
    if seq_lens.numel() < num_reqs:
        raise ValueError("seq_lens must contain one entry per request")

    _, num_heads, _ = query.shape
    q_per_kv = num_heads // layout.num_kv_heads
    output_parts: list[torch.Tensor] = []

    for req_idx in range(start_req_idx, num_reqs):
        q_start = int(query_start_loc[req_idx].item())
        q_end = int(query_start_loc[req_idx + 1].item())
        query_len = q_end - q_start
        if query_len <= 0:
            continue

        seq_len = int(seq_lens[req_idx].item())
        context_len = seq_len - query_len
        if context_len < 0:
            raise ValueError("prefill query_len cannot exceed seq_len")

        key, value = _gather_request_kv(
            kv_cache, block_table[req_idx], seq_len, layout
        )
        key_f32 = key.to(torch.float32)
        value_f32 = value.to(torch.float32)
        req_output = torch.empty(
            query_len, num_heads, layout.head_size_v, dtype=query.dtype
        )

        for local_idx, token_idx in enumerate(range(q_start, q_end)):
            attend_len = (
                context_len + local_idx + 1 if causal else seq_len
            )
            if attend_len == 0:
                req_output[local_idx].zero_()
                continue
            for q_head in range(num_heads):
                kv_head = q_head // q_per_kv
                q_vec = query[token_idx, q_head].to(torch.float32)
                scores = torch.matmul(
                    key_f32[:attend_len, kv_head, :], q_vec
                ) * scale
                probs = torch.softmax(scores, dim=0)
                out = torch.matmul(probs, value_f32[:attend_len, kv_head, :])
                req_output[local_idx, q_head] = out.to(query.dtype)
        output_parts.append(req_output)

    if not output_parts:
        return torch.empty(0, num_heads, layout.head_size_v, dtype=query.dtype)
    return torch.cat(output_parts, dim=0)
