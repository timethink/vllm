# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch eager Byte-v2 correctness fallback.

This module is deliberately not a performance path. It keeps Byte-v2 execution
on the input device, including CUDA tensors, so the backend can run smoke/e2e
tests before the fused CUDA kernels are implemented.
"""

import math

import torch

from vllm.v1.attention.backends.byte_v2_codec import (
    BYTE_V2_PACKED_TILE_ELEMS,
    BYTE_V2_TILE_ELEMS,
    BYTE_V2_TILE_SIZE,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_EMPTY,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    ByteV2PageLayout,
    ByteV2PageStatusCounts,
)


def _bf16_to_u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _u16_to_bf16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def _best_window_base(exp: torch.Tensor) -> tuple[int, int]:
    hist = torch.bincount(exp.to(torch.int64), minlength=256)
    window = int(hist[:16].sum().item())
    best = window
    best_start = 0
    for start in range(1, 241):
        window += int(hist[start + 15].item()) - int(hist[start - 1].item())
        if window > best:
            best = window
            best_start = start
    return best_start, best


def _validate_kv_block(
    block: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    name: str,
) -> torch.Tensor:
    expected_shape = (block_size, num_kv_heads, head_size)
    if tuple(block.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {block.shape}")
    if block.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use bfloat16 dtype")
    return block.contiguous()


def _store_raw_block(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: str,
    block: torch.Tensor,
) -> None:
    offset, _ = layout.raw_offsets(kind)  # type: ignore[arg-type]
    raw_bytes = block.contiguous().view(torch.uint8).flatten()
    page[offset : offset + raw_bytes.numel()] = raw_bytes


def _load_raw_block(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: str,
) -> torch.Tensor:
    offset, num_bytes = layout.raw_offsets(kind)  # type: ignore[arg-type]
    head_size = layout.head_size if kind == "k" else layout.head_size_v
    raw_bytes = page[offset : offset + num_bytes].contiguous()
    return raw_bytes.view(torch.bfloat16).reshape(
        layout.block_size, layout.num_kv_heads, head_size
    ).clone()


def _load_raw_block_from_sparse_fallback(
    fallback_pool: torch.Tensor,
    fallback_block_ids: torch.Tensor,
    physical_block: int,
    layout: ByteV2PageLayout,
    kind: str,
) -> torch.Tensor:
    fallback_slot = int(fallback_block_ids[physical_block].item())
    if fallback_slot < 0 or fallback_slot >= fallback_pool.shape[0]:
        raise ValueError(
            "Byte-v2 raw fallback page has no valid sparse fallback slot"
        )
    if fallback_pool.shape[1] != layout.raw_block_bytes:
        raise ValueError(
            "Byte-v2 fallback_pool must have shape "
            f"[pool_blocks, {layout.raw_block_bytes}]"
        )

    head_size = layout.head_size if kind == "k" else layout.head_size_v
    offset = 0 if kind == "k" else layout.raw_key_bytes
    num_bytes = layout.raw_key_bytes if kind == "k" else layout.raw_value_bytes
    raw_bytes = fallback_pool[fallback_slot, offset : offset + num_bytes].contiguous()
    return raw_bytes.view(torch.bfloat16).reshape(
        layout.block_size, layout.num_kv_heads, head_size
    ).clone()


def _pack_raw_kv_block_to_page(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    valid_rows: int,
    layout: ByteV2PageLayout,
    page: torch.Tensor,
) -> None:
    if valid_rows < 0 or valid_rows > layout.block_size:
        raise ValueError("valid_rows must be in [0, block_size]")
    key_block = key_block.contiguous()
    value_block = value_block.contiguous()
    full_key = torch.zeros(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size,
        dtype=torch.bfloat16,
        device=page.device,
    )
    full_value = torch.zeros(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size_v,
        dtype=torch.bfloat16,
        device=page.device,
    )
    full_key[:valid_rows] = key_block[:valid_rows]
    full_value[:valid_rows] = value_block[:valid_rows]

    page.zero_()
    page[BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    page[BYTE_V2_PAGE_VALID_ROWS_OFFSET] = valid_rows
    _store_raw_block(page, layout, "k", full_key)
    _store_raw_block(page, layout, "v", full_value)


def _compress_fast_tile(tile: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    vals = _bf16_to_u16(tile).reshape(-1)
    exp = (vals >> 7) & 0xFF
    base_i, covered = _best_window_base(exp)
    if covered != BYTE_V2_TILE_ELEMS:
        return None

    low = (vals & 0xFF).to(torch.uint8)
    delta = exp - base_i
    sign = (vals >> 15) & 1
    code = ((sign << 3) | ((delta >> 1) & 0x07)).to(torch.uint8)
    code_packed = code[0::2] | (code[1::2] << 4)
    base = torch.tensor(base_i, dtype=torch.uint8, device=tile.device)
    fallback = torch.zeros((), dtype=torch.uint8, device=tile.device)
    return base, fallback, low, code_packed


def _store_fast_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: str,
    kv_head: int,
    dim_tile: int,
    tile: torch.Tensor,
) -> bool:
    payload = _compress_fast_tile(tile)
    if payload is None:
        return False

    base, fallback, low, code_packed = payload
    offsets = layout.tile_offsets(kind, kv_head, dim_tile)  # type: ignore[arg-type]
    page[offsets.base] = base
    page[offsets.fallback] = fallback
    page[offsets.low_bytes : offsets.low_bytes + BYTE_V2_TILE_ELEMS] = low
    page[
        offsets.code_packed : offsets.code_packed + BYTE_V2_PACKED_TILE_ELEMS
    ] = code_packed
    return True


def pack_byte_v2_kv_block_to_page_torch(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    layout: ByteV2PageLayout,
    page: torch.Tensor,
) -> bool:
    """Pack one full K/V block on the tensor device.

    Returns True when the block used the fast compressed layout. If any tile is
    not representable by the fast codec, the whole page is stored as raw
    fallback and False is returned.
    """
    key_block = _validate_kv_block(
        key_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size,
        name="key_block",
    )
    value_block = _validate_kv_block(
        value_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size_v,
        name="value_block",
    )
    page.zero_()
    page[BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_COMPRESSED
    page[BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size

    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            if not _store_fast_tile(page, layout, "k", kv_head, dim_tile, tile):
                _pack_raw_kv_block_to_page(
                    key_block, value_block, layout.block_size, layout, page
                )
                return False
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            if not _store_fast_tile(page, layout, "v", kv_head, dim_tile, tile):
                _pack_raw_kv_block_to_page(
                    key_block, value_block, layout.block_size, layout, page
                )
                return False
    return True


def _load_fast_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: str,
    kv_head: int,
    dim_tile: int,
    *,
    fallback_pool: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    physical_block: int | None = None,
) -> torch.Tensor:
    offsets = layout.tile_offsets(kind, kv_head, dim_tile)  # type: ignore[arg-type]
    fallback = int(page[offsets.fallback].item())
    if fallback:
        if (
            fallback_pool is None
            or fallback_tile_ids is None
            or physical_block is None
        ):
            raise NotImplementedError(
                "Byte-v2 tile fallback pages require fallback_tile_ids"
            )
        kind_offset = 0 if kind == "k" else layout.k_dim_tiles
        tile_idx = (
            kv_head * (layout.k_dim_tiles + layout.v_dim_tiles)
            + kind_offset
            + dim_tile
        )
        tile_slot = int(fallback_tile_ids[physical_block, tile_idx].item())
        tile_bytes = BYTE_V2_TILE_ELEMS * 2
        flat_pool = fallback_pool.flatten()
        start = tile_slot * tile_bytes
        if tile_slot < 0 or start + tile_bytes > flat_pool.numel():
            raise ValueError("Byte-v2 tile fallback slot is invalid")
        return flat_pool[start : start + tile_bytes].view(torch.bfloat16).reshape(
            BYTE_V2_TILE_SIZE,
            BYTE_V2_TILE_SIZE,
        )

    base_i = page[offsets.base].to(torch.int32)
    low = page[
        offsets.low_bytes : offsets.low_bytes + BYTE_V2_TILE_ELEMS
    ].to(torch.int32)
    packed = page[
        offsets.code_packed : offsets.code_packed + BYTE_V2_PACKED_TILE_ELEMS
    ].to(torch.int32)
    code = torch.empty(BYTE_V2_TILE_ELEMS, dtype=torch.int32, device=page.device)
    code[0::2] = packed & 0x0F
    code[1::2] = (packed >> 4) & 0x0F

    low_exp_lsb = low >> 7
    delta_hi = code & 0x07
    exp_hi = (base_i >> 1) + delta_hi + ((base_i & 1) & (low_exp_lsb ^ 1))
    high = ((code & 0x08) << 4) | exp_hi
    tile_bits = (high << 8) | low
    return _u16_to_bf16(
        tile_bits,
        (BYTE_V2_TILE_SIZE, BYTE_V2_TILE_SIZE),
    )


def unpack_byte_v2_kv_block_from_page_torch(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    *,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    physical_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
    if status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
        if layout.page_size_bytes - layout.page_header_bytes < layout.raw_block_bytes:
            if (
                fallback_pool is None
                or fallback_block_ids is None
                or physical_block is None
            ):
                raise NotImplementedError(
                    "Byte-v2 compressed-only raw fallback pages require a "
                    "sparse fallback pool"
                )
            return (
                _load_raw_block_from_sparse_fallback(
                    fallback_pool, fallback_block_ids, physical_block, layout, "k"
                ),
                _load_raw_block_from_sparse_fallback(
                    fallback_pool, fallback_block_ids, physical_block, layout, "v"
                ),
            )
        return _load_raw_block(page, layout, "k"), _load_raw_block(page, layout, "v")
    if status != BYTE_V2_PAGE_STATUS_COMPRESSED:
        raise NotImplementedError(f"Byte-v2 page status {status} is not supported")

    key_block = torch.empty(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size,
        dtype=torch.bfloat16,
        device=page.device,
    )
    value_block = torch.empty(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size_v,
        dtype=torch.bfloat16,
        device=page.device,
    )

    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = _load_fast_tile(
                page,
                layout,
                "k",
                kv_head,
                dim_tile,
                fallback_pool=fallback_pool,
                fallback_tile_ids=fallback_tile_ids,
                physical_block=physical_block,
            )
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = _load_fast_tile(
                page,
                layout,
                "v",
                kv_head,
                dim_tile,
                fallback_pool=fallback_pool,
                fallback_tile_ids=fallback_tile_ids,
                physical_block=physical_block,
            )
    return key_block, value_block


def count_byte_v2_page_statuses_torch(
    kv_cache: torch.Tensor,
    layout: ByteV2PageLayout,
) -> ByteV2PageStatusCounts:
    """Count Byte-v2 page states on CPU or CUDA tensors."""
    if kv_cache.dtype != torch.uint8:
        raise ValueError("Byte-v2 kv_cache must use uint8 dtype")
    if kv_cache.ndim != 2 or kv_cache.shape[1] != layout.page_size_bytes:
        raise ValueError(
            "kv_cache must have shape "
            f"[num_blocks, {layout.page_size_bytes}], got {tuple(kv_cache.shape)}"
        )

    statuses = kv_cache[:, BYTE_V2_PAGE_STATUS_OFFSET].detach().cpu().to(torch.int64)
    valid_rows = kv_cache[:, BYTE_V2_PAGE_VALID_ROWS_OFFSET].detach().cpu().to(
        torch.int64
    )
    empty = int(
        ((statuses == BYTE_V2_PAGE_STATUS_EMPTY) & (valid_rows == 0)).sum().item()
    )
    compressed = int(
        (
            (statuses == BYTE_V2_PAGE_STATUS_COMPRESSED)
            & (valid_rows == layout.block_size)
        )
        .sum()
        .item()
    )
    partial_compressed = int(
        (
            (statuses == BYTE_V2_PAGE_STATUS_COMPRESSED)
            & (valid_rows > 0)
            & (valid_rows < layout.block_size)
        )
        .sum()
        .item()
    )
    partial_raw = int(
        (
            (statuses == BYTE_V2_PAGE_STATUS_RAW_FALLBACK)
            & (valid_rows >= 0)
            & (valid_rows < layout.block_size)
        )
        .sum()
        .item()
    )
    full_raw_fallback = int(
        (
            (statuses == BYTE_V2_PAGE_STATUS_RAW_FALLBACK)
            & (valid_rows == layout.block_size)
        )
        .sum()
        .item()
    )
    classified = (
        empty + compressed + partial_compressed + partial_raw + full_raw_fallback
    )
    invalid = int(kv_cache.shape[0]) - classified
    return ByteV2PageStatusCounts(
        empty=empty,
        compressed=compressed,
        partial_compressed=partial_compressed,
        partial_raw=partial_raw,
        full_raw_fallback=full_raw_fallback,
        invalid=invalid,
    )


def byte_v2_reshape_and_cache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: ByteV2PageLayout,
) -> torch.Tensor:
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("Byte-v2 cache update expects BF16 key/value")
    if kv_cache.dtype != torch.uint8:
        raise ValueError("Byte-v2 cache update expects a uint8 cache")
    if kv_cache.ndim != 2 or kv_cache.shape[1] != layout.page_size_bytes:
        raise ValueError(
            "kv_cache must have shape "
            f"[num_blocks, {layout.page_size_bytes}], got {tuple(kv_cache.shape)}"
        )

    key = key.contiguous()
    value = value.contiguous()
    slots = slot_mapping.detach().cpu().flatten().to(torch.int64)
    if slots.numel() < key.shape[0]:
        raise ValueError("slot_mapping must contain at least one slot per token")
    slots = slots[: key.shape[0]]

    block_to_tokens: dict[int, dict[int, int]] = {}
    for token_idx, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block_id = slot // layout.block_size
        block_offset = slot % layout.block_size
        if block_id < 0 or block_id >= kv_cache.shape[0]:
            raise ValueError(f"slot_mapping points to invalid block id {block_id}")
        offsets = block_to_tokens.setdefault(block_id, {})
        if block_offset in offsets:
            raise ValueError(
                f"slot_mapping contains duplicate block offset {block_offset} "
                f"for block {block_id}"
            )
        offsets[block_offset] = token_idx

    packed_block_ids: list[int] = []
    for block_id, offsets in sorted(block_to_tokens.items()):
        page = kv_cache[block_id]
        status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
        existing_valid_rows = 0
        if status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
            key_block, value_block = unpack_byte_v2_kv_block_from_page_torch(
                page, layout
            )
            existing_valid_rows = int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item())
        elif status == BYTE_V2_PAGE_STATUS_EMPTY:
            key_block = torch.zeros(
                layout.block_size,
                layout.num_kv_heads,
                layout.head_size,
                dtype=torch.bfloat16,
                device=kv_cache.device,
            )
            value_block = torch.zeros(
                layout.block_size,
                layout.num_kv_heads,
                layout.head_size_v,
                dtype=torch.bfloat16,
                device=kv_cache.device,
            )
        else:
            raise ValueError(f"cannot update finalized Byte-v2 block {block_id}")

        for block_offset, token_idx in offsets.items():
            key_block[block_offset] = key[token_idx]
            value_block[block_offset] = value[token_idx]

        present_offsets = set(range(existing_valid_rows)) | set(offsets)
        valid_rows = max(present_offsets) + 1
        if present_offsets != set(range(valid_rows)):
            raise ValueError(
                "Byte-v2 partial raw block updates must form a contiguous "
                f"prefix, got offsets {sorted(present_offsets)}"
            )

        if valid_rows != layout.block_size:
            _pack_raw_kv_block_to_page(
                key_block, value_block, valid_rows, layout, page
            )
            continue

        if pack_byte_v2_kv_block_to_page_torch(
            key_block, value_block, layout, page
        ):
            packed_block_ids.append(block_id)

    return torch.tensor(
        packed_block_ids, device=slot_mapping.device, dtype=torch.int64
    )


def _gather_request_kv_torch(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    layout: ByteV2PageLayout,
    *,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if seq_len == 0:
        key = torch.empty(
            0,
            layout.num_kv_heads,
            layout.head_size,
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )
        value = torch.empty(
            0,
            layout.num_kv_heads,
            layout.head_size_v,
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )
        return key, value

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
        valid_rows = int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item())
        key_block, value_block = unpack_byte_v2_kv_block_from_page_torch(
            page,
            layout,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_tile_ids=fallback_tile_ids,
            physical_block=physical_block,
        )
        rows = min(remaining, layout.block_size, valid_rows)
        key_parts.append(key_block[:rows])
        value_parts.append(value_block[:rows])
        remaining -= rows

    if remaining:
        raise ValueError("seq_len exceeds valid rows in Byte-v2 pages")
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def _attention_one_request(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    layout: ByteV2PageLayout,
    *,
    causal_context_len: int | None = None,
) -> torch.Tensor:
    num_query_tokens, num_heads, _ = query.shape
    q_per_kv = num_heads // layout.num_kv_heads
    output = torch.empty(
        num_query_tokens,
        num_heads,
        layout.head_size_v,
        dtype=query.dtype,
        device=query.device,
    )
    key_f32 = key.to(torch.float32)
    value_f32 = value.to(torch.float32)

    for token_idx in range(num_query_tokens):
        attend_len = key.shape[0]
        if causal_context_len is not None:
            attend_len = causal_context_len + token_idx + 1
        if attend_len == 0:
            output[token_idx].zero_()
            continue
        for q_head in range(num_heads):
            kv_head = q_head // q_per_kv
            q_vec = query[token_idx, q_head].to(torch.float32)
            scores = torch.matmul(key_f32[:attend_len, kv_head, :], q_vec)
            probs = torch.softmax(scores * scale, dim=0)
            out = torch.matmul(probs, value_f32[:attend_len, kv_head, :])
            output[token_idx, q_head] = out.to(query.dtype)
    return output


def byte_v2_paged_decode_attention_torch(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
    *,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.dtype != torch.bfloat16:
        raise ValueError("Byte-v2 decode expects BF16 query")
    if query.ndim != 3:
        raise ValueError("query must have shape [num_decode_tokens, num_heads, D]")
    if kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must use uint8 dtype")
    if query.shape[2] != layout.head_size:
        raise ValueError(
            f"query head size must be {layout.head_size}, got {query.shape[2]}"
        )
    if query.shape[1] % layout.num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    block_table_cpu = block_table.detach().cpu()
    seq_lens_cpu = seq_lens.detach().cpu().to(torch.int64)
    num_decode_tokens = query.shape[0]
    output_parts: list[torch.Tensor] = []
    for req_idx in range(num_decode_tokens):
        seq_len = int(seq_lens_cpu[req_idx].item())
        key, value = _gather_request_kv_torch(
            kv_cache,
            block_table_cpu[req_idx],
            seq_len,
            layout,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_tile_ids=fallback_tile_ids,
        )
        output_parts.append(
            _attention_one_request(
                query[req_idx : req_idx + 1], key, value, scale, layout
            )
        )
    if not output_parts:
        return torch.empty(
            0,
            query.shape[1],
            layout.head_size_v,
            dtype=query.dtype,
            device=query.device,
        )
    return torch.cat(output_parts, dim=0)


def byte_v2_paged_prefill_attention_torch(
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
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    block_table_cpu = block_table.detach().cpu()
    seq_lens_cpu = seq_lens.detach().cpu().to(torch.int64)
    query_start_loc_cpu = query_start_loc.detach().cpu().to(torch.int64)
    num_reqs = query_start_loc_cpu.numel() - 1
    output_parts: list[torch.Tensor] = []

    for req_idx in range(start_req_idx, num_reqs):
        q_start = int(query_start_loc_cpu[req_idx].item())
        q_end = int(query_start_loc_cpu[req_idx + 1].item())
        query_len = q_end - q_start
        if query_len <= 0:
            continue
        seq_len = int(seq_lens_cpu[req_idx].item())
        context_len = seq_len - query_len
        if context_len < 0:
            raise ValueError("prefill query_len cannot exceed seq_len")
        key, value = _gather_request_kv_torch(
            kv_cache,
            block_table_cpu[req_idx],
            seq_len,
            layout,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_tile_ids=fallback_tile_ids,
        )
        output_parts.append(
            _attention_one_request(
                query[q_start:q_end],
                key,
                value,
                scale,
                layout,
                causal_context_len=context_len if causal else None,
            )
        )

    if not output_parts:
        return torch.empty(
            0,
            query.shape[1],
            layout.head_size_v,
            dtype=query.dtype,
            device=query.device,
        )
    return torch.cat(output_parts, dim=0)


def byte_v2_raw_prefill_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    layout: ByteV2PageLayout,
    scale: float,
    *,
    start_req_idx: int = 0,
    causal: bool = True,
) -> torch.Tensor | None:
    """Run prefill from raw K/V when every prefill request has no prefix.

    Returns None when a request has a non-zero context length, so the caller can
    fall back to page-gather prefill.
    """
    if key.numel() == 0 or value.numel() == 0:
        return None

    seq_lens_cpu = seq_lens.detach().cpu().to(torch.int64)
    query_start_loc_cpu = query_start_loc.detach().cpu().to(torch.int64)
    num_reqs = query_start_loc_cpu.numel() - 1
    output_parts: list[torch.Tensor] = []

    for req_idx in range(start_req_idx, num_reqs):
        q_start = int(query_start_loc_cpu[req_idx].item())
        q_end = int(query_start_loc_cpu[req_idx + 1].item())
        query_len = q_end - q_start
        if query_len <= 0:
            continue
        seq_len = int(seq_lens_cpu[req_idx].item())
        context_len = seq_len - query_len
        if context_len != 0:
            return None
        output_parts.append(
            _attention_one_request(
                query[q_start:q_end],
                key[q_start:q_end],
                value[q_start:q_end],
                scale,
                layout,
                causal_context_len=0 if causal else None,
            )
        )

    if not output_parts:
        return torch.empty(
            0,
            query.shape[1],
            layout.head_size_v,
            dtype=query.dtype,
            device=query.device,
        )
    return torch.cat(output_parts, dim=0)
