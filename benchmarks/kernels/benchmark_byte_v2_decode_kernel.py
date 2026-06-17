# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only benchmark for ByteV2 paged attention kernels.

This script isolates ``byte_v2_paged_decode_attention`` from vLLM scheduler,
sampling, model layers, and CUDA graph overhead. It is intended for local
kernel experiments: run it before and after one kernel change, compare median
latency, and keep the change only if the improvement is stable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _make_layout(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    compressed_only: bool,
    payload_layout: str,
):
    from vllm.v1.attention.backends.byte_v2_layout import (
        ByteV2PageLayout,
        ByteV2PageLayoutV3,
    )

    if payload_layout == "v3":
        if not compressed_only:
            raise ValueError("ByteV2 V3 benchmark layout requires compressed-only")
        return ByteV2PageLayoutV3(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            raw_tail_bytes=0,
        )
    return ByteV2PageLayout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        raw_tail_bytes=0 if compressed_only else None,
    )


def _raw_block_bytes(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
) -> int:
    return block_size * num_kv_heads * (head_size + head_size_v) * 2


def _make_compressible_kv(
    num_tokens: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    key = 1.0 + 0.01 * torch.randn(
        num_tokens,
        num_kv_heads,
        head_size,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    value = 2.0 + 0.01 * torch.randn(
        num_tokens,
        num_kv_heads,
        head_size_v,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    return key.to(torch.bfloat16), value.to(torch.bfloat16)


def _inject_fallback_blocks(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
    fallback_ratio: float,
    fallback_pattern: str,
    outlier_exp_scale: float,
    seed: int,
) -> int:
    if fallback_ratio <= 0.0:
        return 0

    num_blocks = math.ceil(key.shape[0] / block_size)
    num_fallback_blocks = min(
        num_blocks, max(1, math.ceil(num_blocks * fallback_ratio))
    )
    generator = torch.Generator(device=key.device)
    generator.manual_seed(seed + 17)
    block_ids = torch.randperm(num_blocks, device=key.device,
                               generator=generator)[:num_fallback_blocks]

    if fallback_pattern == "single_outlier":
        for block_id in block_ids.tolist():
            start = block_id * block_size
            if start < key.shape[0]:
                key[start, 0, 0] = outlier_exp_scale
        return num_fallback_blocks

    if fallback_pattern == "single_outlier_per_k_tile":
        dim_tiles = key.shape[2] // block_size
        for block_id in block_ids.tolist():
            start = block_id * block_size
            if start >= key.shape[0]:
                continue
            for kv_head in range(key.shape[1]):
                for dim_tile in range(dim_tiles):
                    key[start, kv_head, dim_tile * block_size] = (
                        outlier_exp_scale
                    )
        return num_fallback_blocks

    # Use 17 distinct K exponents in each 16x16 tile so the block exceeds the
    # current ByteV2 16-exponent window and is routed to sparse fallback. Keep V
    # at the normal benchmark scale so correctness diffs remain interpretable.
    for block_id in block_ids.tolist():
        start = block_id * block_size
        end = min(start + block_size, key.shape[0])
        rows = end - start
        element_ids = torch.arange(
            rows * key.shape[1] * key.shape[2],
            device=key.device,
            dtype=torch.int32,
        ).reshape(rows, key.shape[1], key.shape[2])
        key_exponent = ((element_ids % 17) - 8).to(torch.float32)
        key_sign = torch.where((element_ids & 1) == 0, 1.0, -1.0)
        key[start:end] = (key_sign * torch.pow(2.0, key_exponent)).to(
            torch.bfloat16
        )
    return num_fallback_blocks


def _build_inputs(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    block_size: int,
    compressed_only: bool,
    fallback_ratio: float,
    fallback_pattern: str,
    outlier_exp_scale: float,
    tile_fallback_pool: bool,
    outlier_arena_entries_per_block: float,
    outlier_arena_min_entries: int,
    use_outlier_block_flags: bool,
    use_outlier_tile_bitmap: bool,
    payload_layout: str,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    import vllm._custom_ops as ops

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_size % 16 or head_size_v % 16:
        raise ValueError("ByteV2 benchmark requires 16-aligned head sizes")

    layout = _make_layout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        compressed_only=compressed_only,
        payload_layout=payload_layout,
    )
    pages_per_request = math.ceil(seq_len / block_size)
    num_blocks = batch_size * pages_per_request
    num_tokens = batch_size * seq_len

    key, value = _make_compressible_kv(
        num_tokens,
        num_kv_heads,
        head_size,
        head_size_v,
        device=device,
        seed=seed,
    )
    requested_fallback_blocks = _inject_fallback_blocks(
        key,
        value,
        block_size=block_size,
        fallback_ratio=fallback_ratio,
        fallback_pattern=fallback_pattern,
        outlier_exp_scale=outlier_exp_scale,
        seed=seed,
    )

    kv_cache = torch.zeros(
        num_blocks, layout.page_size_bytes, device=device, dtype=torch.uint8
    )
    slot_mapping = torch.empty(num_tokens, device=device, dtype=torch.int64)
    for req_idx in range(batch_size):
        physical_block_base = req_idx * pages_per_request
        token_base = req_idx * seq_len
        slots = (
            physical_block_base * block_size
            + torch.arange(seq_len, device=device, dtype=torch.int64)
        )
        slot_mapping[token_base:token_base + seq_len] = slots

    raw_bytes = _raw_block_bytes(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
    )
    fallback_pool = None
    fallback_block_ids = None
    fallback_next_slot = None
    fallback_tile_ids = None
    fallback_tile_next_slot = None
    outlier_arena = None
    outlier_block_flags = None
    outlier_tile_bitmap = None
    outlier_tile_meta = None
    outlier_next_entry = None
    if compressed_only and fallback_ratio > 0.0:
        pool_blocks = min(
            num_blocks,
            max(requested_fallback_blocks + 4,
                math.ceil(num_blocks * fallback_ratio * 2.0)),
        )
        fallback_pool = torch.empty(
            pool_blocks, raw_bytes, device=device, dtype=torch.uint8
        )
        fallback_block_ids = torch.full(
            (num_blocks,), -1, device=device, dtype=torch.int32
        )
        fallback_next_slot = torch.zeros(1, device=device, dtype=torch.int32)
        if tile_fallback_pool:
            fallback_tile_ids = torch.full(
                (num_blocks, layout.total_tiles),
                -1,
                device=device,
                dtype=torch.int32,
            )
            fallback_tile_next_slot = torch.zeros(
                1, device=device, dtype=torch.int32
            )

    use_outlier_arena = (
        compressed_only
        and (
            outlier_arena_entries_per_block > 0.0
            or outlier_arena_min_entries > 0
        )
    )
    if use_outlier_arena:
        outlier_entries = max(
            int(math.ceil(num_blocks * outlier_arena_entries_per_block)),
            outlier_arena_min_entries,
        )
        if outlier_entries <= 0:
            raise ValueError("outlier arena must have at least one entry")
        outlier_arena = torch.full(
            (outlier_entries,), -1, device=device, dtype=torch.int32
        )
        if use_outlier_block_flags:
            outlier_block_flags = torch.zeros(
                num_blocks, device=device, dtype=torch.int32
            )
        if use_outlier_tile_bitmap:
            bitmap_words = (layout.total_tiles + 31) // 32
            outlier_tile_bitmap = torch.zeros(
                (num_blocks, bitmap_words), device=device, dtype=torch.int32
            )
        outlier_tile_meta = torch.full(
            (num_blocks, layout.total_tiles),
            -1,
            device=device,
            dtype=torch.int32,
        )
        outlier_next_entry = torch.zeros(1, device=device, dtype=torch.int32)

    packed = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        layout.page_size_bytes,
        fallback_pool,
        fallback_block_ids,
        fallback_next_slot,
        fallback_tile_ids,
        fallback_tile_next_slot,
        None,
        outlier_arena,
        outlier_block_flags,
        outlier_tile_bitmap,
        outlier_tile_meta,
        outlier_next_entry,
    )
    torch.cuda.synchronize(device)

    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(batch_size, pages_per_request)
    seq_lens = torch.full((batch_size,), seq_len, device=device,
                          dtype=torch.int32)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 23)
    query = (0.5 + 0.01 * torch.randn(
        batch_size,
        num_heads,
        head_size,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )).to(torch.bfloat16)

    actual_fallback_blocks = 0
    fallback_capacity = 0
    actual_tile_fallbacks = 0
    tile_fallback_capacity = 0
    actual_outlier_entries = 0
    actual_outlier_blocks = 0
    actual_outlier_bitmap_tiles = 0
    actual_outlier_tiles = 0
    outlier_capacity = 0
    if fallback_next_slot is not None:
        actual_fallback_blocks = int(fallback_next_slot.cpu().item())
        fallback_capacity = int(fallback_pool.shape[0])
    if fallback_tile_next_slot is not None:
        actual_tile_fallbacks = int(fallback_tile_next_slot.cpu().item())
        tile_fallback_capacity = (
            int(fallback_pool.numel()) // (16 * 16 * 2)
            if fallback_pool is not None
            else 0
        )
    if outlier_next_entry is not None:
        actual_outlier_entries = int(outlier_next_entry.cpu().item())
        outlier_capacity = int(outlier_arena.numel())
    if outlier_tile_meta is not None:
        actual_outlier_tiles = int((outlier_tile_meta >= 0).sum().cpu().item())
    if outlier_block_flags is not None:
        actual_outlier_blocks = int(
            (outlier_block_flags != 0).sum().cpu().item()
        )
    if outlier_tile_bitmap is not None:
        actual_outlier_bitmap_tiles = int(
            sum(
                (int(word) & 0xFFFFFFFF).bit_count()
                for word in outlier_tile_bitmap.cpu().view(-1).tolist()
            )
        )

    return {
        "layout": layout,
        "key": key.reshape(batch_size, seq_len, num_kv_heads, head_size),
        "value": value.reshape(batch_size, seq_len, num_kv_heads, head_size_v),
        "kv_cache": kv_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "query": query,
        "fallback_pool": fallback_pool,
        "fallback_block_ids": fallback_block_ids,
        "fallback_next_slot": fallback_next_slot,
        "fallback_tile_ids": fallback_tile_ids,
        "fallback_tile_next_slot": fallback_tile_next_slot,
        "outlier_arena": outlier_arena,
        "outlier_block_flags": outlier_block_flags,
        "outlier_tile_bitmap": outlier_tile_bitmap,
        "outlier_tile_meta": outlier_tile_meta,
        "outlier_next_entry": outlier_next_entry,
        "packed_blocks": int(packed.numel()),
        "requested_fallback_blocks": requested_fallback_blocks,
        "actual_fallback_blocks": actual_fallback_blocks,
        "fallback_capacity": fallback_capacity,
        "actual_tile_fallbacks": actual_tile_fallbacks,
        "tile_fallback_capacity": tile_fallback_capacity,
        "actual_outlier_entries": actual_outlier_entries,
        "actual_outlier_blocks": actual_outlier_blocks,
        "actual_outlier_bitmap_tiles": actual_outlier_bitmap_tiles,
        "actual_outlier_tiles": actual_outlier_tiles,
        "outlier_capacity": outlier_capacity,
        "pages_per_request": pages_per_request,
        "num_blocks": num_blocks,
    }


def _raw_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    q_per_kv = query.shape[1] // key.shape[2]
    expanded_key = key.repeat_interleave(q_per_kv, dim=2)
    expanded_value = value.repeat_interleave(q_per_kv, dim=2)
    scores = torch.einsum(
        "bhd,bshd->bhs", query.float(), expanded_key.float()
    ) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshv->bhv", probs, expanded_value.float()).to(
        query.dtype
    )


def _run_decode(
    inputs: dict[str, Any],
    *,
    scale: float,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    partial_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    import vllm._custom_ops as ops

    layout = inputs["layout"]
    return ops.byte_v2_paged_decode_attention(
        inputs["query"],
        inputs["kv_cache"],
        inputs["block_table"],
        inputs["seq_lens"],
        scale,
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        layout.page_size_bytes,
        inputs["fallback_pool"],
        inputs["fallback_block_ids"],
        inputs["fallback_tile_ids"],
        partial_workspace,
        inputs["outlier_arena"],
        inputs["outlier_block_flags"],
        inputs["outlier_tile_bitmap"],
        inputs["outlier_tile_meta"],
    )


def _measure_us(
    inputs: dict[str, Any],
    *,
    scale: float,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    warmup_runs: int,
    num_runs: int,
    device: torch.device,
    partial_workspace: torch.Tensor | None,
) -> tuple[list[float], torch.Tensor]:
    last_output = None
    for _ in range(warmup_runs):
        last_output = _run_decode(
            inputs,
            scale=scale,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            partial_workspace=partial_workspace,
        )
    torch.cuda.synchronize(device)

    times_us: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(num_runs):
        start_event.record()
        last_output = _run_decode(
            inputs,
            scale=scale,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            partial_workspace=partial_workspace,
        )
        end_event.record()
        torch.cuda.synchronize(device)
        times_us.append(float(start_event.elapsed_time(end_event)) * 1000.0)

    assert last_output is not None
    return times_us, last_output


def _summarize_times(times_us: list[float]) -> dict[str, float]:
    sorted_times = sorted(times_us)

    def percentile(p: float) -> float:
        if not sorted_times:
            return float("nan")
        idx = min(len(sorted_times) - 1, max(0, math.ceil(p * len(sorted_times)) - 1))
        return sorted_times[idx]

    return {
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "p50_us": percentile(0.50),
        "p90_us": percentile(0.90),
        "p99_us": percentile(0.99),
        "min_us": min(times_us),
        "max_us": max(times_us),
    }


def run_case(
    *,
    args: argparse.Namespace,
    seq_len: int,
    split_k: int,
    fallback_ratio: float,
    device: torch.device,
) -> dict[str, Any]:
    if args.variant == "page_fastpath":
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH", None)
    if args.cute_stage1:
        os.environ["VLLM_BYTE_V2_DECODE_CUTE_STAGE1"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_CUTE_STAGE1", None)
    if args.cute_stage1_auto:
        os.environ["VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO", None)
    if args.fast_stage1:
        os.environ["VLLM_BYTE_V2_DECODE_FAST_STAGE1"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_FAST_STAGE1", None)
    if args.flash_stage1:
        os.environ["VLLM_BYTE_V2_DECODE_FLASH_STAGE1"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_FLASH_STAGE1", None)
    if args.v4_stage1:
        os.environ["VLLM_BYTE_V2_DECODE_V4_STAGE1"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES"] = str(
            args.v4_macro_pages
        )
        if args.v4_block128:
            os.environ["VLLM_BYTE_V2_DECODE_V4_BLOCK128"] = "1"
        else:
            os.environ.pop("VLLM_BYTE_V2_DECODE_V4_BLOCK128", None)
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_V4_STAGE1", None)
        os.environ.pop("VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES", None)
        os.environ.pop("VLLM_BYTE_V2_DECODE_V4_BLOCK128", None)
    if args.cute_stage1_early_exit_mode > 0:
        os.environ["VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT"] = str(
            args.cute_stage1_early_exit_mode
        )
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT", None)
    if args.aligned_u16_payload_load:
        os.environ["VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD", None)
    payload_layout = "v3" if args.v4_stage1 else args.payload_layout
    os.environ["VLLM_BYTE_V2_PAYLOAD_LAYOUT"] = payload_layout
    if args.v3_warp_stripe_load:
        os.environ["VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD", None)
    if args.v3_cp_async_stage:
        os.environ["VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE", None)
    if args.v3_outlier_only_no_fallback:
        os.environ["VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK", None)
    tile_fastpath_mode = args.tile_fastpath_mode
    if args.tile_fastpath:
        tile_fastpath_mode = "on"
    if tile_fastpath_mode == "on":
        os.environ["VLLM_BYTE_V2_DECODE_TILE_FASTPATH"] = "1"
    elif tile_fastpath_mode == "off":
        os.environ["VLLM_BYTE_V2_DECODE_TILE_FASTPATH"] = "0"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_TILE_FASTPATH", None)
    parallel_reduce_mode = args.parallel_reduce_mode
    if args.parallel_reduce:
        parallel_reduce_mode = "on"
    if parallel_reduce_mode == "on":
        os.environ["VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE"] = "1"
    elif parallel_reduce_mode == "off":
        os.environ["VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE"] = "0"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE", None)
    os.environ["VLLM_BYTE_V2_DECODE_SPLIT_K"] = str(split_k)
    os.environ["VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE"] = str(
        max(0, args.outlier_max_per_tile)
    )
    use_tile_fallback_pool = (
        args.tile_fallback_pool
        or (
            not args.v3_outlier_only_no_fallback
            and (
                args.outlier_arena_entries_per_block > 0.0
                or args.outlier_arena_min_entries > 0
            )
        )
    )
    inputs = _build_inputs(
        batch_size=args.batch_size,
        seq_len=seq_len,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
        block_size=args.block_size,
        compressed_only=not args.raw_overlay_pages,
        fallback_ratio=fallback_ratio,
        fallback_pattern=args.fallback_pattern,
        outlier_exp_scale=args.outlier_exp_scale,
        tile_fallback_pool=use_tile_fallback_pool,
        outlier_arena_entries_per_block=args.outlier_arena_entries_per_block,
        outlier_arena_min_entries=args.outlier_arena_min_entries,
        use_outlier_block_flags=args.use_outlier_block_flags,
        use_outlier_tile_bitmap=args.use_outlier_tile_bitmap,
        payload_layout=payload_layout,
        device=device,
        seed=args.seed,
    )
    partial_workspace = None
    if args.partial_workspace:
        pages_per_request = inputs["pages_per_request"]
        num_kv_splits = (
            min(split_k, pages_per_request)
            if split_k > 1 and pages_per_request >= 16
            else 1
        )
        if args.num_heads // args.num_kv_heads > 1 and num_kv_splits > 1:
            partial_workspace = torch.empty(
                args.batch_size
                * args.num_heads
                * num_kv_splits
                * (args.head_size_v + 1),
                dtype=torch.float32,
                device=device,
            )
    times_us, output = _measure_us(
        inputs,
        scale=args.scale,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        partial_workspace=partial_workspace,
    )
    timing = _summarize_times(times_us)

    max_abs_diff = None
    max_rel_diff = None
    correctness_elapsed_s = None
    if not args.skip_correctness:
        start = time.perf_counter()
        expected = _raw_attention_reference(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            scale=args.scale,
        )
        torch.cuda.synchronize(device)
        correctness_elapsed_s = time.perf_counter() - start
        diff = (output.float() - expected.float()).abs()
        denom = expected.float().abs().clamp_min(1e-6)
        max_abs_diff = float(diff.max().item())
        max_rel_diff = float((diff / denom).max().item())

    output_tokens_per_s = (
        args.batch_size * 1_000_000.0 / timing["median_us"]
        if timing["median_us"] > 0
        else float("inf")
    )
    return {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "q_per_kv": args.num_heads // args.num_kv_heads,
        "head_size": args.head_size,
        "head_size_v": args.head_size_v,
        "block_size": args.block_size,
        "pages_per_request": inputs["pages_per_request"],
        "num_blocks": inputs["num_blocks"],
        "page_size_bytes": inputs["layout"].page_size_bytes,
        "payload_layout": payload_layout,
        "split_k": split_k,
        "page_fastpath": (
            os.environ.get("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH") == "1"
        ),
        "cute_stage1": args.cute_stage1,
        "cute_stage1_auto": args.cute_stage1_auto,
        "cute_stage1_early_exit_mode": args.cute_stage1_early_exit_mode,
        "aligned_u16_payload_load": args.aligned_u16_payload_load,
        "v3_warp_stripe_load": args.v3_warp_stripe_load,
        "v3_cp_async_stage": args.v3_cp_async_stage,
        "v3_outlier_only_no_fallback": args.v3_outlier_only_no_fallback,
        "fast_stage1": args.fast_stage1,
        "flash_stage1": args.flash_stage1,
        "v4_stage1": args.v4_stage1,
        "v4_macro_pages": args.v4_macro_pages,
        "v4_block128": args.v4_block128,
        "tile_fastpath": tile_fastpath_mode != "off",
        "tile_fastpath_mode": tile_fastpath_mode,
        "tile_fastpath_env": os.environ.get(
            "VLLM_BYTE_V2_DECODE_TILE_FASTPATH"
        ),
        "parallel_reduce_mode": parallel_reduce_mode,
        "parallel_reduce_env": os.environ.get(
            "VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE"
        ),
        "fallback_ratio": fallback_ratio,
        "fallback_pattern": args.fallback_pattern,
        "outlier_exp_scale": args.outlier_exp_scale,
        "tile_fallback_pool": use_tile_fallback_pool,
        "outlier_max_per_tile": args.outlier_max_per_tile,
        "outlier_arena_entries_per_block": (
            args.outlier_arena_entries_per_block
        ),
        "outlier_arena_min_entries": args.outlier_arena_min_entries,
        "use_outlier_block_flags": args.use_outlier_block_flags,
        "use_outlier_tile_bitmap": args.use_outlier_tile_bitmap,
        "compressed_only": not args.raw_overlay_pages,
        "external_partial_workspace": partial_workspace is not None,
        "packed_blocks": inputs["packed_blocks"],
        "requested_fallback_blocks": inputs["requested_fallback_blocks"],
        "actual_fallback_blocks": inputs["actual_fallback_blocks"],
        "fallback_capacity": inputs["fallback_capacity"],
        "actual_tile_fallbacks": inputs["actual_tile_fallbacks"],
        "tile_fallback_capacity": inputs["tile_fallback_capacity"],
        "actual_outlier_entries": inputs["actual_outlier_entries"],
        "actual_outlier_blocks": inputs["actual_outlier_blocks"],
        "actual_outlier_bitmap_tiles": inputs["actual_outlier_bitmap_tiles"],
        "actual_outlier_tiles": inputs["actual_outlier_tiles"],
        "outlier_capacity": inputs["outlier_capacity"],
        "fallback_pool_exhausted": (
            inputs["fallback_capacity"] > 0
            and inputs["actual_fallback_blocks"] >= inputs["fallback_capacity"]
        ),
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "effective_output_tok_s": output_tokens_per_s,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correctness_elapsed_s": correctness_elapsed_s,
        **timing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native ByteV2 paged decode kernels."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", default="512")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--payload-layout", choices=("v1", "v3"), default="v1")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--fallback-ratio", default="0.0")
    parser.add_argument(
        "--fallback-pattern",
        choices=(
            "window17",
            "single_outlier",
            "single_outlier_per_k_tile",
        ),
        default="window17",
        help=(
            "Pattern used for selected fallback-ratio blocks. window17 keeps "
            "the original 17-exponent stress pattern; single_outlier injects "
            "one high-exponent K value per selected block; "
            "single_outlier_per_k_tile injects one K outlier into each K tile "
            "of every selected block."
        ),
    )
    parser.add_argument(
        "--outlier-exp-scale",
        type=float,
        default=1.0e20,
        help="Float value used by --fallback-pattern single_outlier.",
    )
    parser.add_argument(
        "--tile-fallback-pool",
        action="store_true",
        help="Use tile-level sparse fallback metadata instead of block fallback.",
    )
    parser.add_argument(
        "--outlier-max-per-tile",
        type=int,
        default=0,
        help=(
            "Maximum element-level outliers encoded per tile. Requires an "
            "outlier arena to affect cache update; max=0 still profiles the "
            "decode overlay path when arena tensors are allocated."
        ),
    )
    parser.add_argument(
        "--outlier-arena-entries-per-block",
        type=float,
        default=0.0,
        help="Outlier arena entries allocated per physical block.",
    )
    parser.add_argument(
        "--outlier-arena-min-entries",
        type=int,
        default=0,
        help="Minimum outlier arena entries allocated for the benchmark.",
    )
    parser.add_argument(
        "--use-outlier-block-flags",
        action="store_true",
        help=(
            "Pass per-block has-outlier flags to native decode. This helps "
            "only when many blocks have no compact outlier entries."
        ),
    )
    parser.add_argument(
        "--use-outlier-tile-bitmap",
        action="store_true",
        help=(
            "Pass per-tile has-outlier bitmap to native decode. This skips "
            "outlier_tile_meta reads for compressed tiles without overlay "
            "entries."
        ),
    )
    parser.add_argument("--split-k", default="1,2,4,8,16")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument(
        "--cute-stage1",
        action="store_true",
        help=(
            "Use the experimental CUTE-style split-K stage1. This also "
            "enables page fastpath."
        ),
    )
    parser.add_argument(
        "--cute-stage1-auto",
        action="store_true",
        help=(
            "Use the conservative auto heuristic for the CUTE-style split-K "
            "stage1. This also enables page fastpath."
        ),
    )
    parser.add_argument(
        "--fast-stage1",
        action="store_true",
        help=(
            "Use the strict no-fallback fixed-shape split-K stage1. This also "
            "enables page fastpath."
        ),
    )
    parser.add_argument(
        "--flash-stage1",
        action="store_true",
        help=(
            "Use the FlashInfer-style warp-per-query no-fallback fixed-shape "
            "split-K stage1. This also enables page fastpath."
        ),
    )
    parser.add_argument(
        "--v4-stage1",
        action="store_true",
        help=(
            "Use the V3-only no-fallback FlashAttention-style V4 split-K "
            "stage1. This also enables page fastpath and V3 payload layout."
        ),
    )
    parser.add_argument(
        "--v4-macro-pages",
        choices=(4, 8),
        type=int,
        default=8,
        help="Number of 16-token pages per V4 macro descriptor.",
    )
    parser.add_argument(
        "--v4-block128",
        action="store_true",
        help=(
            "Use the experimental V4 block128 score-buffer/two-pass stage1. "
            "Requires --v4-stage1 and --v4-macro-pages 8."
        ),
    )
    parser.add_argument(
        "--cute-stage1-early-exit-mode",
        type=int,
        choices=range(0, 8),
        default=0,
        metavar="{0,1,2,3,4,5,6,7}",
        help=(
            "Profile-only CUTE stage1 early exit: 0/4=full, 1=load/decode, "
            "2=load/decode+QK, 3=load/decode+QK+softmax, "
            "5=metadata/status traversal, 6=metadata+payload read, "
            "7=metadata+payload read+register decode. Modes 1-3 and 5-7 "
            "produce invalid attention output and should be used with "
            "--skip-correctness."
        ),
    )
    parser.add_argument(
        "--aligned-u16-payload-load",
        action="store_true",
        help=(
            "Profile-only payload read experiment: use direct aligned uint16 "
            "loads for ByteV2 low bytes in the CUTE stage1 compressed tile "
            "fastpath."
        ),
    )
    parser.add_argument(
        "--v3-warp-stripe-load",
        action="store_true",
        help=(
            "V3-only payload read experiment: use 24 aligned uint32 loads per "
            "96B stripe plus warp shuffles instead of per-lane scalar byte "
            "loads."
        ),
    )
    parser.add_argument(
        "--v3-cp-async-stage",
        action="store_true",
        help=(
            "V3-only CUTE stage1 experiment: stage 384B compressed tile "
            "payloads through shared memory using cp.async and a two-buffer "
            "dim-tile pipeline."
        ),
    )
    parser.add_argument(
        "--v3-outlier-only-no-fallback",
        action="store_true",
        help=(
            "V3 experiment: encode exponent-window misses into the outlier "
            "arena and do not allocate/use tile fallback metadata."
        ),
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-overlay-pages", action="store_true")
    parser.add_argument("--partial-workspace", action="store_true")
    parser.add_argument("--parallel-reduce", action="store_true")
    parser.add_argument(
        "--parallel-reduce-mode",
        choices=("auto", "off", "on"),
        default="auto",
    )
    parser.add_argument("--tile-fastpath", action="store_true")
    parser.add_argument(
        "--tile-fastpath-mode",
        choices=("auto", "off", "on"),
        default="auto",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native ByteV2 kernel benchmark")

    # The benchmark must time the native extension, not the PyTorch fallback.
    os.environ["VLLM_BYTE_V2_USE_NATIVE_KERNELS"] = "1"
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices")
    device_index = 0 if device.index is None else device.index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    if args.scale is None:
        args.scale = 1.0 / math.sqrt(args.head_size)

    seq_lens = _parse_csv_ints(args.seq_len)
    split_ks = _parse_csv_ints(args.split_k)
    fallback_ratios = _parse_csv_floats(args.fallback_ratio)
    results = []
    for fallback_ratio in fallback_ratios:
        for seq_len in seq_lens:
            for split_k in split_ks:
                result = run_case(
                    args=args,
                    seq_len=seq_len,
                    split_k=split_k,
                    fallback_ratio=fallback_ratio,
                    device=device,
                )
                results.append(result)
                print(
                    "seq_len={seq_len} split_k={split_k} fallback={fallback:.3f} "
                    "median_us={median:.2f} p90_us={p90:.2f} tok/s={tps:.2f} "
                    "fallback_blocks={fb}/{cap} fallback_tiles={tiles}/{tile_cap} "
                    "outlier_blocks={outlier_blocks} "
                    "outlier_bitmap_tiles={outlier_bitmap_tiles} "
                    "outlier_tiles={outlier_tiles} "
                    "outlier_entries={outlier_entries}/{outlier_cap} "
                    "max_abs_diff={diff}".format(
                        seq_len=seq_len,
                        split_k=split_k,
                        fallback=fallback_ratio,
                        median=result["median_us"],
                        p90=result["p90_us"],
                        tps=result["effective_output_tok_s"],
                        fb=result["actual_fallback_blocks"],
                        cap=result["fallback_capacity"],
                        tiles=result["actual_tile_fallbacks"],
                        tile_cap=result["tile_fallback_capacity"],
                        outlier_blocks=result["actual_outlier_blocks"],
                        outlier_bitmap_tiles=(
                            result["actual_outlier_bitmap_tiles"]
                        ),
                        outlier_tiles=result["actual_outlier_tiles"],
                        outlier_entries=result["actual_outlier_entries"],
                        outlier_cap=result["outlier_capacity"],
                        diff=result["max_abs_diff"],
                    ),
                    flush=True,
                )

    payload = {
        "benchmark": "byte_v2_decode_kernel",
        "device": torch.cuda.get_device_name(device),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
