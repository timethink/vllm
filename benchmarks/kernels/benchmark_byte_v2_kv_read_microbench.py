# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming-read benchmark for raw KV cache vs ByteV2 compressed pages.

This microbenchmark intentionally does not decode ByteV2 payloads or run
attention. It answers one narrow question: if the kernel only has to stream KV
cache bytes from HBM, does the smaller ByteV2 page layout reduce read time?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _stream_i32_sum_kernel(
    data,
    out,
    n_words: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = offsets < n_words
    values = tl.load(data + offsets, mask=mask, other=0)
    summed = tl.sum(values, axis=0)
    tl.store(out + pid, summed)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _make_layout(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
):
    from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayout

    return ByteV2PageLayout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        raw_tail_bytes=0,
    )


def _time_kernel(
    data: torch.Tensor,
    *,
    requested_bytes: int,
    read_block_bytes: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    word_bytes = data.element_size()
    if requested_bytes % word_bytes:
        raise ValueError("requested_bytes must be divisible by tensor element size")
    if read_block_bytes % word_bytes:
        raise ValueError("read_block_bytes must be divisible by tensor element size")
    block_words = read_block_bytes // word_bytes
    if not _is_power_of_two(block_words):
        raise ValueError("read_block_bytes / word_bytes must be a power of two")

    n_words = requested_bytes // word_bytes
    grid = (triton.cdiv(n_words, block_words),)
    out = torch.empty(grid[0], dtype=torch.int32, device=data.device)

    for _ in range(warmup):
        _stream_i32_sum_kernel[grid](
            data,
            out,
            n_words,
            BLOCK_WORDS=block_words,
            num_warps=8,
        )
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        _stream_i32_sum_kernel[grid](
            data,
            out,
            n_words,
            BLOCK_WORDS=block_words,
            num_warps=8,
        )
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    checksum = int(out[: min(1024, out.numel())].sum().item())
    median_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    p90_ms = sorted(times_ms)[math.ceil(0.9 * len(times_ms)) - 1]
    gb = requested_bytes / 1e9
    return {
        "requested_bytes": requested_bytes,
        "read_block_bytes": read_block_bytes,
        "programs": grid[0],
        "median_ms": median_ms,
        "min_ms": min_ms,
        "p90_ms": p90_ms,
        "median_gb_per_s": gb / (median_ms / 1000.0),
        "min_time_gb_per_s": gb / (min_ms / 1000.0),
        "checksum_sample": checksum,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.block_size != 16:
        raise ValueError("ByteV2 layout currently requires block_size=16")

    device = torch.device("cuda")
    layout = _make_layout(
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
    )
    raw_block_bytes = layout.raw_block_bytes
    compressed_page_bytes = layout.page_size_bytes
    if raw_block_bytes % 4 or compressed_page_bytes % 4:
        raise ValueError("raw/compressed byte counts must be int32 aligned")

    raw_bytes = args.num_blocks * raw_block_bytes
    compressed_bytes = args.num_blocks * compressed_page_bytes
    raw = torch.empty(raw_bytes // 4, dtype=torch.int32, device=device)
    compressed = torch.empty(compressed_bytes // 4, dtype=torch.int32,
                             device=device)

    # Touch allocations once so the timed region measures steady-state reads.
    raw.zero_()
    compressed.zero_()
    torch.cuda.synchronize()

    results: list[dict[str, Any]] = []
    for read_block_bytes in _parse_csv_ints(args.read_block_bytes):
        raw_result = _time_kernel(
            raw,
            requested_bytes=raw_bytes,
            read_block_bytes=read_block_bytes,
            warmup=args.warmup,
            iters=args.iters,
        )
        compressed_result = _time_kernel(
            compressed,
            requested_bytes=compressed_bytes,
            read_block_bytes=read_block_bytes,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append({
            "read_block_bytes": read_block_bytes,
            "raw": raw_result,
            "compressed": compressed_result,
            "compressed_time_over_raw": (
                compressed_result["median_ms"] / raw_result["median_ms"]
            ),
            "compressed_bytes_over_raw": compressed_bytes / raw_bytes,
        })

    return {
        "benchmark": "byte_v2_kv_read_microbench",
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "num_kv_heads": args.num_kv_heads,
        "head_size": args.head_size,
        "head_size_v": args.head_size_v,
        "raw_block_bytes": raw_block_bytes,
        "compressed_page_bytes": compressed_page_bytes,
        "compressed_page_over_raw_block": compressed_page_bytes
        / raw_block_bytes,
        "raw_total_bytes": raw_bytes,
        "compressed_total_bytes": compressed_bytes,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark raw KV vs ByteV2 compressed-page HBM reads.")
    parser.add_argument("--num-blocks", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--read-block-bytes", type=str,
                        default="1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    for item in result["results"]:
        raw = item["raw"]
        compressed = item["compressed"]
        print(
            "read_block_bytes={rb} raw={raw_ms:.4f}ms "
            "({raw_gbps:.1f} GB/s) compressed={cmp_ms:.4f}ms "
            "({cmp_gbps:.1f} GB/s) time_ratio={tr:.3f} "
            "bytes_ratio={br:.3f}".format(
                rb=item["read_block_bytes"],
                raw_ms=raw["median_ms"],
                raw_gbps=raw["median_gb_per_s"],
                cmp_ms=compressed["median_ms"],
                cmp_gbps=compressed["median_gb_per_s"],
                tr=item["compressed_time_over_raw"],
                br=item["compressed_bytes_over_raw"],
            ))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
