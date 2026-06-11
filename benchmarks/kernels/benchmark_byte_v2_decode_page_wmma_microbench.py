# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark ByteV2 compressed-page decode plus WMMA QK/PV.

This is the B2b baseline: it reads real ByteV2 compressed pages, decodes K/V
tiles into shared memory, and runs the same synthetic QK/PV math as
``benchmark_byte_v2_wmma_microbench.py``. It does not include softmax,
split-K, fallback metadata, scheduler, or production decode plumbing.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return ordered[idx]

    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": min(values),
        "max": max(values),
    }


def _build_compressed_pages(
    *,
    num_pages: int,
    num_kv_heads: int,
    seed: int,
) -> torch.Tensor:
    from vllm.v1.attention.backends.byte_v2_layout import (
        ByteV2PageLayout,
        pack_byte_v2_kv_block_to_page,
    )

    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        raw_tail_bytes=0,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    key = (
        1.0
        + 0.01
        * torch.randn(
            num_pages,
            16,
            num_kv_heads,
            128,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)
    value = (
        2.0
        + 0.01
        * torch.randn(
            num_pages,
            16,
            num_kv_heads,
            128,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)
    kv_cache = torch.zeros(num_pages, layout.page_size_bytes, dtype=torch.uint8)
    for page_idx in range(num_pages):
        pack_byte_v2_kv_block_to_page(
            key[page_idx], value[page_idx], layout, page=kv_cache[page_idx]
        )
    return kv_cache


def run_case(
    *,
    num_pages: int,
    num_kv_heads: int,
    kv_head: int,
    repeat_count: int,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import _custom_ops as ops

    device = torch.device("cuda")
    kv_cache_cpu = _build_compressed_pages(
        num_pages=num_pages,
        num_kv_heads=num_kv_heads,
        seed=seed,
    )
    kv_cache = kv_cache_cpu.to(device=device, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    query = (
        0.125
        * torch.randn(
            num_pages,
            16,
            128,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)

    for _ in range(warmup):
        ops.byte_v2_decode_page_wmma_microbench(
            query,
            kv_cache,
            num_kv_heads=num_kv_heads,
            kv_head=kv_head,
            page_size_bytes=kv_cache.shape[1],
            repeat_count=repeat_count,
        )
    torch.cuda.synchronize()

    elapsed_us: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ops.byte_v2_decode_page_wmma_microbench(
            query,
            kv_cache,
            num_kv_heads=num_kv_heads,
            kv_head=kv_head,
            page_size_bytes=kv_cache.shape[1],
            repeat_count=repeat_count,
        )
        end.record()
        end.synchronize()
        elapsed_us.append(start.elapsed_time(end) * 1000.0)

    summary = _summarize(elapsed_us)
    summary["per_page_repeat_us"] = (
        summary["median"] / max(1, num_pages * repeat_count)
    )
    return {
        "num_pages": num_pages,
        "num_kv_heads": num_kv_heads,
        "kv_head": kv_head,
        "repeat_count": repeat_count,
        "warmup": warmup,
        "iterations": iterations,
        "page_size_bytes": kv_cache.shape[1],
        "summary_us": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("BF16 WMMA requires Ampere or newer")
    if args.kv_head < 0 or args.kv_head >= args.num_kv_heads:
        raise ValueError("kv-head must be in [0, num-kv-heads)")

    result = {
        "benchmark": "byte_v2_decode_page_wmma_microbench",
        "device": torch.cuda.get_device_name(),
        "case": run_case(
            num_pages=args.num_pages,
            num_kv_heads=args.num_kv_heads,
            kv_head=args.kv_head,
            repeat_count=args.repeat_count,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
