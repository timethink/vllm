# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the ByteV2 16x128 WMMA QK/PV data path.

This isolates the shared-memory layout plus WMMA path from paged metadata,
ByteV2 decode, softmax, split-K, and vLLM scheduler overhead. It is intended
for B2 CUTE/CUTLASS-style layout experiments.
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


def run_case(
    *,
    num_tiles: int,
    repeat_count: int,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import _custom_ops as ops

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    query = (
        0.125
        * torch.randn(
            num_tiles,
            16,
            128,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)
    key = (
        0.125
        * torch.randn(
            num_tiles,
            16,
            128,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)
    value = (
        0.125
        * torch.randn(
            num_tiles,
            16,
            128,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)

    for _ in range(warmup):
        ops.byte_v2_wmma_layout_microbench(
            query, key, value, variant=0, repeat_count=repeat_count
        )
    torch.cuda.synchronize()

    elapsed_us: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ops.byte_v2_wmma_layout_microbench(
            query, key, value, variant=0, repeat_count=repeat_count
        )
        end.record()
        end.synchronize()
        elapsed_us.append(start.elapsed_time(end) * 1000.0)

    summary = _summarize(elapsed_us)
    summary["per_tile_repeat_us"] = (
        summary["median"] / max(1, num_tiles * repeat_count)
    )
    return {
        "num_tiles": num_tiles,
        "variant": 0,
        "repeat_count": repeat_count,
        "warmup": warmup,
        "iterations": iterations,
        "summary_us": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tiles", type=int, default=256)
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

    result = {
        "benchmark": "byte_v2_wmma_layout_microbench",
        "device": torch.cuda.get_device_name(),
        "case": run_case(
            num_tiles=args.num_tiles,
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
