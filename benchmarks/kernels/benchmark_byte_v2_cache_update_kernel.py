# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark ByteV2 KV cache update kernels.

This script isolates ``byte_v2_reshape_and_cache`` from the vLLM scheduler and
model layers. It is intended for cache-update experiments: run it before and
after one change, compare prefill and decode-append latency, and keep the change
only if the improvement is stable.
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


def _make_compressible_kv(
    num_tokens: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    *,
    device: torch.device,
    seed: int,
    outlier_block_ratio: float = 0.0,
    outlier_exp_scale: float = 1.0e20,
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
    if outlier_block_ratio > 0.0 and num_tokens > 0:
        num_blocks = math.ceil(num_tokens / 16)
        num_outlier_blocks = min(
            num_blocks, max(1, math.ceil(num_blocks * outlier_block_ratio))
        )
        block_ids = torch.randperm(
            num_blocks, device=device, generator=generator
        )[:num_outlier_blocks]
        token_ids = block_ids * 16
        token_ids = token_ids[token_ids < num_tokens]
        key[token_ids, 0, 0] = outlier_exp_scale
    return key.to(torch.bfloat16), value.to(torch.bfloat16)


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


def _empty_cache_state(
    *,
    num_blocks: int,
    page_size_bytes: int,
    raw_block_bytes: int,
    fallback_pool_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_cache = torch.zeros(
        num_blocks, page_size_bytes, device=device, dtype=torch.uint8
    )
    fallback_pool = torch.empty(
        fallback_pool_blocks, raw_block_bytes, device=device, dtype=torch.uint8
    )
    fallback_block_ids = torch.full(
        (num_blocks,), -1, device=device, dtype=torch.int32
    )
    fallback_next_slot = torch.zeros(1, device=device, dtype=torch.int32)
    return kv_cache, fallback_pool, fallback_block_ids, fallback_next_slot


def _run_update(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor,
    fallback_block_ids: torch.Tensor,
    fallback_next_slot: torch.Tensor,
) -> torch.Tensor:
    import vllm._custom_ops as ops

    return ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        page_size_bytes,
        fallback_pool,
        fallback_block_ids,
        fallback_next_slot,
    )


def _measure_prefill_case(
    *,
    args: argparse.Namespace,
    prompt_len: int,
    layout: Any,
    device: torch.device,
) -> dict[str, Any]:
    key, value = _make_compressible_kv(
        prompt_len,
        args.num_kv_heads,
        args.head_size,
        args.head_size_v,
        device=device,
        seed=args.seed + prompt_len,
        outlier_block_ratio=args.outlier_block_ratio,
        outlier_exp_scale=args.outlier_exp_scale,
    )
    slot_mapping = torch.arange(prompt_len, dtype=torch.int64, device=device)
    fallback_pool_blocks = min(
        args.num_blocks,
        max(1, args.fallback_pool_min_blocks,
            math.ceil(args.num_blocks * args.fallback_pool_ratio)),
    )

    cuda_times_us: list[float] = []
    wall_times_us: list[float] = []
    packed_blocks = 0
    fallback_used = 0
    for run_idx in range(args.warmup_runs + args.num_runs):
        kv_cache, fallback_pool, fallback_block_ids, fallback_next_slot = (
            _empty_cache_state(
                num_blocks=args.num_blocks,
                page_size_bytes=layout.page_size_bytes,
                raw_block_bytes=layout.raw_block_bytes,
                fallback_pool_blocks=fallback_pool_blocks,
                device=device,
            )
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_wall = time.perf_counter()
        start_event.record()
        packed = _run_update(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            head_size_v=args.head_size_v,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
        )
        end_event.record()
        torch.cuda.synchronize(device)
        wall_us = (time.perf_counter() - start_wall) * 1_000_000.0
        cuda_us = float(start_event.elapsed_time(end_event)) * 1000.0
        if run_idx >= args.warmup_runs:
            cuda_times_us.append(cuda_us)
            wall_times_us.append(wall_us)
        packed_blocks = int(packed.numel())
        fallback_used = int(fallback_next_slot.cpu().item())

    return {
        "mode": "prefill",
        "prompt_len": prompt_len,
        "num_blocks": args.num_blocks,
        "fallback_pool_blocks": fallback_pool_blocks,
        "packed_blocks": packed_blocks,
        "fallback_used": fallback_used,
        "cuda_us": _summarize(cuda_times_us),
        "wall_us": _summarize(wall_times_us),
    }


def _measure_decode_append_case(
    *,
    args: argparse.Namespace,
    decode_steps: int,
    layout: Any,
    device: torch.device,
) -> dict[str, Any]:
    key, value = _make_compressible_kv(
        decode_steps,
        args.num_kv_heads,
        args.head_size,
        args.head_size_v,
        device=device,
        seed=args.seed + 1009 + decode_steps,
        outlier_block_ratio=args.outlier_block_ratio,
        outlier_exp_scale=args.outlier_exp_scale,
    )
    slot_mapping = torch.arange(decode_steps, dtype=torch.int64, device=device)
    fallback_pool_blocks = min(
        args.num_blocks,
        max(1, args.fallback_pool_min_blocks,
            math.ceil(args.num_blocks * args.fallback_pool_ratio)),
    )

    cuda_times_us: list[float] = []
    wall_times_us: list[float] = []
    packed_blocks = 0
    fallback_used = 0
    for run_idx in range(args.warmup_runs + args.num_runs):
        kv_cache, fallback_pool, fallback_block_ids, fallback_next_slot = (
            _empty_cache_state(
                num_blocks=args.num_blocks,
                page_size_bytes=layout.page_size_bytes,
                raw_block_bytes=layout.raw_block_bytes,
                fallback_pool_blocks=fallback_pool_blocks,
                device=device,
            )
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_wall = time.perf_counter()
        start_event.record()
        last_packed = None
        for step in range(decode_steps):
            last_packed = _run_update(
                key=key[step:step + 1],
                value=value[step:step + 1],
                kv_cache=kv_cache,
                slot_mapping=slot_mapping[step:step + 1],
                block_size=args.block_size,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                head_size_v=args.head_size_v,
                page_size_bytes=layout.page_size_bytes,
                fallback_pool=fallback_pool,
                fallback_block_ids=fallback_block_ids,
                fallback_next_slot=fallback_next_slot,
            )
        end_event.record()
        torch.cuda.synchronize(device)
        wall_us = (time.perf_counter() - start_wall) * 1_000_000.0
        cuda_us = float(start_event.elapsed_time(end_event)) * 1000.0
        if run_idx >= args.warmup_runs:
            cuda_times_us.append(cuda_us)
            wall_times_us.append(wall_us)
        packed_blocks = 0 if last_packed is None else int(last_packed.numel())
        fallback_used = int(fallback_next_slot.cpu().item())

    per_step_cuda = [value / decode_steps for value in cuda_times_us]
    per_step_wall = [value / decode_steps for value in wall_times_us]
    return {
        "mode": "decode_append",
        "decode_steps": decode_steps,
        "num_blocks": args.num_blocks,
        "fallback_pool_blocks": fallback_pool_blocks,
        "last_call_packed_blocks": packed_blocks,
        "fallback_used": fallback_used,
        "cuda_us": _summarize(cuda_times_us),
        "wall_us": _summarize(wall_times_us),
        "per_step_cuda_us": _summarize(per_step_cuda),
        "per_step_wall_us": _summarize(per_step_wall),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native ByteV2 KV cache update kernels."
    )
    parser.add_argument("--mode", default="prefill,decode_append")
    parser.add_argument("--prompt-len", default="2048")
    parser.add_argument("--decode-steps", default="32")
    parser.add_argument("--num-blocks", type=int, default=8192)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--fallback-pool-ratio", type=float, default=0.03)
    parser.add_argument("--fallback-pool-min-blocks", type=int, default=512)
    parser.add_argument(
        "--outlier-block-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of logical 16-token blocks that receive one high-exponent "
            "K outlier in the first K tile. Used to test lossy fallback "
            "reduction; default keeps the original compressible benchmark."
        ),
    )
    parser.add_argument(
        "--outlier-exp-scale",
        type=float,
        default=1.0e20,
        help="Float32 value used for injected high-exponent K outliers.",
    )
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native ByteV2 kernel benchmark")
    if args.block_size != 16:
        raise ValueError("ByteV2 cache update benchmark requires block_size=16")

    os.environ["VLLM_BYTE_V2_USE_NATIVE_KERNELS"] = "1"
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices")
    device_index = 0 if device.index is None else device.index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    layout = _make_layout(
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
    )
    modes = {item.strip() for item in args.mode.split(",") if item.strip()}
    results: list[dict[str, Any]] = []

    if "prefill" in modes:
        for prompt_len in _parse_csv_ints(args.prompt_len):
            result = _measure_prefill_case(
                args=args, prompt_len=prompt_len, layout=layout, device=device
            )
            results.append(result)
            print(
                "mode=prefill prompt_len={prompt_len} "
                "median_cuda_us={cuda:.2f} median_wall_us={wall:.2f} "
                "packed={packed} fallback={fallback}/{capacity}".format(
                    prompt_len=prompt_len,
                    cuda=result["cuda_us"]["median"],
                    wall=result["wall_us"]["median"],
                    packed=result["packed_blocks"],
                    fallback=result["fallback_used"],
                    capacity=result["fallback_pool_blocks"],
                ),
                flush=True,
            )

    if "decode_append" in modes:
        for decode_steps in _parse_csv_ints(args.decode_steps):
            result = _measure_decode_append_case(
                args=args,
                decode_steps=decode_steps,
                layout=layout,
                device=device,
            )
            results.append(result)
            print(
                "mode=decode_append steps={steps} "
                "median_cuda_us={cuda:.2f} per_step_cuda_us={step_cuda:.2f} "
                "median_wall_us={wall:.2f} per_step_wall_us={step_wall:.2f} "
                "fallback={fallback}/{capacity}".format(
                    steps=decode_steps,
                    cuda=result["cuda_us"]["median"],
                    step_cuda=result["per_step_cuda_us"]["median"],
                    wall=result["wall_us"]["median"],
                    step_wall=result["per_step_wall_us"]["median"],
                    fallback=result["fallback_used"],
                    capacity=result["fallback_pool_blocks"],
                ),
                flush=True,
            )

    payload = {
        "benchmark": "byte_v2_cache_update_kernel",
        "device": torch.cuda.get_device_name(device),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "layout": {
            "page_size_bytes": layout.page_size_bytes,
            "raw_block_bytes": layout.raw_block_bytes,
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
