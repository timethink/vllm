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
):
    from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayout

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
    if fallback_next_slot is not None:
        actual_fallback_blocks = int(fallback_next_slot.cpu().item())
        fallback_capacity = int(fallback_pool.shape[0])

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
        "packed_blocks": int(packed.numel()),
        "requested_fallback_blocks": requested_fallback_blocks,
        "actual_fallback_blocks": actual_fallback_blocks,
        "fallback_capacity": fallback_capacity,
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
        None,
        partial_workspace,
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
        "split_k": split_k,
        "page_fastpath": (
            os.environ.get("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH") == "1"
        ),
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
        "compressed_only": not args.raw_overlay_pages,
        "external_partial_workspace": partial_workspace is not None,
        "packed_blocks": inputs["packed_blocks"],
        "requested_fallback_blocks": inputs["requested_fallback_blocks"],
        "actual_fallback_blocks": inputs["actual_fallback_blocks"],
        "fallback_capacity": inputs["fallback_capacity"],
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
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--fallback-ratio", default="0.0")
    parser.add_argument(
        "--fallback-pattern",
        choices=("window17", "single_outlier"),
        default="window17",
        help=(
            "Pattern used for selected fallback-ratio blocks. window17 keeps "
            "the original 17-exponent stress pattern; single_outlier injects "
            "one high-exponent K value per selected block."
        ),
    )
    parser.add_argument(
        "--outlier-exp-scale",
        type=float,
        default=1.0e20,
        help="Float value used by --fallback-pattern single_outlier.",
    )
    parser.add_argument("--split-k", default="1,2,4,8,16")
    parser.add_argument("--variant", default="baseline")
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
                    "fallback_blocks={fb}/{cap} max_abs_diff={diff}".format(
                        seq_len=seq_len,
                        split_k=split_k,
                        fallback=fallback_ratio,
                        median=result["median_us"],
                        p90=result["p90_us"],
                        tps=result["effective_output_tok_s"],
                        fb=result["actual_fallback_blocks"],
                        cap=result["fallback_capacity"],
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
