# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle benchmark for ByteV2 decompress-to-BF16 + raw FlashInfer decode.

This benchmark is intentionally outside the production path. It answers one
question from the ByteV2 optimization plan: would it be faster to decode
compressed ByteV2 pages into a temporary BF16 paged KV workspace and then reuse
FlashInfer's raw paged decode kernel?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from benchmark_byte_v2_decode_kernel import (
    _build_inputs,
    _parse_csv_floats,
    _parse_csv_ints,
    _raw_attention_reference,
    _run_decode,
    _summarize_times,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def _measure_us(
    fn: Callable[[], torch.Tensor | None],
    *,
    warmup_runs: int,
    num_runs: int,
    device: torch.device,
    nvtx_label: str | None = None,
) -> tuple[list[float], torch.Tensor | None]:
    last_output = None
    for _ in range(warmup_runs):
        last_output = fn()
    torch.cuda.synchronize(device)

    times_us: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(num_runs):
        start_event.record()
        if nvtx_label is None:
            last_output = fn()
        else:
            torch.cuda.nvtx.range_push(nvtx_label)
            try:
                last_output = fn()
            finally:
                torch.cuda.nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize(device)
        times_us.append(float(start_event.elapsed_time(end_event)) * 1000.0)
    return times_us, last_output


def _make_flashinfer_wrapper(
    *,
    batch_size: int,
    pages_per_request: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    block_table: torch.Tensor,
    last_page_len: torch.Tensor,
    scale: float,
    workspace_mb: int,
    use_tensor_cores: bool,
    device: torch.device,
) -> BatchDecodeWithPagedKVCacheWrapper:
    workspace = torch.empty(
        workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
        backend="auto",
    )
    indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_request,
        pages_per_request,
        device=device,
        dtype=torch.int32,
    )
    wrapper.plan(
        indptr,
        block_table.reshape(-1).contiguous().to(torch.int32),
        last_page_len,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        data_type=torch.bfloat16,
        sm_scale=scale,
    )
    return wrapper


def _set_fused_decode_env(args: argparse.Namespace) -> None:
    os.environ["VLLM_BYTE_V2_USE_NATIVE_KERNELS"] = "1"
    if args.fused_cute_stage1_auto:
        os.environ["VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO", None)
    if args.fused_cute_stage1:
        os.environ["VLLM_BYTE_V2_DECODE_CUTE_STAGE1"] = "1"
        os.environ["VLLM_BYTE_V2_DECODE_PAGE_FASTPATH"] = "1"
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_CUTE_STAGE1", None)
    if args.fused_split_k is not None:
        os.environ["VLLM_BYTE_V2_DECODE_SPLIT_K"] = str(args.fused_split_k)
    else:
        os.environ.pop("VLLM_BYTE_V2_DECODE_SPLIT_K", None)
    os.environ["VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE"] = str(
        max(0, args.outlier_max_per_tile)
    )


def run_case(
    *,
    args: argparse.Namespace,
    seq_len: int,
    fallback_ratio: float,
    device: torch.device,
) -> dict[str, Any]:
    _set_fused_decode_env(args)
    import vllm._custom_ops as ops

    if args.head_size != args.head_size_v:
        raise ValueError(
            "FlashInfer oracle currently requires head_size == head_size_v"
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
        tile_fallback_pool=args.tile_fallback_pool
        or args.outlier_arena_entries_per_block > 0.0
        or args.outlier_arena_min_entries > 0,
        outlier_arena_entries_per_block=args.outlier_arena_entries_per_block,
        outlier_arena_min_entries=args.outlier_arena_min_entries,
        use_outlier_block_flags=args.use_outlier_block_flags,
        use_outlier_tile_bitmap=args.use_outlier_tile_bitmap,
        device=device,
        seed=args.seed,
    )
    pages_per_request = inputs["pages_per_request"]
    num_blocks = inputs["num_blocks"]
    last_page = ((seq_len - 1) % args.block_size) + 1
    last_page_len = torch.full(
        (args.batch_size,), last_page, device=device, dtype=torch.int32
    )

    key_workspace = torch.empty(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    value_workspace = torch.empty(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_size_v,
        device=device,
        dtype=torch.bfloat16,
    )
    flashinfer_output = torch.empty(
        args.batch_size,
        args.num_heads,
        args.head_size_v,
        device=device,
        dtype=torch.bfloat16,
    )
    wrapper = _make_flashinfer_wrapper(
        batch_size=args.batch_size,
        pages_per_request=pages_per_request,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        block_table=inputs["block_table"],
        last_page_len=last_page_len,
        scale=args.scale,
        workspace_mb=args.flashinfer_workspace_mb,
        use_tensor_cores=not args.disable_flashinfer_tensor_cores,
        device=device,
    )

    def decompress_once() -> None:
        ops.byte_v2_decompress_cache_to_bf16(
            inputs["kv_cache"],
            key_workspace,
            value_workspace,
            args.block_size,
            args.num_kv_heads,
            args.head_size,
            args.head_size_v,
            inputs["layout"].page_size_bytes,
            inputs["fallback_pool"],
            inputs["fallback_block_ids"],
            inputs["fallback_tile_ids"],
            inputs["outlier_arena"],
            inputs["outlier_tile_bitmap"],
            inputs["outlier_tile_meta"],
        )
        return None

    def flashinfer_once() -> torch.Tensor:
        return wrapper.run(
            inputs["query"],
            (key_workspace, value_workspace),
            out=flashinfer_output,
        )

    def oracle_once() -> torch.Tensor:
        decompress_once()
        return flashinfer_once()

    def fused_once() -> torch.Tensor:
        return _run_decode(
            inputs,
            scale=args.scale,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            head_size_v=args.head_size_v,
            partial_workspace=None,
        )

    decompress_times, _ = _measure_us(
        decompress_once,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        nvtx_label="experiment_b_decompress_only",
    )
    decompress_once()
    torch.cuda.synchronize(device)
    flashinfer_times, flashinfer_last = _measure_us(
        flashinfer_once,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        nvtx_label="experiment_b_flashinfer_only",
    )
    oracle_times, oracle_last = _measure_us(
        oracle_once,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        nvtx_label="experiment_b_oracle_total",
    )
    fused_times, fused_last = _measure_us(
        fused_once,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        nvtx_label="experiment_b_fused_bytev2",
    )

    decomp_key_diff = None
    decomp_value_diff = None
    oracle_abs_diff = None
    fused_abs_diff = None
    if not args.skip_correctness:
        decompress_once()
        torch.cuda.synchronize(device)
        key_by_req = key_workspace.reshape(
            args.batch_size,
            pages_per_request * args.block_size,
            args.num_kv_heads,
            args.head_size,
        )[:, :seq_len]
        value_by_req = value_workspace.reshape(
            args.batch_size,
            pages_per_request * args.block_size,
            args.num_kv_heads,
            args.head_size_v,
        )[:, :seq_len]
        decomp_key_diff = float(
            (key_by_req.float() - inputs["key"].float()).abs().max().item()
        )
        decomp_value_diff = float(
            (value_by_req.float() - inputs["value"].float()).abs().max().item()
        )
        expected = _raw_attention_reference(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            scale=args.scale,
        )
        if oracle_last is None:
            oracle_last = oracle_once()
        if fused_last is None:
            fused_last = fused_once()
        if flashinfer_last is None:
            flashinfer_last = flashinfer_once()
        torch.cuda.synchronize(device)
        oracle_abs_diff = float(
            (oracle_last.float() - expected.float()).abs().max().item()
        )
        fused_abs_diff = float(
            (fused_last.float() - expected.float()).abs().max().item()
        )

    decompress = _summarize_times(decompress_times)
    flashinfer = _summarize_times(flashinfer_times)
    oracle = _summarize_times(oracle_times)
    fused = _summarize_times(fused_times)
    return {
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "q_per_kv": args.num_heads // args.num_kv_heads,
        "head_size": args.head_size,
        "head_size_v": args.head_size_v,
        "block_size": args.block_size,
        "pages_per_request": pages_per_request,
        "num_blocks": num_blocks,
        "page_size_bytes": inputs["layout"].page_size_bytes,
        "fallback_ratio": fallback_ratio,
        "fallback_pattern": args.fallback_pattern,
        "tile_fallback_pool": args.tile_fallback_pool,
        "outlier_arena_entries_per_block": (
            args.outlier_arena_entries_per_block
        ),
        "outlier_arena_min_entries": args.outlier_arena_min_entries,
        "use_outlier_tile_bitmap": args.use_outlier_tile_bitmap,
        "raw_overlay_pages": args.raw_overlay_pages,
        "flashinfer_tensor_cores": not args.disable_flashinfer_tensor_cores,
        "fused_cute_stage1": args.fused_cute_stage1,
        "fused_cute_stage1_auto": args.fused_cute_stage1_auto,
        "fused_split_k": args.fused_split_k,
        "packed_blocks": inputs["packed_blocks"],
        "requested_fallback_blocks": inputs["requested_fallback_blocks"],
        "actual_fallback_blocks": inputs["actual_fallback_blocks"],
        "fallback_capacity": inputs["fallback_capacity"],
        "actual_tile_fallbacks": inputs["actual_tile_fallbacks"],
        "actual_outlier_entries": inputs["actual_outlier_entries"],
        "actual_outlier_tiles": inputs["actual_outlier_tiles"],
        "decompress": decompress,
        "flashinfer": flashinfer,
        "oracle_total": oracle,
        "fused_bytev2": fused,
        "oracle_component_sum_median_us": (
            decompress["median_us"] + flashinfer["median_us"]
        ),
        "oracle_total_vs_fused_median": (
            oracle["median_us"] / fused["median_us"]
            if fused["median_us"] > 0
            else float("inf")
        ),
        "decompress_key_max_abs_diff": decomp_key_diff,
        "decompress_value_max_abs_diff": decomp_value_diff,
        "oracle_max_abs_diff": oracle_abs_diff,
        "fused_max_abs_diff": fused_abs_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ByteV2 decompress-to-BF16 + FlashInfer oracle."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", default="512")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--fallback-ratio", default="0.03")
    parser.add_argument(
        "--fallback-pattern",
        choices=(
            "window17",
            "single_outlier",
            "single_outlier_per_k_tile",
        ),
        default="window17",
    )
    parser.add_argument("--outlier-exp-scale", type=float, default=1.0e20)
    parser.add_argument("--tile-fallback-pool", action="store_true")
    parser.add_argument("--outlier-max-per-tile", type=int, default=0)
    parser.add_argument("--outlier-arena-entries-per-block", type=float, default=0.0)
    parser.add_argument("--outlier-arena-min-entries", type=int, default=0)
    parser.add_argument("--use-outlier-block-flags", action="store_true")
    parser.add_argument("--use-outlier-tile-bitmap", action="store_true")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-overlay-pages", action="store_true")
    parser.add_argument("--fused-cute-stage1", action="store_true")
    parser.add_argument(
        "--no-fused-cute-stage1-auto",
        dest="fused_cute_stage1_auto",
        action="store_false",
    )
    parser.set_defaults(fused_cute_stage1_auto=True)
    parser.add_argument("--fused-split-k", type=int, default=None)
    parser.add_argument("--disable-flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=128)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices")
    device_index = 0 if device.index is None else device.index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    if args.scale is None:
        args.scale = 1.0 / math.sqrt(args.head_size)

    results = []
    for fallback_ratio in _parse_csv_floats(args.fallback_ratio):
        for seq_len in _parse_csv_ints(args.seq_len):
            result = run_case(
                args=args,
                seq_len=seq_len,
                fallback_ratio=fallback_ratio,
                device=device,
            )
            results.append(result)
            print(
                "seq_len={seq_len} batch={batch} fallback={fallback:.3f} "
                "fused={fused:.2f}us decomp={decomp:.2f}us "
                "flashinfer={flashinfer:.2f}us total={total:.2f}us "
                "oracle/fused={ratio:.3f} key_diff={key_diff} "
                "oracle_diff={oracle_diff}".format(
                    seq_len=seq_len,
                    batch=args.batch_size,
                    fallback=fallback_ratio,
                    fused=result["fused_bytev2"]["median_us"],
                    decomp=result["decompress"]["median_us"],
                    flashinfer=result["flashinfer"]["median_us"],
                    total=result["oracle_total"]["median_us"],
                    ratio=result["oracle_total_vs_fused_median"],
                    key_diff=result["decompress_key_max_abs_diff"],
                    oracle_diff=result["oracle_max_abs_diff"],
                )
            )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
    if results:
        ratios = [item["oracle_total_vs_fused_median"] for item in results]
        print(f"median oracle/fused ratio: {statistics.median(ratios):.3f}")


if __name__ == "__main__":
    main()
