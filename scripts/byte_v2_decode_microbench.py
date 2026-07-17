# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark isolated ByteV2 paged decode kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

DIRECT_DIAGNOSTIC_MODE_IDS = {
    "current": 0,
    "decode-stage-only": 1,
    "fake-decode-zero": 2,
    "raw-same-skeleton": 3,
    "k-decode-stage-only": 4,
    "v-decode-stage-only": 5,
    "k-decode-qk-only": 6,
    "v-decode-pv-only": 7,
    "k-decode-producer3-stage-only": 8,
    "v-decode-producer3-stage-only": 9,
    "k-decode-producer2-stage-only": 10,
    "v-decode-producer2-stage-only": 11,
    "k-decode-no-store-stage-only": 12,
    "v-decode-no-store-stage-only": 13,
    "k-decode-low-only-stage-only": 14,
    "v-decode-low-only-stage-only": 15,
    "k-decode-high-only-stage-only": 16,
    "v-decode-high-only-stage-only": 17,
    "phase-profile": 18,
    "stage-window16": 19,
    "stage-window32": 20,
    "stage-window16-specialized": 21,
    "qk-window16-profile": 22,
    "effective-m16-profile": 23,
    "v-decode-pv-no-gemm-only": 24,
    "v-decode-pv-no-accum-only": 25,
    "k-decode-qk-no-output": 26,
    "unnormalized-partition-output": 27,
}

PHASE_PROFILE_KEYS = (
    "total_cycles",
    "stage_cycles",
    "qk_cycles",
    "softmax_cycles",
    "pv_cycles",
    "stage_wait_warp0_cycles",
    "stage_wait_warp1_cycles",
    "stage_wait_warp2_cycles",
    "stage_wait_warp3_cycles",
    "compute_wait_warp0_cycles",
    "compute_wait_warp1_cycles",
    "compute_wait_warp2_cycles",
    "compute_wait_warp3_cycles",
    "tile_count",
    "compute_block_n_profiled",
    "partition_size_profiled",
    "compute_region_warp0_cycles",
    "compute_region_warp1_cycles",
    "compute_region_warp2_cycles",
    "compute_region_warp3_cycles",
    "qk_window_count",
    "qk_window_dim",
    "qk_windows_per_tile",
)


def _prepend_venv_bin_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    venv_bin = repo_root / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="KV lengths to benchmark.",
    )
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--partition-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--compute-block-ns",
        nargs="+",
        type=int,
        default=[64],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--include-single", action="store_true")
    parser.add_argument("--include-flash", action="store_true")
    parser.add_argument("--fa-version", type=int, default=None)
    parser.add_argument(
        "--no-outlier-inputs",
        action="store_true",
        help="Construct ByteV2 inputs whose codec tiles do not need outliers.",
    )
    parser.add_argument(
        "--force-outlier-inputs",
        action="store_true",
        help=(
            "Construct inputs that exercise ByteV2 outlier overlays even when "
            "--assume-no-outlier is set for guarded fast-path dispatch."
        ),
    )
    parser.add_argument(
        "--assume-no-outlier",
        action="store_true",
        help=(
            "Use ByteV2 no-fallback/no-outlier decode specialization and "
            "assume inputs satisfy that condition."
        ),
    )
    parser.add_argument(
        "--gqa-packed",
        action="store_true",
        help="Use the experimental GQA4 packed ByteV2 split-k decode kernel.",
    )
    parser.add_argument(
        "--gqa-fa2-like",
        action="store_true",
        help=(
            "Use the p64-only experimental GQA4 FA2-like split-k kernel. "
            "Requires --gqa-packed."
        ),
    )
    parser.add_argument(
        "--gqa-fa2-qk-mma",
        action="store_true",
        help=(
            "Enable the experimental QK-MMA subpath inside the FA2-like "
            "GQA4 kernel. Requires --gqa-fa2-like."
        ),
    )
    parser.add_argument(
        "--gqa-fa2-mainloop",
        action="store_true",
        help=(
            "Enable the experimental FA2-style single-warp QK/softmax/PV "
            "mainloop inside the FA2-like GQA4 kernel. Requires "
            "--gqa-fa2-qk-mma."
        ),
    )
    parser.add_argument(
        "--gqa-fa2-multiwarp",
        action="store_true",
        help=(
            "Enable the experimental FA2-style multi-warp PV mainloop inside "
            "the FA2-like GQA4 kernel. Requires --gqa-fa2-qk-mma."
        ),
    )
    parser.add_argument(
        "--gqa-fa2-direct",
        action="store_true",
        help=(
            "Enable the experimental direct FA2-style 4-warp M-layout path "
            "inside the FA2-like GQA4 kernel. Requires --gqa-fa2-qk-mma."
        ),
    )
    parser.add_argument(
        "--high-byte-payload",
        action="store_true",
        help=(
            "Use the experimental low+raw-high-byte ByteV2 payload format. "
            "This is an upper-bound format/kernel co-design experiment and "
            "currently requires --gqa-fa2-direct."
        ),
    )
    parser.add_argument(
        "--sideband-high-payload",
        action="store_true",
        help=(
            "Use the experimental tile-local outlier high-byte sideband format. "
            "Safe tiles keep the default fixed payload; outlier tiles read high "
            "bytes from the outlier payload sideband. Requires guarded "
            "FA2-direct decode."
        ),
    )
    parser.add_argument(
        "--guarded-split",
        action="store_true",
        help=("Use the split-k no-outlier fast path guarded by per-page unsafe flags."),
    )
    parser.add_argument(
        "--direct-diagnostic-modes",
        nargs="+",
        choices=tuple(DIRECT_DIAGNOSTIC_MODE_IDS),
        default=["current"],
        help=(
            "FA2-direct diagnostic variants to benchmark. Non-current modes "
            "are benchmark-only and currently require q_per_kv=4."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_decode_microbench.jsonl",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Wrap timed iterations in cudaProfilerStart/Stop for Nsight "
            "Compute --profile-from-start off."
        ),
    )
    return parser.parse_args()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _stats_ms(samples: list[float]) -> dict[str, float]:
    sorted_samples = sorted(samples)
    p90_index = min(len(sorted_samples) - 1, int(0.9 * (len(sorted_samples) - 1)))
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": sorted_samples[p90_index],
    }


def _time_cuda_kernel(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    cuda_profiler_range: bool = False,
) -> tuple[dict[str, float], Any]:
    import torch

    result = None
    for _ in range(warmup):
        result = fn()
    torch.accelerator.synchronize()

    events = []
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        events.append((start_event, end_event))
    torch.accelerator.synchronize()
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()

    return _stats_ms([start.elapsed_time(end) for start, end in events]), result


def _max_abs_diff(lhs: Any, rhs: Any) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _workspace_mib(*tensors: Any) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / 1024 / 1024


def _make_inputs(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    import torch

    from vllm.v1.attention.backends.byte_v2_layout import (
        ByteV2CodecPayloadPolicy,
        ByteV2PageLayoutV4,
        ByteV2RawStagingLayout,
    )
    from vllm.v1.attention.backends.byte_v2_ops import (
        byte_v2_reshape_and_cache,
        byte_v2_reshape_and_cache_high_byte,
        byte_v2_reshape_and_cache_sideband_high,
        byte_v2_update_cache_unsafe_flags,
    )

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if args.block_size != 16:
        raise ValueError("ByteV2 decode kernels currently require block_size=16")

    device = torch.device("cuda")
    blocks_per_seq = _ceil_div(seq_len, args.block_size)
    num_blocks = args.num_seqs * blocks_per_seq
    total_tokens = args.num_seqs * seq_len

    token_base = torch.arange(
        total_tokens * args.num_kv_heads * args.head_size,
        dtype=torch.float32,
        device=device,
    )
    use_no_outlier_inputs = (
        args.no_outlier_inputs or args.assume_no_outlier
    ) and not args.force_outlier_inputs
    key_values = (
        ((token_base % 255) + 1) / 1024
        if use_no_outlier_inputs
        else (token_base % 257) / 1024
    )
    value_values = (
        (((token_base + 17) % 255) + 1) / 1024
        if use_no_outlier_inputs
        else ((token_base + 17) % 263) / 1024
    )
    key = key_values.reshape(total_tokens, args.num_kv_heads, args.head_size).to(
        torch.bfloat16
    )
    value = value_values.reshape(total_tokens, args.num_kv_heads, args.head_size).to(
        torch.bfloat16
    )
    query_base = torch.arange(
        args.num_seqs * args.num_heads * args.head_size,
        dtype=torch.float32,
        device=device,
    )
    query = (
        ((query_base % 127) / 31)
        .reshape(args.num_seqs, args.num_heads, args.head_size)
        .to(torch.bfloat16)
    )

    slot_ranges = []
    table_rows = []
    for seq_idx in range(args.num_seqs):
        block_start = seq_idx * blocks_per_seq
        slot_start = block_start * args.block_size
        slot_ranges.append(
            slot_start + torch.arange(seq_len, dtype=torch.int64, device=device)
        )
        table_rows.append(
            torch.arange(
                block_start,
                block_start + blocks_per_seq,
                dtype=torch.int32,
                device=device,
            )
        )
    slot_mapping = torch.cat(slot_ranges)
    block_tables = torch.stack(table_rows)
    seq_lens = torch.full(
        (args.num_seqs,),
        seq_len,
        dtype=torch.int32,
        device=device,
    )

    if args.high_byte_payload:
        layout = ByteV2PageLayoutV4(
            codec_payload_policy=ByteV2CodecPayloadPolicy(exponent_code_bits=8),
            num_kv_heads=args.num_kv_heads,
        )
    elif args.sideband_high_payload:
        layout = ByteV2PageLayoutV4(
            codec_payload_policy=ByteV2CodecPayloadPolicy(
                outlier_high_sideband=True,
            ),
            num_kv_heads=args.num_kv_heads,
        )
    else:
        layout = ByteV2PageLayoutV4(num_kv_heads=args.num_kv_heads)
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device=device,
    )
    if args.high_byte_payload:
        byte_v2_reshape_and_cache_high_byte(
            key,
            value,
            kv_cache,
            slot_mapping,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )
    elif args.sideband_high_payload:
        byte_v2_reshape_and_cache_sideband_high(
            key,
            value,
            kv_cache,
            slot_mapping,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )
    else:
        byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )
    page_unsafe_flags = None
    if args.guarded_split:
        page_unsafe_flags = torch.empty(
            (num_blocks,),
            dtype=torch.int32,
            device=device,
        )
        byte_v2_update_cache_unsafe_flags(
            page_unsafe_flags,
            kv_cache,
            slot_mapping,
            tile_policy=(16, 16, 16, 64, args.head_size, args.head_size),
        )

    raw_key_cache = torch.zeros(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    raw_value_cache = torch.zeros_like(raw_key_cache)
    raw_key_cache.view(-1, args.num_kv_heads, args.head_size)[slot_mapping] = key
    raw_value_cache.view(-1, args.num_kv_heads, args.head_size)[slot_mapping] = value
    raw_layout = ByteV2RawStagingLayout(num_kv_heads=args.num_kv_heads)
    raw_skeleton_stride = max(layout.page_size_bytes, raw_layout.slot_size_bytes)
    raw_skeleton_cache = torch.empty(
        (num_blocks, raw_skeleton_stride),
        dtype=torch.uint8,
        device=device,
    )
    raw_key_bytes = (
        raw_key_cache.permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(num_blocks, -1)
    )
    raw_value_bytes = (
        raw_value_cache.permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(num_blocks, -1)
    )
    raw_skeleton_cache[:, : raw_layout.key_bytes].copy_(raw_key_bytes)
    raw_skeleton_cache[
        :,
        raw_layout.value_base_bytes : raw_layout.value_base_bytes
        + raw_layout.value_bytes,
    ].copy_(raw_value_bytes)

    return {
        "query": query,
        "kv_cache": kv_cache,
        "raw_skeleton_cache": raw_skeleton_cache,
        "raw_key_cache": raw_key_cache,
        "raw_value_cache": raw_value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "page_unsafe_flags": page_unsafe_flags,
        "scale": args.head_size**-0.5,
        "blocks_per_seq": blocks_per_seq,
        "num_blocks": num_blocks,
    }


def _byte_v2_split_fn(
    args: argparse.Namespace,
    tensors: dict[str, Any],
    *,
    seq_len: int,
    partition_size: int,
    compute_block_n: int,
    direct_diagnostic_mode: str = "current",
):
    import torch

    from vllm.v1.attention.backends.byte_v2_ops import (
        byte_v2_paged_decode_attention_split_k,
        byte_v2_paged_decode_attention_split_k_guarded,
    )

    num_partitions = _ceil_div(seq_len, partition_size)
    output = torch.empty_like(tensors["query"])
    exp_sums = torch.empty(
        args.num_seqs,
        args.num_heads,
        num_partitions,
        dtype=torch.float32,
        device=tensors["query"].device,
    )
    if direct_diagnostic_mode in {
        "phase-profile",
        "stage-window16",
        "stage-window32",
        "stage-window16-specialized",
        "qk-window16-profile",
        "effective-m16-profile",
        "unnormalized-partition-output",
    }:
        max_logits = torch.empty_like(exp_sums)
    else:
        max_logits = torch.empty(
            (0,),
            dtype=torch.float32,
            device=tensors["query"].device,
        )
    tmp_out = torch.empty(
        args.num_seqs,
        args.num_heads,
        num_partitions,
        args.head_size,
        dtype=torch.float32,
        device=tensors["query"].device,
    )
    if args.gqa_packed:
        tile_policy = (
            16,
            16,
            16,
            compute_block_n,
            args.head_size,
            args.head_size,
            0,
            1,
            1,
        )
        if args.gqa_fa2_like:
            tile_policy += (1,)
            if args.gqa_fa2_qk_mma:
                tile_policy += (1,)
                if args.gqa_fa2_multiwarp:
                    tile_policy += (0, 1)
                elif args.gqa_fa2_mainloop:
                    tile_policy += (1,)
                elif args.gqa_fa2_direct:
                    tile_policy += (0, 0, 1)
                    diagnostic_mode_id = DIRECT_DIAGNOSTIC_MODE_IDS[
                        direct_diagnostic_mode
                    ]
                    if diagnostic_mode_id:
                        tile_policy += (diagnostic_mode_id,)
    elif args.assume_no_outlier:
        tile_policy = (
            16,
            16,
            16,
            compute_block_n,
            args.head_size,
            args.head_size,
            0,
            1,
        )
    else:
        tile_policy = (16, 16, 16, compute_block_n, args.head_size, args.head_size)
    if args.high_byte_payload or args.sideband_high_payload:
        if len(tile_policy) == 14:
            tile_policy += (0,)
        tile_policy += (1 if args.high_byte_payload else 2,)

    if direct_diagnostic_mode == "raw-same-skeleton":
        kv_cache = tensors["raw_skeleton_cache"]
    else:
        kv_cache = tensors["kv_cache"]

    def run():
        if args.guarded_split:
            byte_v2_paged_decode_attention_split_k_guarded(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                tensors["query"],
                kv_cache,
                tensors["page_unsafe_flags"],
                tensors["block_tables"],
                tensors["seq_lens"],
                scale=tensors["scale"],
                num_kv_heads=args.num_kv_heads,
                block_size=args.block_size,
                max_seq_len=seq_len,
                partition_size=partition_size,
                tile_policy=tile_policy,
            )
        else:
            byte_v2_paged_decode_attention_split_k(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                tensors["query"],
                kv_cache,
                tensors["block_tables"],
                tensors["seq_lens"],
                scale=tensors["scale"],
                num_kv_heads=args.num_kv_heads,
                block_size=args.block_size,
                max_seq_len=seq_len,
                partition_size=partition_size,
                tile_policy=tile_policy,
            )
        return output

    if direct_diagnostic_mode in {
        "phase-profile",
        "stage-window16",
        "stage-window32",
        "stage-window16-specialized",
        "qk-window16-profile",
        "effective-m16-profile",
    }:
        run.phase_profile_tensor = max_logits  # type: ignore[attr-defined]

    return run, output, _workspace_mib(exp_sums, max_logits, tmp_out)


def _byte_v2_single_fn(
    args: argparse.Namespace,
    tensors: dict[str, Any],
    *,
    seq_len: int,
    compute_block_n: int,
):
    import torch

    from vllm.v1.attention.backends.byte_v2_ops import byte_v2_paged_decode_attention

    output = torch.empty_like(tensors["query"])
    if args.gqa_packed:
        tile_policy = (
            16,
            16,
            16,
            compute_block_n,
            args.head_size,
            args.head_size,
            0,
            1,
            1,
        )
    elif args.assume_no_outlier:
        tile_policy = (
            16,
            16,
            16,
            compute_block_n,
            args.head_size,
            args.head_size,
            0,
            1,
        )
    else:
        tile_policy = (16, 16, 16, compute_block_n, args.head_size, args.head_size)

    def run():
        byte_v2_paged_decode_attention(
            output,
            tensors["query"],
            tensors["kv_cache"],
            tensors["block_tables"],
            tensors["seq_lens"],
            scale=tensors["scale"],
            num_kv_heads=args.num_kv_heads,
            block_size=args.block_size,
            max_seq_len=seq_len,
            tile_policy=tile_policy,
        )
        return output

    return run, output, 0.0


def _flash_paged_fn(
    args: argparse.Namespace,
    tensors: dict[str, Any],
    *,
    seq_len: int,
    fa_version: int,
):
    import torch

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    output = torch.empty_like(tensors["query"])
    cu_seqlens_q = torch.arange(
        0,
        args.num_seqs + 1,
        dtype=torch.int32,
        device=tensors["query"].device,
    )

    def run():
        return flash_attn_varlen_func(
            q=tensors["query"],
            k=tensors["raw_key_cache"],
            v=tensors["raw_value_cache"],
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=tensors["seq_lens"],
            block_table=tensors["block_tables"],
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            softmax_scale=tensors["scale"],
            causal=True,
            fa_version=fa_version,
        )

    return run, output, 0.0


def _benchmark_one(args: argparse.Namespace, seq_len: int) -> list[dict[str, Any]]:
    from vllm.v1.attention.backends.byte_v2_ops import byte_v2_custom_ops_are_available
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    if not byte_v2_custom_ops_are_available():
        raise RuntimeError("ByteV2 custom ops are not available")

    tensors = _make_inputs(args, seq_len)
    rows: list[dict[str, Any]] = []
    reference = None

    fa_version = args.fa_version
    if fa_version is None and args.include_flash:
        fa_version = get_flash_attn_version(head_size=args.head_size)

    variants = []
    for compute_block_n in args.compute_block_ns:
        if args.include_single and not args.gqa_fa2_like:
            suffix = ""
            if args.gqa_packed:
                suffix = "_gqa4_nooutlier"
                if args.gqa_fa2_like:
                    suffix += "_fa2_like"
                    if args.gqa_fa2_qk_mma:
                        suffix += "_qk_mma"
                        if args.gqa_fa2_multiwarp:
                            suffix += "_multiwarp"
                        elif args.gqa_fa2_mainloop:
                            suffix += "_mainloop"
                        elif args.gqa_fa2_direct:
                            suffix += "_direct"
            elif args.assume_no_outlier:
                suffix = "_nooutlier"
            if args.high_byte_payload:
                suffix += "_highbyte"
            elif args.sideband_high_payload:
                suffix += "_sideband_high"
            if args.guarded_split:
                suffix += "_guarded"
            variants.append(
                (
                    f"byte_v2_single_bn{compute_block_n}{suffix}",
                    None,
                    compute_block_n,
                    "current",
                    *_byte_v2_single_fn(
                        args,
                        tensors,
                        seq_len=seq_len,
                        compute_block_n=compute_block_n,
                    ),
                )
            )
        for partition_size in args.partition_sizes:
            if args.gqa_fa2_like and (
                compute_block_n != 64
                or (not args.gqa_fa2_direct and partition_size != compute_block_n)
            ):
                continue
            suffix = ""
            if args.gqa_packed:
                suffix = "_gqa4_nooutlier"
                if args.gqa_fa2_like:
                    suffix += "_fa2_like"
                    if args.gqa_fa2_qk_mma:
                        suffix += "_qk_mma"
                        if args.gqa_fa2_multiwarp:
                            suffix += "_multiwarp"
                        elif args.gqa_fa2_mainloop:
                            suffix += "_mainloop"
                        elif args.gqa_fa2_direct:
                            suffix += "_direct"
            elif args.assume_no_outlier:
                suffix = "_nooutlier"
            if args.high_byte_payload:
                suffix += "_highbyte"
            elif args.sideband_high_payload:
                suffix += "_sideband_high"
            diagnostic_modes = (
                args.direct_diagnostic_modes if args.gqa_fa2_direct else ["current"]
            )
            for diagnostic_mode in diagnostic_modes:
                diagnostic_suffix = (
                    "" if diagnostic_mode == "current" else f"_diag_{diagnostic_mode}"
                )
                variants.append(
                    (
                        f"byte_v2_split_p{partition_size}_bn{compute_block_n}"
                        f"{suffix}{diagnostic_suffix}",
                        partition_size,
                        compute_block_n,
                        diagnostic_mode,
                        *_byte_v2_split_fn(
                            args,
                            tensors,
                            seq_len=seq_len,
                            partition_size=partition_size,
                            compute_block_n=compute_block_n,
                            direct_diagnostic_mode=diagnostic_mode,
                        ),
                    )
                )
    if args.include_flash:
        if fa_version is None:
            raise RuntimeError("could not determine FlashAttention version")
        variants.append(
            (
                f"vllm_flash_paged_fa{fa_version}",
                None,
                None,
                "current",
                *_flash_paged_fn(args, tensors, seq_len=seq_len, fa_version=fa_version),
            )
        )

    for (
        name,
        partition_size,
        compute_block_n,
        diagnostic_mode,
        run,
        output,
        workspace_mib,
    ) in variants:
        try:
            stats, result = _time_cuda_kernel(
                run,
                warmup=args.warmup,
                iters=args.iters,
                cuda_profiler_range=args.cuda_profiler_range,
            )
            result = result[0] if isinstance(result, tuple) else result
            if result is not output:
                output.copy_(result)
            if reference is None:
                reference = output.detach().clone()
                max_abs_diff = 0.0
            else:
                max_abs_diff = _max_abs_diff(output, reference)
            row = {
                "name": name,
                "seq_len": seq_len,
                "num_seqs": args.num_seqs,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_size": args.head_size,
                "block_size": args.block_size,
                "kv_blocks_per_seq": tensors["blocks_per_seq"],
                "partition_size": partition_size,
                "num_partitions": _ceil_div(seq_len, partition_size)
                if partition_size is not None
                else None,
                "compute_block_n": compute_block_n,
                "no_outlier_inputs": (args.no_outlier_inputs or args.assume_no_outlier)
                and not args.force_outlier_inputs,
                "force_outlier_inputs": args.force_outlier_inputs,
                "assume_no_outlier": args.assume_no_outlier,
                "gqa_packed": args.gqa_packed,
                "gqa_fa2_like": args.gqa_fa2_like,
                "gqa_fa2_qk_mma": args.gqa_fa2_qk_mma,
                "gqa_fa2_mainloop": args.gqa_fa2_mainloop,
                "gqa_fa2_multiwarp": args.gqa_fa2_multiwarp,
                "gqa_fa2_direct": args.gqa_fa2_direct,
                "high_byte_payload": args.high_byte_payload,
                "sideband_high_payload": args.sideband_high_payload,
                "direct_diagnostic_mode": diagnostic_mode,
                "guarded_split": args.guarded_split,
                "workspace_mib": workspace_mib,
                "warmup": args.warmup,
                "iters": args.iters,
                "max_abs_diff_vs_first": max_abs_diff,
                **stats,
            }
            phase_profile_tensor = getattr(run, "phase_profile_tensor", None)
            if phase_profile_tensor is not None:
                phase_values = (
                    phase_profile_tensor.detach()
                    .flatten()[: len(PHASE_PROFILE_KEYS)]
                    .cpu()
                    .tolist()
                )
                row["phase_profile_cycles"] = dict(
                    zip(PHASE_PROFILE_KEYS, phase_values, strict=False)
                )
        except Exception as exc:
            row = {
                "name": name,
                "seq_len": seq_len,
                "num_seqs": args.num_seqs,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_size": args.head_size,
                "block_size": args.block_size,
                "partition_size": partition_size,
                "compute_block_n": compute_block_n,
                "no_outlier_inputs": (args.no_outlier_inputs or args.assume_no_outlier)
                and not args.force_outlier_inputs,
                "force_outlier_inputs": args.force_outlier_inputs,
                "assume_no_outlier": args.assume_no_outlier,
                "gqa_packed": args.gqa_packed,
                "gqa_fa2_like": args.gqa_fa2_like,
                "gqa_fa2_qk_mma": args.gqa_fa2_qk_mma,
                "gqa_fa2_mainloop": args.gqa_fa2_mainloop,
                "gqa_fa2_multiwarp": args.gqa_fa2_multiwarp,
                "gqa_fa2_direct": args.gqa_fa2_direct,
                "high_byte_payload": args.high_byte_payload,
                "sideband_high_payload": args.sideband_high_payload,
                "direct_diagnostic_mode": diagnostic_mode,
                "guarded_split": args.guarded_split,
                "warmup": args.warmup,
                "iters": args.iters,
                "error": repr(exc),
            }
        rows.append(row)
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "seq_len,name,median_ms,mean_ms,min_ms,p90_ms,partition_size,"
        "num_partitions,compute_block_n,no_outlier_inputs,"
        "force_outlier_inputs,assume_no_outlier,gqa_packed,gqa_fa2_like,"
        "gqa_fa2_qk_mma,gqa_fa2_mainloop,gqa_fa2_multiwarp,gqa_fa2_direct,"
        "high_byte_payload,sideband_high_payload,direct_diagnostic_mode,"
        "guarded_split,workspace_mib,"
        "max_abs_diff_vs_first,error"
    )
    for row in rows:
        print(
            f"{row['seq_len']},{row['name']},"
            f"{row.get('median_ms', '')},{row.get('mean_ms', '')},"
            f"{row.get('min_ms', '')},{row.get('p90_ms', '')},"
            f"{row.get('partition_size', '')},{row.get('num_partitions', '')},"
            f"{row.get('compute_block_n', '')},"
            f"{row.get('no_outlier_inputs', '')},"
            f"{row.get('force_outlier_inputs', '')},"
            f"{row.get('assume_no_outlier', '')},"
            f"{row.get('gqa_packed', '')},"
            f"{row.get('gqa_fa2_like', '')},"
            f"{row.get('gqa_fa2_qk_mma', '')},"
            f"{row.get('gqa_fa2_mainloop', '')},"
            f"{row.get('gqa_fa2_multiwarp', '')},"
            f"{row.get('gqa_fa2_direct', '')},"
            f"{row.get('high_byte_payload', '')},"
            f"{row.get('sideband_high_payload', '')},"
            f"{row.get('direct_diagnostic_mode', '')},"
            f"{row.get('guarded_split', '')},"
            f"{row.get('workspace_mib', '')},"
            f"{row.get('max_abs_diff_vs_first', '')},{row.get('error', '')}"
        )


def main() -> None:
    _prepend_venv_bin_to_path()

    import torch

    args = parse_args()
    if args.gqa_packed and not args.assume_no_outlier:
        raise ValueError("--gqa-packed requires --assume-no-outlier")
    if args.force_outlier_inputs and args.no_outlier_inputs:
        raise ValueError("--force-outlier-inputs conflicts with --no-outlier-inputs")
    if args.gqa_fa2_like and not args.gqa_packed:
        raise ValueError("--gqa-fa2-like requires --gqa-packed")
    if args.gqa_fa2_qk_mma and not args.gqa_fa2_like:
        raise ValueError("--gqa-fa2-qk-mma requires --gqa-fa2-like")
    if args.gqa_fa2_mainloop and not args.gqa_fa2_qk_mma:
        raise ValueError("--gqa-fa2-mainloop requires --gqa-fa2-qk-mma")
    if args.gqa_fa2_multiwarp and not args.gqa_fa2_qk_mma:
        raise ValueError("--gqa-fa2-multiwarp requires --gqa-fa2-qk-mma")
    if args.gqa_fa2_direct and not args.gqa_fa2_qk_mma:
        raise ValueError("--gqa-fa2-direct requires --gqa-fa2-qk-mma")
    payload_mode_count = sum(
        (
            args.high_byte_payload,
            args.sideband_high_payload,
        )
    )
    if payload_mode_count > 1:
        raise ValueError(
            "only one of --high-byte-payload and --sideband-high-payload may be set"
        )
    if args.high_byte_payload and not (args.assume_no_outlier and args.gqa_fa2_direct):
        raise ValueError(
            "--high-byte-payload requires --assume-no-outlier and --gqa-fa2-direct"
        )
    if args.high_byte_payload and args.include_single:
        raise ValueError("--high-byte-payload currently supports split-k only")
    if args.high_byte_payload and args.direct_diagnostic_modes != ["current"]:
        raise ValueError(
            "--high-byte-payload currently supports only the current direct "
            "diagnostic mode"
        )
    if args.sideband_high_payload and not (
        args.assume_no_outlier and args.gqa_fa2_direct and args.guarded_split
    ):
        raise ValueError(
            "--sideband-high-payload requires --assume-no-outlier, "
            "--gqa-fa2-direct, and --guarded-split"
        )
    if args.sideband_high_payload and args.include_single:
        raise ValueError("--sideband-high-payload currently supports split-k only")
    if args.sideband_high_payload and args.direct_diagnostic_modes != ["current"]:
        raise ValueError(
            "--sideband-high-payload currently supports only the current "
            "direct diagnostic mode"
        )
    if args.direct_diagnostic_modes != ["current"] and not args.gqa_fa2_direct:
        raise ValueError("--direct-diagnostic-modes requires --gqa-fa2-direct")
    q_per_kv = args.num_heads // args.num_kv_heads
    effective_m16_enabled = any(
        mode == "effective-m16-profile" for mode in args.direct_diagnostic_modes
    )
    other_diagnostic_enabled = any(
        mode not in {"current", "effective-m16-profile"}
        for mode in args.direct_diagnostic_modes
    )
    if effective_m16_enabled and q_per_kv != 16:
        raise ValueError("effective-m16-profile requires q_per_kv=16")
    if other_diagnostic_enabled and q_per_kv != 4:
        raise ValueError(
            "non-current --direct-diagnostic-modes currently require q_per_kv=4"
        )
    enabled_fa2_subpaths = sum(
        int(flag)
        for flag in (
            args.gqa_fa2_mainloop,
            args.gqa_fa2_multiwarp,
            args.gqa_fa2_direct,
        )
    )
    if enabled_fa2_subpaths > 1:
        raise ValueError(
            "--gqa-fa2-mainloop, --gqa-fa2-multiwarp, and "
            "--gqa-fa2-direct are mutually exclusive"
        )
    if args.guarded_split and not args.assume_no_outlier:
        raise ValueError("--guarded-split requires --assume-no-outlier")
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA-compatible accelerator is required")

    rows: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        print(f"decode_microbench.start seq_len={seq_len}", flush=True)
        rows.extend(_benchmark_one(args, seq_len))
        print(f"decode_microbench.done seq_len={seq_len}", flush=True)

    _write_jsonl(args.output_jsonl, rows)
    print(f"decode_microbench.output_jsonl={args.output_jsonl}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
