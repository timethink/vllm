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
        "--guarded-split",
        action="store_true",
        help=("Use the split-k no-outlier fast path guarded by per-page unsafe flags."),
    )
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_decode_microbench.jsonl",
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
) -> tuple[dict[str, float], Any]:
    import torch

    result = None
    for _ in range(warmup):
        result = fn()
    torch.accelerator.synchronize()

    events = []
    for _ in range(iters):
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        events.append((start_event, end_event))
    torch.accelerator.synchronize()

    return _stats_ms([start.elapsed_time(end) for start, end in events]), result


def _max_abs_diff(lhs: Any, rhs: Any) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _workspace_mib(*tensors: Any) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / 1024 / 1024


def _make_inputs(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    import torch

    from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayoutV4
    from vllm.v1.attention.backends.byte_v2_ops import (
        byte_v2_reshape_and_cache,
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
    use_no_outlier_inputs = args.no_outlier_inputs or args.assume_no_outlier
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

    layout = ByteV2PageLayoutV4(num_kv_heads=args.num_kv_heads)
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device=device,
    )
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

    return {
        "query": query,
        "kv_cache": kv_cache,
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
        if args.guarded_split:
            byte_v2_paged_decode_attention_split_k_guarded(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                tensors["query"],
                tensors["kv_cache"],
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
                tensors["kv_cache"],
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
            if args.guarded_split:
                suffix += "_guarded"
            variants.append(
                (
                    f"byte_v2_single_bn{compute_block_n}{suffix}",
                    None,
                    compute_block_n,
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
            variants.append(
                (
                    f"byte_v2_split_p{partition_size}_bn{compute_block_n}{suffix}",
                    partition_size,
                    compute_block_n,
                    *_byte_v2_split_fn(
                        args,
                        tensors,
                        seq_len=seq_len,
                        partition_size=partition_size,
                        compute_block_n=compute_block_n,
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
                *_flash_paged_fn(args, tensors, seq_len=seq_len, fa_version=fa_version),
            )
        )

    for name, partition_size, compute_block_n, run, output, workspace_mib in variants:
        try:
            stats, result = _time_cuda_kernel(
                run,
                warmup=args.warmup,
                iters=args.iters,
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
                "no_outlier_inputs": args.no_outlier_inputs or args.assume_no_outlier,
                "assume_no_outlier": args.assume_no_outlier,
                "gqa_packed": args.gqa_packed,
                "gqa_fa2_like": args.gqa_fa2_like,
                "gqa_fa2_qk_mma": args.gqa_fa2_qk_mma,
                "gqa_fa2_mainloop": args.gqa_fa2_mainloop,
                "gqa_fa2_multiwarp": args.gqa_fa2_multiwarp,
                "gqa_fa2_direct": args.gqa_fa2_direct,
                "guarded_split": args.guarded_split,
                "workspace_mib": workspace_mib,
                "warmup": args.warmup,
                "iters": args.iters,
                "max_abs_diff_vs_first": max_abs_diff,
                **stats,
            }
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
                "no_outlier_inputs": args.no_outlier_inputs or args.assume_no_outlier,
                "assume_no_outlier": args.assume_no_outlier,
                "gqa_packed": args.gqa_packed,
                "gqa_fa2_like": args.gqa_fa2_like,
                "gqa_fa2_qk_mma": args.gqa_fa2_qk_mma,
                "gqa_fa2_mainloop": args.gqa_fa2_mainloop,
                "gqa_fa2_multiwarp": args.gqa_fa2_multiwarp,
                "gqa_fa2_direct": args.gqa_fa2_direct,
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
        "assume_no_outlier,gqa_packed,gqa_fa2_like,gqa_fa2_qk_mma,"
        "gqa_fa2_mainloop,gqa_fa2_multiwarp,gqa_fa2_direct,guarded_split,"
        "workspace_mib,"
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
            f"{row.get('assume_no_outlier', '')},"
            f"{row.get('gqa_packed', '')},"
            f"{row.get('gqa_fa2_like', '')},"
            f"{row.get('gqa_fa2_qk_mma', '')},"
            f"{row.get('gqa_fa2_mainloop', '')},"
            f"{row.get('gqa_fa2_multiwarp', '')},"
            f"{row.get('gqa_fa2_direct', '')},"
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
