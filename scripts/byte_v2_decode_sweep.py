# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep ByteV2 cached decode parameters using the E2E profiler."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

RESULT_PREFIX = "BYTE_V2_E2E_PROFILE_RESULT "


def _prepend_venv_bin_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    venv_bin = repo_root / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "BYTE_V2_SMOKE_MODEL",
            "/mnt/sdb/yxz/ByteV2/Meta-Llama-3.1-8B-Instruct",
        ),
    )
    parser.add_argument(
        "--context-lens",
        nargs="+",
        type=int,
        default=[4096],
        help="Prompt token lengths to profile.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--partition-sizes",
        nargs="+",
        type=int,
        default=[16],
        help="BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE values.",
    )
    parser.add_argument(
        "--min-seq-lens",
        nargs="+",
        type=int,
        default=[24],
        help="BYTE_V2_DECODE_SPLIT_K_MIN_SEQ_LEN values.",
    )
    parser.add_argument(
        "--compute-block-ns",
        nargs="+",
        type=int,
        default=[64],
        help="BYTE_V2_COMPUTE_BLOCK_N values.",
    )
    parser.add_argument(
        "--include-no-split-k",
        action="store_true",
        help="Also run one no-split-k variant for each context/compute block.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
        help="Allow vLLM compile/cudagraph paths in the child profiler.",
    )
    parser.set_defaults(enforce_eager=True)
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_decode_sweep.jsonl",
        help="Path for machine-readable sweep rows.",
    )
    return parser.parse_args()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _parse_profile_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError("worker did not print a profile result")


def _op_by_name(profile: dict[str, Any], op_name: str) -> dict[str, Any]:
    for op in profile["ops"]:
        if op["name"] == op_name:
            return op
    return {
        "name": op_name,
        "count": 0,
        "cpu_total_ms": 0.0,
        "cpu_avg_us": 0.0,
        "cuda_total_ms": 0.0,
        "cuda_avg_us": 0.0,
    }


def _sum_ops_by_name(
    profile: dict[str, Any], op_names: tuple[str, ...]
) -> dict[str, Any]:
    ops = [_op_by_name(profile, op_name) for op_name in op_names]
    count = sum(op["count"] for op in ops)
    cpu_total_ms = sum(op["cpu_total_ms"] for op in ops)
    cuda_total_ms = sum(op["cuda_total_ms"] for op in ops)
    return {
        "name": "+".join(op_names),
        "count": count,
        "cpu_total_ms": cpu_total_ms,
        "cpu_avg_us": cpu_total_ms * 1000.0 / count if count else 0.0,
        "cuda_total_ms": cuda_total_ms,
        "cuda_avg_us": cuda_total_ms * 1000.0 / count if count else 0.0,
    }


def _run_child(
    args: argparse.Namespace,
    *,
    context_len: int,
    compute_block_n: int,
    partition_size: int,
    min_seq_len: int,
    split_k: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("BYTE_V2_DECODE_RAW_FALLBACK", "0")
    env.setdefault("BYTE_V2_PREFILL_BACKEND", "sdpa")
    env["BYTE_V2_COMPUTE_BLOCK_N"] = str(compute_block_n)
    env["BYTE_V2_DECODE_SPLIT_K"] = "1" if split_k else "0"
    env["BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE"] = str(partition_size)
    env["BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE"] = str(partition_size)
    env["BYTE_V2_DECODE_SPLIT_K_MIN_SEQ_LEN"] = str(min_seq_len)

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("byte_v2_e2e_profile.py")),
        "--model",
        args.model,
        "--prompt-token-len",
        str(context_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--profile-cached",
    ]
    if not args.enforce_eager:
        cmd.append("--no-enforce-eager")

    label = (
        f"ctx={context_len} compute_block_n={compute_block_n} "
        f"partition={partition_size} min_seq={min_seq_len} split_k={split_k}"
    )
    print(f"decode_sweep.start {label}", flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        return {
            "context_len": context_len,
            "max_tokens": args.max_tokens,
            "compute_block_n": compute_block_n,
            "partition_size": partition_size,
            "min_seq_len": min_seq_len,
            "split_k": split_k,
            "error": f"worker exited with code {proc.returncode}",
        }

    try:
        profile_result = _parse_profile_result(proc.stdout)
    except Exception as exc:
        print(proc.stdout[-6000:])
        return {
            "context_len": context_len,
            "max_tokens": args.max_tokens,
            "compute_block_n": compute_block_n,
            "partition_size": partition_size,
            "min_seq_len": min_seq_len,
            "split_k": split_k,
            "error": repr(exc),
        }

    cached_profile = next(
        profile
        for profile in profile_result["profiles"]
        if profile["label"] == "cached"
    )
    first_profile = next(
        profile for profile in profile_result["profiles"] if profile["label"] == "first"
    )
    split_op = _sum_ops_by_name(
        cached_profile,
        (
            "byte_v2_paged_decode_attention_split_k",
            "byte_v2_paged_decode_attention_split_k_guarded",
        ),
    )
    single_op = _op_by_name(cached_profile, "byte_v2_paged_decode_attention")
    cache_update_op = _op_by_name(
        cached_profile,
        "byte_v2_update_cache_single_token",
    )
    max_decode_seq_len = context_len + args.max_tokens
    partitions = _ceil_div(max_decode_seq_len, partition_size)
    kv_blocks = _ceil_div(max_decode_seq_len, 16)
    row = {
        "context_len": context_len,
        "max_tokens": args.max_tokens,
        "max_decode_seq_len": max_decode_seq_len,
        "kv_blocks": kv_blocks,
        "compute_block_n": compute_block_n,
        "partition_size": partition_size,
        "partitions": partitions,
        "min_seq_len": min_seq_len,
        "split_k": split_k,
        "enforce_eager": args.enforce_eager,
        "first_wall_ms": first_profile["wall_generate_ms"],
        "cached_wall_ms": cached_profile["wall_generate_ms"],
        "cached_custom_cuda_ms": cached_profile["byte_v2_custom_cuda_total_ms"],
        "split_k_count": split_op["count"],
        "split_k_cuda_total_ms": split_op["cuda_total_ms"],
        "split_k_cuda_avg_us": split_op["cuda_avg_us"],
        "single_decode_count": single_op["count"],
        "single_decode_cuda_total_ms": single_op["cuda_total_ms"],
        "single_decode_cuda_avg_us": single_op["cuda_avg_us"],
        "cache_update_count": cache_update_op["count"],
        "cache_update_cuda_total_ms": cache_update_op["cuda_total_ms"],
        "token_ids": profile_result["token_ids"],
    }
    print(
        "decode_sweep.done "
        f"{label} cached={row['cached_wall_ms']:.3f}ms "
        f"split_total={row['split_k_cuda_total_ms']:.3f}ms "
        f"split_avg={row['split_k_cuda_avg_us']:.3f}us "
        f"count={row['split_k_count']}",
        flush=True,
    )
    return row


def _iter_variants(args: argparse.Namespace):
    for context_len, compute_block_n, partition_size, min_seq_len in product(
        args.context_lens,
        args.compute_block_ns,
        args.partition_sizes,
        args.min_seq_lens,
    ):
        if partition_size % 16 != 0:
            print(
                f"decode_sweep.skip partition_size={partition_size} not divisible by 16"
            )
            continue
        yield {
            "context_len": context_len,
            "compute_block_n": compute_block_n,
            "partition_size": partition_size,
            "min_seq_len": min_seq_len,
            "split_k": True,
        }
    if args.include_no_split_k:
        for context_len, compute_block_n in product(
            args.context_lens,
            args.compute_block_ns,
        ):
            yield {
                "context_len": context_len,
                "compute_block_n": compute_block_n,
                "partition_size": 16,
                "min_seq_len": 24,
                "split_k": False,
            }


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _mark_token_matches(rows: list[dict[str, Any]]) -> None:
    baseline_by_context: dict[int, list[list[int]]] = {}
    for row in rows:
        if "error" in row:
            continue
        baseline_by_context.setdefault(row["context_len"], row["token_ids"])
        row["tokens_match_context_baseline"] = (
            row["token_ids"] == baseline_by_context[row["context_len"]]
        )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "context_len,compute_block_n,partition_size,partitions,kv_blocks,"
        "split_k,cached_wall_ms,split_k_count,split_k_cuda_total_ms,"
        "split_k_cuda_avg_us,tokens_match,error"
    )
    for row in rows:
        print(
            f"{row['context_len']},{row['compute_block_n']},"
            f"{row['partition_size']},{row.get('partitions', '')},"
            f"{row.get('kv_blocks', '')},{row['split_k']},"
            f"{row.get('cached_wall_ms', '')},"
            f"{row.get('split_k_count', '')},"
            f"{row.get('split_k_cuda_total_ms', '')},"
            f"{row.get('split_k_cuda_avg_us', '')},"
            f"{row.get('tokens_match_context_baseline', '')},"
            f"{row.get('error', '')}"
        )


def main() -> None:
    _prepend_venv_bin_to_path()
    args = parse_args()
    rows = [_run_child(args, **variant) for variant in _iter_variants(args)]
    _mark_token_matches(rows)
    _write_jsonl(args.output_jsonl, rows)
    print(f"decode_sweep.output_jsonl={args.output_jsonl}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
