# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E decode-length benchmark for Byte-v2 KV cache experiments.

The parent process launches one child process per mode so CUDA/vLLM engine
state is torn down between raw, raw-overlay Byte-v2, and compressed-only
Byte-v2 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "BYTE_V2_BENCH_RESULT_JSON:"


def _collect_byte_v2_sparse_fallback_stats(llm) -> dict[str, Any]:
    try:
        worker_stats = llm.collective_rpc("get_byte_v2_sparse_fallback_stats")
    except Exception as exc:  # pragma: no cover - benchmark diagnostics.
        return {"error": repr(exc)}

    enabled_workers = [
        stats for stats in worker_stats if stats.get("enabled_layers", 0) > 0
    ]
    return {
        "num_workers": len(worker_stats),
        "workers_with_pool": len(enabled_workers),
        "total_capacity": sum(int(stats["total_capacity"]) for stats in worker_stats),
        "total_next_slot": sum(
            int(stats["total_next_slot"]) for stats in worker_stats
        ),
        "total_assigned_blocks": sum(
            int(stats["total_assigned_blocks"]) for stats in worker_stats
        ),
        "max_next_slot": max(
            (int(stats["max_next_slot"]) for stats in worker_stats),
            default=0,
        ),
        "max_capacity": max(
            (int(stats["max_capacity"]) for stats in worker_stats),
            default=0,
        ),
        "any_exhausted": any(
            bool(stats["any_exhausted"]) for stats in worker_stats
        ),
        "workers": worker_stats,
    }


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _mode_env(mode: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if mode == "raw" or mode == "byte_v2_overlay":
        env["VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE"] = "0"
    elif mode == "byte_v2_compressed_only":
        env["VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE"] = "1"
        if "VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL" not in os.environ:
            env["VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL"] = "1"
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    return env


def _kv_cache_dtype(mode: str) -> str:
    return "auto" if mode == "raw" else "byte_v2"


def _make_prompt_token_ids(model: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        add_special_tokens=False,
    )
    if not base:
        base = [0]
    repeats = (prompt_len + len(base) - 1) // len(base)
    return (base * repeats)[:prompt_len]


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    mode = args.child_mode
    os.environ.update(_mode_env(mode))
    decode_lens = _parse_csv_ints(args.decode_lens)
    max_decode_len = max(decode_lens)
    max_model_len = args.max_model_len or args.prompt_len + max_decode_len + 16
    max_num_batched_tokens = args.max_num_batched_tokens or (
        args.batch_size * (args.prompt_len + max_decode_len)
    )

    prompt_ids = _make_prompt_token_ids(args.model, args.prompt_len)
    prompts = [
        TokensPrompt(prompt_token_ids=list(prompt_ids))
        for _ in range(args.batch_size)
    ]

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        kv_cache_dtype=_kv_cache_dtype(mode),
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        disable_log_stats=True,
    )

    warmup_sampling = SamplingParams(
        max_tokens=args.warmup_decode_len,
        min_tokens=args.warmup_decode_len,
        ignore_eos=True,
        temperature=0.0,
        detokenize=False,
    )
    llm.generate(prompts, warmup_sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sparse_fallback_stats_after_warmup = _collect_byte_v2_sparse_fallback_stats(
        llm
    )

    results: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("byte_v2_bench_measured")
    try:
        for decode_len in decode_lens:
            run_results: list[dict[str, float | int]] = []
            sampling = SamplingParams(
                max_tokens=decode_len,
                min_tokens=decode_len,
                ignore_eos=True,
                temperature=0.0,
                detokenize=False,
            )
            for run_idx in range(args.num_runs):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(
                        f"byte_v2_bench_decode_len_{decode_len}_run_{run_idx}"
                    )
                start = time.perf_counter()
                try:
                    outputs = llm.generate(prompts, sampling, use_tqdm=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                elapsed = time.perf_counter() - start
                output_tokens = sum(
                    len(request_output.outputs[0].token_ids)
                    for request_output in outputs
                )
                cached_tokens = sum(
                    int(request_output.num_cached_tokens or 0)
                    for request_output in outputs
                )
                prompt_tokens = args.batch_size * args.prompt_len
                run_results.append(
                    {
                        "run_idx": run_idx,
                        "elapsed_s": elapsed,
                        "prompt_tokens": prompt_tokens,
                        "cached_tokens": cached_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                        "output_tokens_per_s": output_tokens / elapsed,
                        "total_tokens_per_s": (
                            (prompt_tokens + output_tokens) / elapsed
                        ),
                    }
                )
            sparse_fallback_stats = _collect_byte_v2_sparse_fallback_stats(llm)

            output_tps_values = [
                float(run["output_tokens_per_s"]) for run in run_results
            ]
            total_tps_values = [
                float(run["total_tokens_per_s"]) for run in run_results
            ]
            elapsed_values = [float(run["elapsed_s"]) for run in run_results]
            results.append(
                {
                    "decode_len": decode_len,
                    "runs": run_results,
                    "median_elapsed_s": statistics.median(elapsed_values),
                    "mean_elapsed_s": statistics.fmean(elapsed_values),
                    "median_output_tokens_per_s": statistics.median(
                        output_tps_values
                    ),
                    "mean_output_tokens_per_s": statistics.fmean(
                        output_tps_values
                    ),
                    "median_total_tokens_per_s": statistics.median(
                        total_tps_values
                    ),
                    "mean_total_tokens_per_s": statistics.fmean(
                        total_tps_values
                    ),
                    "sparse_fallback_stats": sparse_fallback_stats,
                }
            )
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    result = {
        "mode": mode,
        "model": args.model,
        "dtype": args.dtype,
        "kv_cache_dtype": _kv_cache_dtype(mode),
        "prompt_len": args.prompt_len,
        "batch_size": args.batch_size,
        "decode_lens": decode_lens,
        "num_runs": args.num_runs,
        "warmup_decode_len": args.warmup_decode_len,
        "block_size": args.block_size,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "sparse_fallback_stats_after_warmup": (
            sparse_fallback_stats_after_warmup
        ),
        "final_sparse_fallback_stats": (
            _collect_byte_v2_sparse_fallback_stats(llm)
        ),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE",
                "VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL",
                "VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO",
                "VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS",
                "VLLM_BYTE_V2_USE_NATIVE_KERNELS",
                "VLLM_BYTE_V2_DECODE_PAGE_FASTPATH",
                "VLLM_BYTE_V2_DECODE_TILE_FASTPATH",
                "VLLM_BYTE_V2_DECODE_CUTE_STAGE1",
                "VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO",
                "VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE",
                "VLLM_BYTE_V2_DECODE_FAST_STAGE1",
                "VLLM_BYTE_V2_DECODE_FLASH_STAGE1",
                "VLLM_BYTE_V2_DECODE_V4_STAGE1",
                "VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES",
                "VLLM_BYTE_V2_DECODE_SPLIT_K",
                "VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE",
                "VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE",
                "VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE",
                "VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA",
                "VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK",
                "VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES",
                "VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE",
                "VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC",
                "VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC",
                "VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK",
                "VLLM_BYTE_V2_PAYLOAD_LAYOUT",
            )
            if os.environ.get(key) is not None
        },
        "results": results,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    return result


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    modes = _parse_csv_strings(args.modes)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for mode in modes:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-mode",
            mode,
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--prompt-len",
            str(args.prompt_len),
            "--decode-lens",
            args.decode_lens,
            "--batch-size",
            str(args.batch_size),
            "--num-runs",
            str(args.num_runs),
            "--warmup-decode-len",
            str(args.warmup_decode_len),
            "--block-size",
            str(args.block_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.disable_prefix_caching:
            cmd.append("--disable-prefix-caching")
        if args.max_model_len is not None:
            cmd.extend(["--max-model-len", str(args.max_model_len)])
        if args.max_num_batched_tokens is not None:
            cmd.extend(
                ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
            )
        if args.enforce_eager:
            cmd.append("--enforce-eager")

        env = os.environ.copy()
        env.update(_mode_env(mode))
        print(f"\n=== Running mode={mode} ===", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        child_result = None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith(RESULT_PREFIX):
                child_result = json.loads(line[len(RESULT_PREFIX) :])
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"mode={mode} failed with exit code {return_code}")
        if child_result is None:
            raise RuntimeError(f"mode={mode} did not emit benchmark JSON")
        all_results.append(child_result)

        payload = {
            "model": args.model,
            "modes": modes,
            "prompt_len": args.prompt_len,
            "batch_size": args.batch_size,
            "decode_lens": _parse_csv_ints(args.decode_lens),
            "num_runs": args.num_runs,
            "results": all_results,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--modes",
        default="raw,byte_v2_overlay,byte_v2_compressed_only",
    )
    parser.add_argument("--child-mode", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-lens", default="16,64,128,256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--warmup-decode-len", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--output-json",
        default="benchmarks/byte_v2_decode_e2e_results.json",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.child_mode is not None:
        run_child(args)
    else:
        result = run_parent(args)
        print("\n=== Summary ===")
        for mode_result in result["results"]:
            print(f"mode={mode_result['mode']}")
            for row in mode_result["results"]:
                print(
                    "  decode_len={decode_len:<4} "
                    "median_output_tps={median_output_tokens_per_s:.2f} "
                    "median_elapsed_s={median_elapsed_s:.3f}".format(**row)
                )
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
