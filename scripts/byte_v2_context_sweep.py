# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a local ByteV2/default context-length sweep.

The parent process launches one worker process per (backend, context length).
This avoids CUDA graph and torch.compile state from one backend affecting the
next backend in the same Python process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "BYTE_V2_CONTEXT_SWEEP_RESULT "


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
        default=[256, 512, 1024, 2048, 4096],
        help="Prompt token lengths to test.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--backend",
        choices=("all", "byte_v2", "default"),
        default="all",
        help="Backend(s) to run.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager mode instead of vLLM compile/cudagraph mode.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_context_sweep.jsonl",
        help="Path for machine-readable worker results.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _make_prompt_token_ids(model: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    seed_text = (
        "Deterministic decoding gives regression tests a stable signal. "
        "Long context attention should preserve the same factual details and "
        "avoid drifting as the prompt grows. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError("tokenizer produced no seed tokens and has no BOS token")
        seed_ids = [int(bos_token_id)]
    repeats = (prompt_len + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:prompt_len]


def _build_llm(args: argparse.Namespace, backend: str, max_model_len: int):
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {}
    if backend == "byte_v2":
        llm_kwargs["attention_config"] = {"backend": "BYTE_V2"}
    return LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        block_size=16,
        **llm_kwargs,
    )


def _timed_generate(llm, prompt, sampling_params) -> tuple[float, list[int], str]:
    import torch

    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    torch.accelerator.synchronize()
    seconds = time.perf_counter() - start
    completion = outputs[0].outputs[0]
    return seconds, list(completion.token_ids), completion.text


def _run_worker(args: argparse.Namespace) -> None:
    import gc

    import torch

    from vllm import SamplingParams, TokensPrompt

    assert args.context_len is not None
    assert args.backend in ("byte_v2", "default")

    context_len = int(args.context_len)
    max_model_len = context_len + args.max_tokens
    prompt_token_ids = _make_prompt_token_ids(args.model, context_len)
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    init_start = time.perf_counter()
    llm = _build_llm(args, args.backend, max_model_len)
    torch.accelerator.synchronize()
    init_seconds = time.perf_counter() - init_start

    try:
        first_seconds, first_token_ids, first_text = _timed_generate(
            llm, prompt, sampling_params
        )
        cached_seconds, cached_token_ids, cached_text = _timed_generate(
            llm, prompt, sampling_params
        )
        result = {
            "backend": args.backend,
            "context_len": context_len,
            "prompt_tokens": len(prompt_token_ids),
            "max_model_len": max_model_len,
            "max_tokens": args.max_tokens,
            "enforce_eager": args.enforce_eager,
            "init_seconds": init_seconds,
            "first_seconds": first_seconds,
            "cached_seconds": cached_seconds,
            "first_token_ids": first_token_ids,
            "cached_token_ids": cached_token_ids,
            "first_text": first_text,
            "cached_text": cached_text,
        }
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()


def _worker_backends(args: argparse.Namespace) -> list[str]:
    if args.backend == "all":
        return ["byte_v2", "default"]
    return [args.backend]


def _parse_worker_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError("worker did not print a context sweep result")


def _run_child(
    args: argparse.Namespace, backend: str, context_len: int
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if backend == "byte_v2":
        env.setdefault("BYTE_V2_DECODE_RAW_FALLBACK", "0")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--model",
        args.model,
        "--context-len",
        str(context_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    label = f"backend={backend} context_len={context_len}"
    print(f"sweep.start {label}", flush=True)
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
        raise RuntimeError(f"sweep worker failed for {label}")
    result = _parse_worker_result(proc.stdout)
    print(
        "sweep.done "
        f"{label} first={result['first_seconds']:.6f}s "
        f"cached={result['cached_seconds']:.6f}s "
        f"init={result['init_seconds']:.3f}s",
        flush=True,
    )
    return result


def _print_summary(results: list[dict[str, Any]]) -> None:
    by_key = {(r["context_len"], r["backend"]): r for r in results}
    print(
        "context_len,byte_v2_first_s,default_first_s,first_speedup,"
        "first_match,byte_v2_cached_s,default_cached_s,cached_speedup,"
        "cached_match"
    )
    for context_len in sorted({r["context_len"] for r in results}):
        byte_v2 = by_key.get((context_len, "byte_v2"))
        default = by_key.get((context_len, "default"))
        if byte_v2 is None or default is None:
            row = [str(context_len)]
            for backend in ("byte_v2", "default"):
                result = by_key.get((context_len, backend))
                if result is None:
                    row.extend(["", ""])
                else:
                    row.extend(
                        [
                            f"{result['first_seconds']:.6f}",
                            f"{result['cached_seconds']:.6f}",
                        ]
                    )
            print(",".join(row))
            continue

        first_speedup = default["first_seconds"] / byte_v2["first_seconds"]
        cached_speedup = default["cached_seconds"] / byte_v2["cached_seconds"]
        first_match = byte_v2["first_token_ids"] == default["first_token_ids"]
        cached_match = byte_v2["cached_token_ids"] == default["cached_token_ids"]
        print(
            f"{context_len},{byte_v2['first_seconds']:.6f},"
            f"{default['first_seconds']:.6f},{first_speedup:.4f},"
            f"{first_match},{byte_v2['cached_seconds']:.6f},"
            f"{default['cached_seconds']:.6f},{cached_speedup:.4f},"
            f"{cached_match}"
        )


def main() -> None:
    _prepend_venv_bin_to_path()
    args = parse_args()
    if args.worker:
        _run_worker(args)
        return

    results = []
    for context_len in args.context_lens:
        for backend in _worker_backends(args):
            results.append(_run_child(args, backend, context_len))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"sweep.output_jsonl={output_path}")
    _print_summary(results)


if __name__ == "__main__":
    main()
