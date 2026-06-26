# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local ByteV2 E2E smoke test.

This script is intentionally not a CI test: it loads a real model and requires
CUDA. It is useful after rebuilding ByteV2 native kernels to verify engine
initialization, warmup, prefill, and decode.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path


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
        help="Model path or name.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
        help="Prompt for the multi-token prefill smoke.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--compare-default",
        action="store_true",
        help=(
            "Also run the default attention backend and compare prefill-only "
            "outputs. Single-token and multi-token decode are printed but not "
            "compared by default because they use the ByteV2 compressed cache."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("byte_v2", "default"),
        default="byte_v2",
        help="Backend to run when not comparing in this process.",
    )
    parser.add_argument(
        "--compare-decode",
        action="store_true",
        help=(
            "Also compare decode-path outputs. Mismatches fail unless "
            "--allow-mismatch is set."
        ),
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print decode mismatches without failing when --compare-decode is set.",
    )
    parser.add_argument(
        "--print-diff",
        action="store_true",
        help="Print token-level diff summaries for compared default-backend outputs.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print coarse per-generate wall-clock timings.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
        help="Allow vLLM compile/cudagraph paths instead of forcing eager mode.",
    )
    parser.set_defaults(enforce_eager=True)
    parser.add_argument(
        "--skip-reading-prefix-cache",
        action="store_true",
        help="Set SamplingParams.skip_reading_prefix_cache for smoke requests.",
    )
    return parser.parse_args()


def _build_llm(args: argparse.Namespace, attention_backend: str | None):
    from vllm import LLM

    llm_kwargs = {}
    if attention_backend is not None:
        llm_kwargs["attention_config"] = {"backend": attention_backend}
    return LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        block_size=16,
        **llm_kwargs,
    )


def _run_smoke(
    args: argparse.Namespace,
    *,
    attention_backend: str | None,
) -> dict[str, tuple[list[int], str]]:
    import torch

    from vllm import SamplingParams, TokensPrompt

    def sampling_params(max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            skip_reading_prefix_cache=args.skip_reading_prefix_cache,
        )

    llm = _build_llm(args, attention_backend)
    try:
        profile_seconds = {}
        outputs_by_label = {}
        generate_specs = (
            (
                "single_token",
                [TokensPrompt(prompt_token_ids=[128000])],
                1,
            ),
            (
                "prefill_only",
                [args.prompt],
                1,
            ),
            (
                "prefill_decode",
                [args.prompt],
                args.max_tokens,
            ),
        )
        for label, prompts, max_tokens in generate_specs:
            start = time.perf_counter()
            outputs_by_label[label] = llm.generate(
                prompts,
                sampling_params(max_tokens=max_tokens),
                use_tqdm=False,
            )
            profile_seconds[label] = time.perf_counter() - start
        results = {}
        label_prefix = attention_backend or "default"
        for label, outputs in outputs_by_label.items():
            out = outputs[0]
            completion = out.outputs[0]
            token_ids = list(completion.token_ids)
            results[label] = (token_ids, completion.text)
            if args.profile:
                print(
                    f"profile.{label_prefix}.{label}.seconds="
                    f"{profile_seconds[label]:.6f}"
                )
            print(f"{label_prefix}.{label}.prompt_token_ids={out.prompt_token_ids}")
            print(f"{label_prefix}.{label}.output_token_ids={token_ids}")
            print(f"{label_prefix}.{label}.text={completion.text!r}")
        return results
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()


def _token_diff_summary(
    byte_v2_token_ids: list[int],
    default_token_ids: list[int],
) -> str:
    first_mismatch = None
    for idx, (byte_v2_token_id, default_token_id) in enumerate(
        zip(byte_v2_token_ids, default_token_ids)
    ):
        if byte_v2_token_id != default_token_id:
            first_mismatch = idx
            break
    if first_mismatch is None and len(byte_v2_token_ids) != len(default_token_ids):
        first_mismatch = min(len(byte_v2_token_ids), len(default_token_ids))
    if first_mismatch is None:
        return "match=True"
    return (
        "match=False "
        f"first_mismatch={first_mismatch} "
        f"BYTE_V2_tail={byte_v2_token_ids[first_mismatch:]} "
        f"default_tail={default_token_ids[first_mismatch:]}"
    )


def _compare_label(
    label: str,
    byte_v2_results: dict[str, tuple[list[int], str]],
    default_results: dict[str, tuple[list[int], str]],
    *,
    allow_mismatch: bool,
    print_diff: bool,
) -> bool:
    byte_v2_token_ids = byte_v2_results[label][0]
    default_token_ids = default_results[label][0]
    summary = _token_diff_summary(byte_v2_token_ids, default_token_ids)
    is_match = byte_v2_token_ids == default_token_ids
    if print_diff or not is_match:
        print(f"diff.{label}.{summary}")
    if not is_match and not allow_mismatch:
        raise AssertionError(
            f"{label} output mismatch: BYTE_V2={byte_v2_token_ids}, "
            f"default={default_token_ids}"
        )
    return is_match


def main() -> None:
    _prepend_venv_bin_to_path()

    args = parse_args()
    if args.backend == "default":
        if args.compare_default or args.compare_decode:
            raise ValueError("--backend=default cannot be combined with compare flags")
        _run_smoke(args, attention_backend=None)
        return

    byte_v2_results = _run_smoke(args, attention_backend="BYTE_V2")

    if not args.compare_default and not args.compare_decode:
        return

    default_results = _run_smoke(args, attention_backend=None)
    labels = []
    _compare_label(
        "prefill_only",
        byte_v2_results,
        default_results,
        allow_mismatch=False,
        print_diff=args.print_diff,
    )
    labels.append("prefill_only")

    if args.compare_decode:
        for label in ("single_token", "prefill_decode"):
            _compare_label(
                label,
                byte_v2_results,
                default_results,
                allow_mismatch=args.allow_mismatch,
                print_diff=True,
            )
            labels.append(label)
    print(f"compare_default.labels={labels}")


if __name__ == "__main__":
    main()
