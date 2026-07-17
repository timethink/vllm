# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile ByteV2 and raw FA2 speculative decoding on a local GPU.

Each worker owns one vLLM engine and one (backend, speculative-token-count)
configuration. The parent process uses separate workers so CUDA state and
Prometheus counters cannot leak between configurations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULT_PREFIX = "BYTE_V2_SPECULATIVE_PROFILE_RESULT "


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
        "--backend",
        choices=("all", "byte_v2", "flash_attn"),
        default="all",
    )
    parser.add_argument(
        "--context-lens",
        nargs="+",
        type=int,
        default=[1024, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--spec-tokens",
        nargs="+",
        type=int,
        default=[0, 1, 3, 7],
        help="Draft-token counts. Zero is the non-speculative reference.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--prompt-lookup-min",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prompt-lookup-max",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_speculative_profile/results.jsonl",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
        help="Allow compile/CUDA-graph paths.",
    )
    parser.add_argument(
        "--compile-size-specialization",
        action="store_true",
        help="Compile a static model graph for the speculative query length.",
    )
    parser.set_defaults(enforce_eager=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-spec-tokens", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compile_size_specialization and args.enforce_eager:
        parser.error("--compile-size-specialization requires --no-enforce-eager")
    return args


@dataclass
class OpStats:
    cpu_seconds: float = 0.0
    event_pairs: list[tuple[Any, Any]] = field(default_factory=list)


class ProfileCollector:
    def __init__(self) -> None:
        self.enabled = False
        self.stats: dict[str, OpStats] = defaultdict(OpStats)
        self.restore_callbacks: list[Callable[[], None]] = []

    def reset(self) -> None:
        self.stats.clear()

    def install(self, backend: str) -> None:
        if backend == "byte_v2":
            self._install_byte_v2()
        else:
            self._install_flash_attn()

    def restore(self) -> None:
        for callback in reversed(self.restore_callbacks):
            callback()
        self.restore_callbacks.clear()

    def _timed_call(self, label: str, fn: Callable, args: tuple, kwargs: dict):
        if not self.enabled:
            return fn(*args, **kwargs)

        import torch

        device = _first_cuda_device(args, kwargs)
        start_event = None
        end_event = None
        if device is not None:
            start_event = torch.Event(enable_timing=True)
            end_event = torch.Event(enable_timing=True)
            start_event.record()
        cpu_start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stats[label].cpu_seconds += time.perf_counter() - cpu_start
            if start_event is not None and end_event is not None:
                end_event.record()
                self.stats[label].event_pairs.append((start_event, end_event))

    def _patch_module_function(
        self,
        module: Any,
        name: str,
        label_fn: Callable[[tuple, dict], str],
    ) -> None:
        if not hasattr(module, name):
            return
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            label = label_fn(args, kwargs)
            return self._timed_call(label, original, args, kwargs)

        setattr(module, name, wrapper)
        self.restore_callbacks.append(lambda: setattr(module, name, original))

    def _patch_method(
        self,
        cls: type,
        name: str,
        label_fn: Callable[[tuple, dict], str],
    ) -> None:
        original = getattr(cls, name)

        def wrapper(instance, *args, **kwargs):
            label = label_fn(args, kwargs)
            return self._timed_call(
                label,
                lambda *call_args, **call_kwargs: original(
                    instance, *call_args, **call_kwargs
                ),
                args,
                kwargs,
            )

        setattr(cls, name, wrapper)
        self.restore_callbacks.append(lambda: setattr(cls, name, original))

    def _install_byte_v2(self) -> None:
        from vllm.v1.attention.backends import byte_v2_attn

        def metadata_q(args: tuple, _kwargs: dict) -> str:
            metadata = next(
                (
                    arg
                    for arg in args
                    if hasattr(arg, "max_query_len") and hasattr(arg, "query_start_loc")
                ),
                None,
            )
            q_len = getattr(metadata, "max_query_len", "unknown")
            return f"byte_v2.forward.q{q_len}"

        def paged_q(args: tuple, kwargs: dict) -> str:
            output = args[0] if args else kwargs.get("output")
            q_len = output.shape[0] if output is not None else "unknown"
            return f"byte_v2.paged_decode.q{q_len}"

        def cache_n(args: tuple, kwargs: dict) -> str:
            slot_mapping = kwargs.get("slot_mapping")
            if slot_mapping is None and len(args) >= 5:
                slot_mapping = args[4]
            count = slot_mapping.shape[0] if slot_mapping is not None else "unknown"
            return f"byte_v2.cache_update.n{count}"

        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "forward",
            metadata_q,
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_forward_prefill_from_cache",
            lambda args, kwargs: metadata_q(args, kwargs).replace(
                "forward", "prefill_from_cache"
            ),
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_run_paged_decode",
            paged_q,
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_run_speculative_verify_gqa",
            lambda args, kwargs: (
                "byte_v2.speculative_verify_gqa.q"
                f"{args[4] if len(args) > 4 else kwargs.get('query_len', 'unknown')}"
            ),
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "do_kv_cache_update",
            cache_n,
        )

        for name in (
            "byte_v2_prepare_raw_staging",
            "byte_v2_hydrate_raw_staging_from_cache",
            "byte_v2_append_raw_staging",
            "byte_v2_commit_raw_staging_to_cache",
            "byte_v2_release_raw_staging",
            "byte_v2_release_raw_staging_and_update_flags",
            "byte_v2_reshape_and_cache",
            "byte_v2_update_cache_single_token",
            "byte_v2_update_cache_raw_staging",
            "byte_v2_update_cache_unsafe_flags",
            "byte_v2_paged_decode_attention",
            "byte_v2_paged_decode_attention_split_k",
            "byte_v2_paged_decode_attention_split_k_guarded",
            "byte_v2_speculative_verify_q4",
        ):
            self._patch_module_function(
                byte_v2_attn,
                name,
                lambda _args, _kwargs, op_name=name: f"byte_v2.op.{op_name}",
            )
        self._patch_module_function(
            byte_v2_attn,
            "byte_v2_speculative_verify_gqa",
            lambda _args, kwargs: (
                "byte_v2.op.byte_v2_speculative_verify_gqa.q"
                f"{kwargs.get('speculative_query_len', 'unknown')}"
            ),
        )

    def _install_flash_attn(self) -> None:
        from vllm.v1.attention.backends import flash_attn

        def metadata_q(args: tuple, _kwargs: dict) -> str:
            metadata = next(
                (arg for arg in args if hasattr(arg, "max_query_len")),
                None,
            )
            q_len = getattr(metadata, "max_query_len", "unknown")
            return f"flash_attn.forward.q{q_len}"

        def cache_n(args: tuple, kwargs: dict) -> str:
            slot_mapping = kwargs.get("slot_mapping")
            if slot_mapping is None and len(args) >= 5:
                slot_mapping = args[4]
            count = slot_mapping.shape[0] if slot_mapping is not None else "unknown"
            return f"flash_attn.cache_update.n{count}"

        self._patch_method(flash_attn.FlashAttentionImpl, "forward", metadata_q)
        self._patch_method(
            flash_attn.FlashAttentionImpl,
            "do_kv_cache_update",
            cache_n,
        )

        def kernel_q(args: tuple, kwargs: dict) -> str:
            q_len = kwargs.get("max_seqlen_q", "unknown")
            return f"flash_attn.op.flash_attn_varlen.q{q_len}"

        self._patch_module_function(
            flash_attn,
            "flash_attn_varlen_func",
            kernel_q,
        )
        self._patch_module_function(
            flash_attn,
            "reshape_and_cache_flash",
            lambda _args, _kwargs: "flash_attn.op.reshape_and_cache_flash",
        )

    def result(self) -> list[dict[str, Any]]:
        import torch

        torch.accelerator.synchronize()
        rows = []
        for name, stats in self.stats.items():
            cuda_ms = sum(start.elapsed_time(end) for start, end in stats.event_pairs)
            count = len(stats.event_pairs)
            cpu_ms = stats.cpu_seconds * 1000.0
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "cpu_total_ms": cpu_ms,
                    "cpu_avg_us": cpu_ms * 1000.0 / count if count else 0.0,
                    "cuda_total_ms": cuda_ms,
                    "cuda_avg_us": cuda_ms * 1000.0 / count if count else 0.0,
                }
            )
        return sorted(rows, key=lambda row: row["cuda_total_ms"], reverse=True)


def _iter_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _first_cuda_device(args: tuple, kwargs: dict):
    for tensor in _iter_tensors(args):
        if tensor.is_cuda:
            return tensor.device
    for tensor in _iter_tensors(kwargs):
        if tensor.is_cuda:
            return tensor.device
    return None


def _make_prompt_token_ids(model: str, prompt_len: int, offset: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    seed_text = (
        " one two three four five six seven eight nine ten eleven twelve."
        " red green blue yellow black white silver gold."
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("tokenizer produced no seed tokens")
    marker_offset = offset
    offset %= len(seed_ids)
    cycle = seed_ids[offset:] + seed_ids[:offset]
    repeats = (prompt_len + len(cycle) - 1) // len(cycle)
    prompt_ids = (cycle * repeats)[:prompt_len]
    if prompt_ids:
        prompt_ids[0] = 1000 + marker_offset
    return prompt_ids


def _build_llm(args: argparse.Namespace, backend: str, spec_tokens: int):
    from vllm import LLM

    max_model_len = max(args.context_lens) + args.max_tokens + spec_tokens + 8
    attention_backend = "BYTE_V2" if backend == "byte_v2" else "FLASH_ATTN"
    speculative_config = None
    if spec_tokens > 0:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": spec_tokens,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
        }
    compilation_config = None
    if args.compile_size_specialization:
        compilation_config = {"compile_sizes": [spec_tokens + 1]}
    return LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=args.batch_size,
        block_size=16,
        enable_prefix_caching=True,
        disable_log_stats=False,
        speculative_config=speculative_config,
        attention_config={"backend": attention_backend},
        compilation_config=compilation_config,
    )


def _metric_snapshot(llm) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in llm.get_metrics():
        if not metric.name.startswith("vllm:spec_decode_"):
            continue
        value = getattr(metric, "value", None)
        if value is not None:
            result[metric.name] = result.get(metric.name, 0) + value
            continue
        values = getattr(metric, "values", None)
        if values is not None:
            result[metric.name] = list(values)
    return result


def _metric_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in after.items():
        old = before.get(name, [0] * len(value) if isinstance(value, list) else 0)
        if isinstance(value, list):
            result[name] = [new - previous for new, previous in zip(value, old)]
        else:
            result[name] = value - old
    drafts = result.get("vllm:spec_decode_num_drafts", 0)
    draft_tokens = result.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = result.get("vllm:spec_decode_num_accepted_tokens", 0)
    result["acceptance_rate"] = accepted / draft_tokens if draft_tokens else None
    result["mean_acceptance_length"] = 1 + accepted / drafts if drafts else 1.0
    return result


def _run_generate(llm, prompts, sampling_params) -> tuple[float, list[list[int]]]:
    import torch

    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.accelerator.synchronize()
    elapsed = time.perf_counter() - start
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    return elapsed, token_ids


def _run_generate_step_profile(
    llm,
    prompts,
    sampling_params,
    collector: ProfileCollector,
    *,
    backend: str,
    verify_query_len: int,
) -> tuple[float, list[list[int]], dict[str, Any]]:
    import torch

    llm._add_completion_requests(  # noqa: SLF001
        prompts=prompts,
        params=sampling_params,
        use_tqdm=False,
    )
    outputs = []
    total_seconds = 0.0
    verify_seconds = 0.0
    other_seconds = 0.0
    verify_steps = 0
    other_steps = 0
    other_query_labels: dict[str, int] = defaultdict(int)
    if backend == "byte_v2":
        verify_labels = {
            f"byte_v2.op.byte_v2_speculative_verify_gqa.q{verify_query_len}",
            f"byte_v2.paged_decode.q{verify_query_len}",
        }
        query_prefixes = (
            "byte_v2.forward.q",
            "byte_v2.paged_decode.q",
            "byte_v2.prefill_from_cache.q",
            "byte_v2.op.byte_v2_speculative_verify_gqa.q",
        )
    else:
        verify_labels = {f"flash_attn.op.flash_attn_varlen.q{verify_query_len}"}
        query_prefixes = (
            "flash_attn.forward.q",
            "flash_attn.op.flash_attn_varlen.q",
        )

    collector.reset()
    collector.enabled = True
    try:
        while llm.llm_engine.has_unfinished_requests():
            before_counts = {
                name: len(stats.event_pairs) for name, stats in collector.stats.items()
            }
            torch.accelerator.synchronize()
            step_start = time.perf_counter()
            step_outputs = llm.llm_engine.step()
            torch.accelerator.synchronize()
            step_seconds = time.perf_counter() - step_start
            total_seconds += step_seconds
            outputs.extend(output for output in step_outputs if output.finished)

            changed_labels = {
                name
                for name, stats in collector.stats.items()
                if len(stats.event_pairs) > before_counts.get(name, 0)
            }
            if changed_labels & verify_labels:
                verify_steps += 1
                verify_seconds += step_seconds
            else:
                other_steps += 1
                other_seconds += step_seconds
                for label in changed_labels:
                    if label.startswith(query_prefixes):
                        other_query_labels[label] += 1
    finally:
        collector.enabled = False

    outputs.sort(key=lambda output: int(output.request_id))
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    return (
        total_seconds,
        token_ids,
        {
            "verify_steps": verify_steps,
            "verify_wall_seconds": verify_seconds,
            "other_steps": other_steps,
            "other_wall_seconds": other_seconds,
            "other_query_labels": dict(sorted(other_query_labels.items())),
        },
    )


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from vllm import SamplingParams, TokensPrompt

    assert args.backend in ("byte_v2", "flash_attn")
    assert args.worker_spec_tokens is not None
    spec_tokens = args.worker_spec_tokens
    collector = ProfileCollector()
    collector.install(args.backend)

    init_start = time.perf_counter()
    llm = _build_llm(args, args.backend, spec_tokens)
    torch.accelerator.synchronize()
    init_seconds = time.perf_counter() - init_start
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    cache_fill_params = SamplingParams(max_tokens=1, temperature=0.0)

    try:
        for context_index, context_len in enumerate(args.context_lens):
            prompts = [
                TokensPrompt(
                    prompt_token_ids=_make_prompt_token_ids(
                        args.model,
                        context_len,
                        offset=context_index * 11 + request_idx * 7,
                    )
                )
                for request_idx in range(args.batch_size)
            ]

            warmup_seconds = 0.0
            for prompt in prompts:
                prompt_seconds, _ = _run_generate(
                    llm,
                    [prompt],
                    cache_fill_params,
                )
                warmup_seconds += prompt_seconds

            before = _metric_snapshot(llm)
            measured_seconds, token_ids = _run_generate(llm, prompts, sampling_params)
            measured_metrics = _metric_diff(before, _metric_snapshot(llm))

            profile_metrics_before = _metric_snapshot(llm)
            profile_seconds, profile_token_ids, step_profile = (
                _run_generate_step_profile(
                    llm,
                    prompts,
                    sampling_params,
                    collector,
                    backend=args.backend,
                    verify_query_len=spec_tokens + 1,
                )
            )
            profile_metrics = _metric_diff(
                profile_metrics_before,
                _metric_snapshot(llm),
            )
            steady_tokens = profile_metrics.get("vllm:spec_decode_num_drafts", 0)
            steady_tokens += profile_metrics.get(
                "vllm:spec_decode_num_accepted_tokens", 0
            )
            verify_seconds = step_profile["verify_wall_seconds"]
            step_profile["steady_output_tokens"] = steady_tokens
            step_profile["steady_output_tokens_per_second"] = (
                steady_tokens / verify_seconds if verify_seconds > 0 else None
            )

            result = {
                "backend": args.backend,
                "spec_tokens": spec_tokens,
                "verify_query_len": spec_tokens + 1,
                "context_len": context_len,
                "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
                "enforce_eager": args.enforce_eager,
                "compile_size_specialization": (args.compile_size_specialization),
                "init_seconds": init_seconds,
                "warmup_seconds": warmup_seconds,
                "measured_seconds": measured_seconds,
                "profile_seconds": profile_seconds,
                "output_tokens": sum(len(tokens) for tokens in token_ids),
                "output_tokens_per_second": (
                    sum(len(tokens) for tokens in token_ids) / measured_seconds
                ),
                "token_ids": token_ids,
                "profile_token_ids_match": token_ids == profile_token_ids,
                "spec_metrics": measured_metrics,
                "profile_ops": collector.result(),
                "step_profile": step_profile,
            }
            print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    finally:
        collector.restore()
        del llm
        gc.collect()
        torch.accelerator.empty_cache()


def _worker_backends(backend: str) -> list[str]:
    return ["byte_v2", "flash_attn"] if backend == "all" else [backend]


def _parse_worker_results(output: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len(RESULT_PREFIX) :])
        for line in output.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]


def _run_child(
    args: argparse.Namespace,
    backend: str,
    spec_tokens: int,
) -> list[dict[str, Any]]:
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
        "--worker-spec-tokens",
        str(spec_tokens),
        "--model",
        args.model,
        "--context-lens",
        *[str(value) for value in args.context_lens],
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(args.batch_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--prompt-lookup-min",
        str(args.prompt_lookup_min),
        "--prompt-lookup-max",
        str(args.prompt_lookup_max),
    ]
    if not args.enforce_eager:
        cmd.append("--no-enforce-eager")
    if args.compile_size_specialization:
        cmd.append("--compile-size-specialization")

    label = f"backend={backend} spec_tokens={spec_tokens}"
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
        print(proc.stdout[-12000:])
        raise RuntimeError(f"worker failed for {label}")
    results = _parse_worker_results(proc.stdout)
    if len(results) != len(args.context_lens):
        print(proc.stdout[-12000:])
        raise RuntimeError(f"worker returned incomplete results for {label}")
    for result in results:
        print(
            "sweep.done "
            f"{label} context={result['context_len']} "
            f"time={result['measured_seconds']:.6f}s "
            f"tokens/s={result['output_tokens_per_second']:.3f} "
            f"accept={result['spec_metrics']['acceptance_rate']}",
            flush=True,
        )
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    print(
        "backend,spec_tokens,context_len,batch_size,seconds,tokens_per_second,"
        "acceptance_rate,mean_acceptance_length,reference_match"
    )
    references = {
        (row["backend"], row["context_len"], row["batch_size"]): row
        for row in results
        if row["spec_tokens"] == 0
    }
    for row in sorted(
        results,
        key=lambda value: (
            value["context_len"],
            value["backend"],
            value["spec_tokens"],
        ),
    ):
        reference = references.get(
            (row["backend"], row["context_len"], row["batch_size"])
        )
        reference_match = (
            reference is not None and reference["token_ids"] == row["token_ids"]
        )
        metrics = row["spec_metrics"]
        print(
            f"{row['backend']},{row['spec_tokens']},{row['context_len']},"
            f"{row['batch_size']},{row['measured_seconds']:.6f},"
            f"{row['output_tokens_per_second']:.3f},"
            f"{metrics['acceptance_rate']},{metrics['mean_acceptance_length']},"
            f"{reference_match}"
        )


def main() -> None:
    _prepend_venv_bin_to_path()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    args = parse_args()
    if args.worker:
        _run_worker(args)
        return
    if any(value < 0 for value in args.spec_tokens):
        raise ValueError("--spec-tokens values must be non-negative")

    results = []
    for backend in _worker_backends(args.backend):
        for spec_tokens in args.spec_tokens:
            results.extend(_run_child(args, backend, spec_tokens))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"sweep.output_jsonl={output_path}")
    _print_summary(results)


if __name__ == "__main__":
    main()
