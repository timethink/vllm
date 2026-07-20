# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile local ByteV2 E2E generation with per-op CUDA event timing."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BYTE_V2_PROFILE_OP_NAMES = (
    "byte_v2_prepare_raw_staging",
    "byte_v2_hydrate_raw_staging_from_cache",
    "byte_v2_hydrate_raw_staging_from_hybrid_cache",
    "byte_v2_append_raw_staging",
    "byte_v2_commit_raw_staging_to_cache",
    "byte_v2_commit_raw_staging_to_hybrid_cache",
    "byte_v2_update_hybrid_cache_raw_staging_q1",
    "byte_v2_test_force_promote_raw_staging_q1",
    "byte_v2_release_raw_staging",
    "byte_v2_reshape_and_cache",
    "byte_v2_reshape_and_cache_sideband_high",
    "byte_v2_update_cache_single_token",
    "byte_v2_update_cache_unsafe_flags",
    "byte_v2_collect_cache_stats",
    "byte_v2_prefill_attention",
    "byte_v2_paged_decode_attention",
    "byte_v2_paged_decode_attention_split_k",
    "byte_v2_paged_decode_attention_split_k_guarded",
    "byte_v2_fa2_hybrid_paged_decode_attention",
    "byte_v2_reset_raw_fallback_pages",
)


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
        "--prompt",
        default=(
            "You are testing ByteV2 prefix cache correctness. Explain why "
            "deterministic decoding matters for regression tests, and keep "
            "the answer concise."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--prompt-token-len",
        type=int,
        default=None,
        help="Use a synthetic TokensPrompt with exactly this many prompt tokens.",
    )
    parser.add_argument(
        "--profile-cached",
        action="store_true",
        help="Run the same request twice and print separate first/cached profiles.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Also collect a torch.profiler trace around the generate call.",
    )
    parser.add_argument(
        "--trace-path",
        default="profiles/byte_v2_torch_profile.json",
        help="Chrome trace path used with --torch-profile.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path for machine-readable first/cached profile rows.",
    )
    parser.add_argument(
        "--collect-hybrid-state",
        action="store_true",
        help=(
            "Collect per-layer persistent raw-fallback state after all timed "
            "generation work has finished."
        ),
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
        help="Allow vLLM compile/cudagraph paths instead of forcing eager mode.",
    )
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args()


@dataclass
class OpStats:
    cpu_seconds: float = 0.0
    event_pairs: list[tuple[Any, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.event_pairs)


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


def _first_cuda_device(args: tuple[Any, ...], kwargs: dict[str, Any]):
    for tensor in _iter_tensors(args):
        if tensor.is_cuda:
            return tensor.device
    for tensor in _iter_tensors(kwargs):
        if tensor.is_cuda:
            return tensor.device
    return None


def _install_byte_v2_timers():
    import torch

    from vllm.v1.attention.backends import byte_v2_attn

    stats: dict[str, OpStats] = defaultdict(OpStats)
    originals = {}

    def make_wrapper(name: str, fn):
        def wrapper(*args, **kwargs):
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
                stats[name].cpu_seconds += time.perf_counter() - cpu_start
                if start_event is not None and end_event is not None:
                    end_event.record()
                    stats[name].event_pairs.append((start_event, end_event))

        return wrapper

    for name in _BYTE_V2_PROFILE_OP_NAMES:
        if hasattr(byte_v2_attn, name):
            fn = getattr(byte_v2_attn, name)
            originals[name] = fn
            setattr(byte_v2_attn, name, make_wrapper(name, fn))

    return stats, originals


def _restore_byte_v2_timers(originals: dict[str, Any]) -> None:
    from vllm.v1.attention.backends import byte_v2_attn

    for name, fn in originals.items():
        setattr(byte_v2_attn, name, fn)


def _reset_stats(stats: dict[str, OpStats]) -> None:
    stats.clear()


def _cuda_ms(stats: OpStats) -> float:
    total = 0.0
    for start_event, end_event in stats.event_pairs:
        total += start_event.elapsed_time(end_event)
    return total


def _print_stats(
    stats: dict[str, OpStats],
    generate_seconds: float,
    *,
    label: str,
) -> dict[str, Any]:
    import torch

    torch.accelerator.synchronize()
    rows = []
    for name, op_stats in stats.items():
        cuda_ms = _cuda_ms(op_stats)
        cpu_ms = op_stats.cpu_seconds * 1000.0
        count = op_stats.count
        rows.append(
            (
                cuda_ms,
                name,
                count,
                cpu_ms,
                cpu_ms * 1000.0 / count if count else 0.0,
                cuda_ms * 1000.0 / count if count else 0.0,
            )
        )
    rows.sort(reverse=True)
    custom_cuda_ms = sum(row[0] for row in rows)
    custom_cpu_ms = sum(row[3] for row in rows)
    result = {
        "label": label,
        "wall_generate_ms": generate_seconds * 1000.0,
        "byte_v2_custom_cuda_total_ms": custom_cuda_ms,
        "byte_v2_custom_cpu_launch_total_ms": custom_cpu_ms,
        "byte_v2_custom_coverage": custom_cuda_ms / (generate_seconds * 1000.0),
        "ops": [
            {
                "name": name,
                "count": count,
                "cpu_total_ms": cpu_ms,
                "cpu_avg_us": cpu_avg_us,
                "cuda_total_ms": cuda_ms,
                "cuda_avg_us": cuda_avg_us,
            }
            for cuda_ms, name, count, cpu_ms, cpu_avg_us, cuda_avg_us in rows
        ],
    }
    print(f"profile.{label}.wall.generate_ms={generate_seconds * 1000.0:.3f}")
    print(f"profile.{label}.byte_v2_custom.cuda_total_ms={custom_cuda_ms:.3f}")
    print(f"profile.{label}.byte_v2_custom.cpu_launch_total_ms={custom_cpu_ms:.3f}")
    print(
        f"profile.{label}.byte_v2_custom.coverage="
        f"{custom_cuda_ms / (generate_seconds * 1000.0):.6f}"
    )
    print(f"profile.{label}.op,count,cpu_total_ms,cpu_avg_us,cuda_total_ms,cuda_avg_us")
    for cuda_ms, name, count, cpu_ms, cpu_avg_us, cuda_avg_us in rows:
        print(
            f"{name},{count},{cpu_ms:.3f},{cpu_avg_us:.3f},"
            f"{cuda_ms:.3f},{cuda_avg_us:.3f}"
        )
    return result


def _write_profile_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _collect_hybrid_raw_fallback_state(llm) -> dict[str, Any]:
    """Collect ByteV2 raw-sidecar state without allocating or mutating it."""
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", {})

    layers = []
    initialized = []
    for layer_name, module in sorted(static_forward_context.items()):
        impl = getattr(module, "impl", None)
        if impl is None or not hasattr(impl, "raw_fallback_store"):
            continue
        store = impl.raw_fallback_store
        state = getattr(store, "current_state", None) if store is not None else None
        layer = {
            "name": layer_name,
            "store_enabled": store is not None,
            "initialized": state is not None,
            "raw_page_count": None,
            "free_count": None,
            "slot_count": None,
            "fatal": None,
        }
        layers.append(layer)
        if state is not None:
            initialized.append((layer, state))

    if initialized:
        import torch

        counters = torch.stack(
            [
                torch.stack(
                    (
                        state.page_to_raw_slot.ge(0).sum(dtype=torch.int64),
                        state.free_count.reshape(-1)[0].to(dtype=torch.int64),
                        state.fatal.reshape(-1)[0].to(dtype=torch.int64),
                    )
                )
                for _, state in initialized
            ]
        )
        counter_rows = counters.detach().cpu().tolist()
        for (layer, state), (raw_page_count, free_count, fatal) in zip(
            initialized, counter_rows
        ):
            layer["raw_page_count"] = int(raw_page_count)
            layer["free_count"] = int(free_count)
            layer["slot_count"] = int(state.raw_pages.shape[0])
            layer["fatal"] = int(fatal)

    enabled_layer_count = sum(layer["store_enabled"] for layer in layers)
    fully_initialized = (
        enabled_layer_count > 0 and len(initialized) == enabled_layer_count
    )
    if fully_initialized:
        raw_page_count = sum(layer["raw_page_count"] for layer in layers)
        free_count = sum(layer["free_count"] for layer in layers)
        slot_count = sum(layer["slot_count"] for layer in layers)
        fatal = max(layer["fatal"] for layer in layers)
    else:
        raw_page_count = free_count = slot_count = fatal = None

    return {
        "enabled": enabled_layer_count > 0,
        "layer_count": len(layers),
        "enabled_layer_count": enabled_layer_count,
        "initialized_layer_count": len(initialized),
        "fully_initialized": fully_initialized,
        "raw_page_count": raw_page_count,
        "free_count": free_count,
        "slot_count": slot_count,
        "fatal": fatal,
        "layers": layers,
    }


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


def _build_llm(args: argparse.Namespace):
    from vllm import LLM

    max_model_len = args.max_model_len
    if args.prompt_token_len is not None:
        max_model_len = max(max_model_len, args.prompt_token_len + args.max_tokens)
    return LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=args.batch_size,
        block_size=16,
        attention_config={"backend": "BYTE_V2"},
    )


def _run_generate(llm, args: argparse.Namespace):
    from vllm import SamplingParams, TokensPrompt

    if args.prompt_token_len is None:
        prompts = [args.prompt for _ in range(args.batch_size)]
    else:
        prompt_token_ids = _make_prompt_token_ids(args.model, args.prompt_token_len)
        prompts = [
            TokensPrompt(prompt_token_ids=prompt_token_ids)
            for _ in range(args.batch_size)
        ]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generate_seconds = time.perf_counter() - start
    return outputs, generate_seconds


def main() -> None:
    _prepend_venv_bin_to_path()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch

    args = parse_args()
    stats, originals = _install_byte_v2_timers()
    llm = _build_llm(args)
    try:
        _reset_stats(stats)
        if args.torch_profile:
            if args.profile_cached:
                raise ValueError(
                    "--torch-profile cannot be combined with --profile-cached"
                )
            from torch.profiler import ProfilerActivity, profile

            trace_path = Path(args.trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                outputs, generate_seconds = _run_generate(llm, args)
            torch.accelerator.synchronize()
            prof.export_chrome_trace(str(trace_path))
            print(f"profile.torch_trace={trace_path}")
            print(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            )
        else:
            outputs, generate_seconds = _run_generate(llm, args)
        profile_results = [_print_stats(stats, generate_seconds, label="first")]
        if args.profile_cached:
            _reset_stats(stats)
            cached_outputs, cached_generate_seconds = _run_generate(llm, args)
            profile_results.append(
                _print_stats(stats, cached_generate_seconds, label="cached")
            )
            outputs = cached_outputs
        token_ids_by_output = []
        for idx, output in enumerate(outputs):
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            token_ids_by_output.append(token_ids)
            print(f"output.{idx}.token_ids={token_ids}")
        result = {
            "prompt_token_len": args.prompt_token_len,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "enforce_eager": args.enforce_eager,
            "profile_cached": args.profile_cached,
            "profiles": profile_results,
            "token_ids": token_ids_by_output,
            "hybrid_raw_fallback_state": (
                _collect_hybrid_raw_fallback_state(llm)
                if args.collect_hybrid_state
                else None
            ),
        }
        if args.output_jsonl is not None:
            _write_profile_jsonl(args.output_jsonl, [result])
        print("BYTE_V2_E2E_PROFILE_RESULT " + json.dumps(result, sort_keys=True))
    finally:
        _restore_byte_v2_timers(originals)
        del llm
        gc.collect()
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
