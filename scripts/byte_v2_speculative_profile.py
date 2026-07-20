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
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULT_PREFIX = "BYTE_V2_SPECULATIVE_PROFILE_RESULT "
POOL_DIAGNOSTIC_PREFIX = "BYTE_V2_OUTLIER_POOL_DIAGNOSTIC "

_BYTE_V2_PROFILE_OP_NAMES = (
    "byte_v2_prepare_raw_staging",
    "byte_v2_hydrate_raw_staging_from_cache",
    "byte_v2_hydrate_raw_staging_from_hybrid_cache",
    "byte_v2_append_raw_staging",
    "byte_v2_commit_raw_staging_to_cache",
    "byte_v2_commit_raw_staging_to_hybrid_cache",
    "byte_v2_release_raw_staging",
    "byte_v2_release_raw_staging_and_update_flags",
    "byte_v2_reshape_and_cache",
    "byte_v2_update_cache_single_token",
    "byte_v2_update_cache_raw_staging",
    "byte_v2_update_hybrid_cache_raw_staging_q1",
    "byte_v2_test_force_promote_raw_staging_q1",
    "byte_v2_update_cache_unsafe_flags",
    "byte_v2_paged_decode_attention",
    "byte_v2_paged_decode_attention_split_k",
    "byte_v2_paged_decode_attention_split_k_guarded",
    "byte_v2_fa2_hybrid_paged_decode_attention",
    "byte_v2_reset_raw_fallback_pages",
    "byte_v2_speculative_verify_q4",
    "byte_v2_speculative_verify_ragged_q4",
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
        "--sharegpt-rows-dir",
        help=(
            "Directory containing JSON responses from the Hugging Face "
            "datasets-server rows API. Use first-turn user prompts instead "
            "of the synthetic repeated prompt."
        ),
    )
    parser.add_argument("--sharegpt-seed", type=int, default=20260717)
    parser.add_argument(
        "--sharegpt-target-lens",
        nargs="+",
        type=int,
        help=(
            "Select the closest distinct ShareGPT prompt for each requested "
            "token length. The number of targets must equal --batch-size."
        ),
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable prefix caching so warmup cannot cache measured prompts.",
    )
    parser.add_argument(
        "--diagnose-outlier-pool",
        action="store_true",
        help=(
            "Inspect raw staging immediately before each ByteV2 cache commit "
            "and report the V5 outlier-pool demand. This disables the fused "
            "native raw-staging update and is not a performance mode."
        ),
    )
    parser.add_argument(
        "--collect-hybrid-state",
        action="store_true",
        help=(
            "Collect per-layer persistent raw-fallback state after all timed "
            "generation and profiler replay work has finished."
        ),
    )
    parser.add_argument(
        "--diagnose-forced-raw-lifecycle",
        action="store_true",
        help=(
            "After warmup, arm a test-only device latch before measured and "
            "profile Q1 runs so each ByteV2 layer promotes one page to the "
            "persistent raw sidecar. This adds a CUDA launch and raw-page "
            "copy, disables prefix caching, and is never valid for TPS "
            "conclusions."
        ),
    )
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
    if args.diagnose_forced_raw_lifecycle:
        diagnostic_spec_tokens = (
            [args.worker_spec_tokens] if args.worker else args.spec_tokens
        )
        if args.backend != "byte_v2":
            parser.error("--diagnose-forced-raw-lifecycle requires --backend byte_v2")
        if diagnostic_spec_tokens != [0]:
            parser.error(
                "--diagnose-forced-raw-lifecycle requires exactly --spec-tokens 0"
            )
        if args.batch_size != 1:
            parser.error("--diagnose-forced-raw-lifecycle requires --batch-size 1")
        if args.max_tokens < 3:
            parser.error("--diagnose-forced-raw-lifecycle requires --max-tokens >= 3")
        if args.diagnose_outlier_pool:
            parser.error(
                "--diagnose-forced-raw-lifecycle cannot be combined with "
                "--diagnose-outlier-pool"
            )
        args.disable_prefix_caching = True
        args.collect_hybrid_state = True
    return args


@dataclass
class OpStats:
    cpu_seconds: float = 0.0
    event_pairs: list[tuple[Any, Any]] = field(default_factory=list)


class ProfileCollector:
    def __init__(self) -> None:
        self.enabled = False
        self.stats: dict[str, OpStats] = defaultdict(OpStats)
        self.byte_v2_prefill_patterns: dict[str, int] = defaultdict(int)
        self.byte_v2_prefill_shapes: dict[str, int] = defaultdict(int)
        self.restore_callbacks: list[Callable[[], None]] = []

    def reset(self) -> None:
        self.stats.clear()
        self.byte_v2_prefill_patterns.clear()
        self.byte_v2_prefill_shapes.clear()

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

        def cached_prefill_q(args: tuple, kwargs: dict) -> str:
            return metadata_q(args, kwargs).replace("forward", "prefill_from_cache")

        def ragged_candidate_q(args: tuple, kwargs: dict) -> str:
            label = metadata_q(args, kwargs).replace("forward", "ragged_candidate")
            if not self.enabled:
                return label
            metadata = next(
                (
                    arg
                    for arg in args
                    if hasattr(arg, "query_start_loc_cpu")
                    and hasattr(arg, "num_actual_tokens")
                ),
                None,
            )
            if metadata is None or not 1 < metadata.max_query_len <= 4:
                return label
            starts = metadata.query_start_loc_cpu.tolist()
            num_actual_tokens = int(metadata.num_actual_tokens)
            query_lens = [
                max(
                    0,
                    min(int(end), num_actual_tokens)
                    - min(int(start), num_actual_tokens),
                )
                for start, end in zip(starts[:-1], starts[1:])
            ]
            active_query_lens = [length for length in query_lens if length > 0]
            pattern = ",".join(str(length) for length in active_query_lens)
            self.byte_v2_prefill_patterns[pattern or "empty"] += 1
            query = args[0] if args else kwargs.get("query")
            output = args[2] if len(args) > 2 else kwargs.get("output")
            shape_key = (
                f"q{metadata.max_query_len}.actual{num_actual_tokens}."
                f"query{query.shape[0]}.output{output.shape[0]}."
                f"requests{len(query_lens)}"
            )
            self.byte_v2_prefill_shapes[shape_key] += 1
            return label

        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "forward",
            metadata_q,
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_forward_prefill_from_cache",
            cached_prefill_q,
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_run_paged_decode",
            paged_q,
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "_speculative_ragged_q4_num_requests",
            ragged_candidate_q,
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
            "_run_speculative_verify_ragged_q4",
            lambda _args, _kwargs: "byte_v2.speculative_verify_ragged_q4",
        )
        self._patch_method(
            byte_v2_attn.ByteV2AttentionImpl,
            "do_kv_cache_update",
            cache_n,
        )

        for name in _BYTE_V2_PROFILE_OP_NAMES:
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

    def byte_v2_pattern_result(self) -> dict[str, int] | None:
        if not self.byte_v2_prefill_patterns:
            return None
        return dict(sorted(self.byte_v2_prefill_patterns.items()))

    def byte_v2_shape_result(self) -> dict[str, int] | None:
        if not self.byte_v2_prefill_shapes:
            return None
        return dict(sorted(self.byte_v2_prefill_shapes.items()))


@dataclass
class OutlierPoolDiagnostic:
    call_count: int = 0
    active_pages: int = 0
    overflow_pages: int = 0
    max_pool_capacity: int = 0
    max_outliers: int = 0
    max_record: dict[str, Any] | None = None

    @staticmethod
    def _segment_capacity(count: int) -> int:
        return 1 << (count - 1).bit_length() if count else 0

    def inspect(self, raw_staging, staging_to_physical_block, valid_rows) -> None:
        import torch

        physical_blocks = staging_to_physical_block.detach().cpu().tolist()
        row_counts = valid_rows.detach().cpu().tolist()
        active_slots = [
            slot
            for slot, physical_block in enumerate(physical_blocks)
            if physical_block >= 0 and row_counts[slot] > 0
        ]
        self.call_count += 1
        if not active_slots:
            return

        active_indices = torch.tensor(
            active_slots,
            dtype=torch.long,
            device=raw_staging.device,
        )
        active_raw = (
            raw_staging.index_select(0, active_indices)
            .detach()
            .cpu()
            .reshape(len(active_slots), -1)
        )
        for active_idx, staging_slot in enumerate(active_slots):
            rows = min(int(row_counts[staging_slot]), 16)
            raw_slot = active_raw[active_idx]
            pool_capacity = 0
            total_outliers = 0
            outlier_tiles = 0
            max_tile_outliers = 0
            row_outliers = [0] * rows
            row_zero_values = [0] * rows
            row_high7_chunks = []
            for kv_side in range(2):
                side_offset = kv_side * 32768
                side_pairs = raw_slot[side_offset : side_offset + 32768].reshape(
                    8, 16, 128, 2
                )
                side_high7 = side_pairs[..., 1].bitwise_and(0x7F)
                row_high7_chunks.append(
                    side_high7[:, :rows].permute(1, 0, 2).reshape(rows, -1)
                )
                side_zeros = side_pairs[..., 0].eq(0) & side_pairs[..., 1].eq(0)
                side_row_zeros = side_zeros[:, :rows].sum(dim=(0, 2)).tolist()
                row_zero_values = [
                    total + int(side_count)
                    for total, side_count in zip(row_zero_values, side_row_zeros)
                ]
                for kv_head in range(8):
                    for dim_tile in range(8):
                        tile_values = side_high7[
                            kv_head,
                            :rows,
                            dim_tile * 16 : (dim_tile + 1) * 16,
                        ]
                        values = tile_values.reshape(-1)
                        high_counts = torch.bincount(values, minlength=128)
                        window_counts = high_counts.unfold(0, 8, 1).sum(dim=1)
                        best_base = int(window_counts.argmax().item())
                        outlier_count = int(
                            values.numel() - window_counts[best_base].item()
                        )
                        if outlier_count:
                            tile_row_outliers = (
                                (tile_values < best_base)
                                | (tile_values >= best_base + 8)
                            ).sum(dim=1)
                            row_outliers = [
                                total + int(tile_count)
                                for total, tile_count in zip(
                                    row_outliers,
                                    tile_row_outliers.tolist(),
                                )
                            ]
                            outlier_tiles += 1
                            total_outliers += outlier_count
                            pool_capacity += self._segment_capacity(outlier_count)
                            max_tile_outliers = max(
                                max_tile_outliers,
                                outlier_count,
                            )

            row_high7 = torch.cat(row_high7_chunks, dim=1)
            row_high7_stats = []
            for row_values in row_high7:
                counts = torch.bincount(row_values, minlength=128)
                row_high7_stats.append(
                    {
                        "min": int(row_values.min().item()),
                        "max": int(row_values.max().item()),
                        "mode": int(counts.argmax().item()),
                        "mode_count": int(counts.max().item()),
                    }
                )

            self.active_pages += 1
            if pool_capacity > 1024:
                self.overflow_pages += 1
            record = {
                "call": self.call_count,
                "staging_slot": staging_slot,
                "physical_block": int(physical_blocks[staging_slot]),
                "valid_rows": rows,
                "pool_capacity": pool_capacity,
                "outliers": total_outliers,
                "outlier_tiles": outlier_tiles,
                "max_tile_outliers": max_tile_outliers,
                "row_outliers": row_outliers,
                "row_zero_values": row_zero_values,
                "row_high7_stats": row_high7_stats,
            }
            if pool_capacity > self.max_pool_capacity:
                self.max_pool_capacity = pool_capacity
                self.max_outliers = total_outliers
                self.max_record = record
                print(
                    POOL_DIAGNOSTIC_PREFIX + json.dumps(record, sort_keys=True),
                    flush=True,
                )

    def result(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "active_pages": self.active_pages,
            "overflow_pages": self.overflow_pages,
            "max_pool_capacity": self.max_pool_capacity,
            "max_outliers": self.max_outliers,
            "max_record": self.max_record,
        }


def _install_outlier_pool_diagnostic(
    collector: ProfileCollector,
) -> OutlierPoolDiagnostic:
    from vllm.v1.attention.backends import byte_v2_attn

    diagnostic = OutlierPoolDiagnostic()
    original = byte_v2_attn.byte_v2_commit_raw_staging_to_cache

    def wrapper(*args, **kwargs):
        raw_staging = args[0] if args else kwargs["raw_staging"]
        staging_to_physical_block = (
            args[2] if len(args) > 2 else kwargs["staging_to_physical_block"]
        )
        valid_rows = args[3] if len(args) > 3 else kwargs["valid_rows"]
        diagnostic.inspect(
            raw_staging,
            staging_to_physical_block,
            valid_rows,
        )
        return original(*args, **kwargs)

    byte_v2_attn.byte_v2_commit_raw_staging_to_cache = wrapper
    collector.restore_callbacks.append(
        lambda: setattr(
            byte_v2_attn,
            "byte_v2_commit_raw_staging_to_cache",
            original,
        )
    )
    return diagnostic


def _iter_forced_raw_diagnostic_stores(llm):
    """Yield each test-enabled raw sidecar store exactly once."""
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", {})
    seen = set()
    for module in static_forward_context.values():
        impl = getattr(module, "impl", None)
        store = getattr(impl, "raw_fallback_store", None)
        if store is None or id(store) in seen:
            continue
        state = getattr(store, "current_state", None)
        diagnostic = getattr(state, "forced_raw_diagnostic", None)
        if not getattr(store, "forced_raw_diagnostic", False) and diagnostic is None:
            continue
        seen.add(id(store))
        yield store


def _control_forced_raw_diagnostic(llm, *, arm: bool) -> dict[str, Any]:
    """Clear or arm all device latches, then return synchronized evidence."""
    stores = list(_iter_forced_raw_diagnostic_stores(llm))
    if not stores:
        raise RuntimeError("ByteV2 forced raw diagnostic stores are unavailable")
    method_name = "arm_forced_raw_promotion" if arm else "clear_forced_raw_diagnostic"
    for store in stores:
        method = getattr(store, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"ByteV2 raw fallback store cannot {method_name.replace('_', ' ')}"
            )
        method()
    return _collect_hybrid_raw_fallback_state(llm)


def _reset_forced_raw_pages_through_runner(llm) -> dict[str, Any]:
    """Exercise the runner's reset-before-zero hook after a diagnostic request."""
    import torch

    mapped_block_ids = set()
    for store in _iter_forced_raw_diagnostic_stores(llm):
        state = getattr(store, "current_state", None)
        if state is None:
            raise RuntimeError("ByteV2 forced raw sidecar state is uninitialized")
        mapped = state.page_to_raw_slot.ge(0).nonzero().reshape(-1)
        mapped_block_ids.update(int(value) for value in mapped.cpu().tolist())
    if not mapped_block_ids:
        raise RuntimeError("ByteV2 forced raw diagnostic published no mapped pages")

    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    driver_worker = getattr(executor, "driver_worker", None)
    model_runner = getattr(driver_worker, "model_runner", None)
    reset = getattr(model_runner, "_zero_block_ids", None)
    if not callable(reset):
        raise RuntimeError("ByteV2 diagnostic cannot access the runner reset hook")
    ordered_block_ids = sorted(mapped_block_ids)
    reset(ordered_block_ids)
    torch.accelerator.synchronize()
    return {
        "source": "runner_zero_block_ids_after_request",
        "physical_block_ids": ordered_block_ids,
    }


def _collect_hybrid_raw_fallback_state(llm) -> dict[str, Any]:
    """Collect ByteV2 raw-sidecar state without allocating or mutating it."""
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", {})

    layers = []
    initialized = []
    diagnostic_initialized = []
    diagnostic_requested = False
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
        diagnostic = getattr(state, "forced_raw_diagnostic", None)
        diagnostic_enabled = bool(
            store is not None
            and (
                getattr(store, "forced_raw_diagnostic", False) or diagnostic is not None
            )
        )
        diagnostic_requested = diagnostic_requested or diagnostic_enabled
        if diagnostic_enabled:
            layer.update(
                {
                    "forced_raw_diagnostic_enabled": True,
                    "forced_raw_latch": None,
                    "promotion_count": None,
                    "mapped_page_visit_count": None,
                }
            )
        if diagnostic is not None:
            if diagnostic.numel() != 3:
                raise RuntimeError(
                    "ByteV2 forced raw diagnostic tensor must have 3 elements"
                )
            diagnostic_initialized.append((layer, diagnostic))
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

    if diagnostic_initialized:
        import torch

        diagnostic_rows = (
            torch.stack(
                [
                    diagnostic.reshape(-1).to(dtype=torch.int64)
                    for _, diagnostic in diagnostic_initialized
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )
        for (layer, _), (latch, promotions, mapped_visits) in zip(
            diagnostic_initialized, diagnostic_rows
        ):
            layer["forced_raw_latch"] = int(latch)
            layer["promotion_count"] = int(promotions)
            layer["mapped_page_visit_count"] = int(mapped_visits)

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

    result = {
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
    if diagnostic_requested:
        diagnostic_layers = [
            layer
            for layer in layers
            if layer.get("forced_raw_diagnostic_enabled", False)
        ]
        diagnostic_fully_initialized = bool(diagnostic_layers) and len(
            diagnostic_initialized
        ) == len(diagnostic_layers)
        if diagnostic_fully_initialized:
            armed_layer_count = sum(
                layer["forced_raw_latch"] == 1 for layer in diagnostic_layers
            )
            promotion_count = sum(
                layer["promotion_count"] for layer in diagnostic_layers
            )
            mapped_page_visit_count = sum(
                layer["mapped_page_visit_count"] for layer in diagnostic_layers
            )
        else:
            armed_layer_count = promotion_count = mapped_page_visit_count = None
        result["forced_raw_lifecycle"] = {
            "enabled": True,
            "enabled_layer_count": len(diagnostic_layers),
            "initialized_layer_count": len(diagnostic_initialized),
            "fully_initialized": diagnostic_fully_initialized,
            "armed_layer_count": armed_layer_count,
            "promotion_count": promotion_count,
            "mapped_page_visit_count": mapped_page_visit_count,
            "all_latches_clear": (
                armed_layer_count == 0 if armed_layer_count is not None else None
            ),
            "request_reset_observed": (
                fully_initialized
                and raw_page_count == 0
                and free_count == slot_count
                and fatal == 0
            ),
        }
    return result


def _forced_raw_phase_result(
    armed_state: dict[str, Any],
    completed_state: dict[str, Any],
) -> dict[str, Any]:
    """Build per-phase proof from arm and request-completion snapshots."""
    armed_layers = {
        layer["name"]: layer
        for layer in armed_state["layers"]
        if layer.get("forced_raw_diagnostic_enabled", False)
    }
    completed_layers = {
        layer["name"]: layer
        for layer in completed_state["layers"]
        if layer.get("forced_raw_diagnostic_enabled", False)
    }
    if not armed_layers or armed_layers.keys() != completed_layers.keys():
        raise RuntimeError("ByteV2 forced raw diagnostic layer set changed")

    layer_results = []
    for name, armed_layer in armed_layers.items():
        completed_layer = completed_layers[name]
        promotion_delta = (
            completed_layer["promotion_count"] - armed_layer["promotion_count"]
        )
        mapped_visit_delta = (
            completed_layer["mapped_page_visit_count"]
            - armed_layer["mapped_page_visit_count"]
        )
        reset_observed = (
            completed_layer["raw_page_count"] == 0
            and completed_layer["free_count"] == completed_layer["slot_count"]
            and completed_layer["fatal"] == 0
        )
        layer_results.append(
            {
                "name": name,
                "armed": armed_layer["forced_raw_latch"] == 1,
                "consumed": completed_layer["forced_raw_latch"] == 0,
                "promotion_delta": promotion_delta,
                "mapped_page_visit_delta": mapped_visit_delta,
                "request_reset_observed": reset_observed,
            }
        )
    verified = all(
        layer["armed"]
        and layer["consumed"]
        and layer["promotion_delta"] == 1
        and layer["mapped_page_visit_delta"] >= 1
        and layer["request_reset_observed"]
        for layer in layer_results
    )
    return {
        "layer_count": len(layer_results),
        "promotion_count": sum(layer["promotion_delta"] for layer in layer_results),
        "mapped_page_visit_count": sum(
            layer["mapped_page_visit_delta"] for layer in layer_results
        ),
        "request_reset_observed": all(
            layer["request_reset_observed"] for layer in layer_results
        ),
        "verified": verified,
        "layers": layer_results,
    }


def _collect_kv_cache_plan(llm) -> dict[str, Any]:
    """Collect the worker KV-cache allocation plan without mutating it."""
    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    driver_worker = getattr(executor, "driver_worker", None)
    model_runner = getattr(driver_worker, "model_runner", None)
    config = getattr(model_runner, "kv_cache_config", None)
    if config is None:
        return {"available": False}

    compact_tensor_bytes = sum(
        int(tensor.size) for tensor in getattr(config, "kv_cache_tensors", ())
    )
    sidecar_bytes = int(getattr(config, "byte_v2_raw_fallback_sidecar_bytes", 0))
    workspace_bytes = int(getattr(config, "byte_v2_raw_staging_workspace_bytes", 0))
    return {
        "available": True,
        "num_blocks": int(config.num_blocks),
        "compact_tensor_bytes": compact_tensor_bytes,
        "raw_fallback_sidecar_bytes": sidecar_bytes,
        "raw_staging_workspace_bytes": workspace_bytes,
        "total_planned_bytes": (compact_tensor_bytes + sidecar_bytes + workspace_bytes),
        "num_byte_v2_layers": int(getattr(config, "num_byte_v2_layers", 0)),
        "raw_fallback_slots_per_layer": int(
            getattr(config, "byte_v2_raw_fallback_slots", 0)
        ),
        "raw_staging_slots": int(getattr(config, "byte_v2_raw_staging_slots", 0)),
    }


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


def _make_sharegpt_prompt_token_ids(
    model: str,
    rows_dir: str,
    *,
    num_requests: int,
    max_prompt_len: int,
    seed: int,
    target_lens: list[int] | None = None,
) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    candidates: list[tuple[int, list[int]]] = []
    for path in sorted(Path(rows_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("rows", []):
            row = item.get("row", item)
            conversations = row.get("conversations", [])
            if not conversations:
                continue
            prompt_text = conversations[0].get("value", "")
            if not prompt_text:
                continue
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if 16 <= len(prompt_ids) <= max_prompt_len:
                candidates.append((int(item.get("row_idx", 0)), prompt_ids))
    if len(candidates) < num_requests:
        raise ValueError(
            f"only {len(candidates)} valid ShareGPT prompts for "
            f"batch_size={num_requests} and max_prompt_len={max_prompt_len}"
        )
    if target_lens is not None:
        if len(target_lens) != num_requests:
            raise ValueError(
                "the number of --sharegpt-target-lens values must equal --batch-size"
            )
        selected: list[list[int]] = []
        remaining = candidates.copy()
        for target_len in target_lens:
            best_idx = min(
                range(len(remaining)),
                key=lambda idx: (
                    abs(len(remaining[idx][1]) - target_len),
                    remaining[idx][0],
                ),
            )
            _, prompt_ids = remaining.pop(best_idx)
            selected.append(prompt_ids)
        return selected
    random.Random(seed).shuffle(candidates)
    return [prompt_ids for _, prompt_ids in candidates[:num_requests]]


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
        enable_prefix_caching=not args.disable_prefix_caching,
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
            "byte_v2.op.byte_v2_speculative_verify_ragged_q4",
            "byte_v2.speculative_verify_ragged_q4",
            f"byte_v2.paged_decode.q{verify_query_len}",
        }
        query_prefixes = (
            "byte_v2.forward.q",
            "byte_v2.paged_decode.q",
            "byte_v2.prefill_from_cache.q",
            "byte_v2.ragged_candidate.q",
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
    if args.diagnose_outlier_pool:
        os.environ["BYTE_V2_NATIVE_RAW_STAGING_UPDATE"] = "0"
    if args.diagnose_forced_raw_lifecycle:
        os.environ["BYTE_V2_FA2_HYBRID_RAW_FALLBACK"] = "1"
        os.environ["BYTE_V2_TEST_FORCE_RAW_PROMOTION"] = "1"

    import torch

    from vllm import SamplingParams, TokensPrompt

    assert args.backend in ("byte_v2", "flash_attn")
    assert args.worker_spec_tokens is not None
    spec_tokens = args.worker_spec_tokens
    collector = ProfileCollector()
    pool_diagnostic = None
    if args.backend == "byte_v2" and args.diagnose_outlier_pool:
        collector.install(args.backend)
        pool_diagnostic = _install_outlier_pool_diagnostic(collector)

    init_start = time.perf_counter()
    llm = _build_llm(args, args.backend, spec_tokens)
    torch.accelerator.synchronize()
    init_seconds = time.perf_counter() - init_start
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    cache_fill_params = SamplingParams(max_tokens=1, temperature=0.0)

    try:
        for context_index, context_len in enumerate(args.context_lens):
            if args.sharegpt_rows_dir:
                prompt_token_ids = _make_sharegpt_prompt_token_ids(
                    args.model,
                    args.sharegpt_rows_dir,
                    num_requests=args.batch_size,
                    max_prompt_len=context_len,
                    seed=args.sharegpt_seed + context_index,
                    target_lens=args.sharegpt_target_lens,
                )
            else:
                prompt_token_ids = [
                    _make_prompt_token_ids(
                        args.model,
                        context_len,
                        offset=context_index * 11 + request_idx * 7,
                    )
                    for request_idx in range(args.batch_size)
                ]
            prompts = [
                TokensPrompt(prompt_token_ids=token_ids)
                for token_ids in prompt_token_ids
            ]

            if args.disable_prefix_caching:
                warmup_seconds, _ = _run_generate(
                    llm,
                    prompts,
                    cache_fill_params,
                )
            else:
                warmup_seconds = 0.0
                for prompt in prompts:
                    prompt_seconds, _ = _run_generate(
                        llm,
                        [prompt],
                        cache_fill_params,
                    )
                    warmup_seconds += prompt_seconds

            forced_raw_measured_armed = None
            if args.diagnose_forced_raw_lifecycle:
                _control_forced_raw_diagnostic(llm, arm=False)
                forced_raw_measured_armed = _control_forced_raw_diagnostic(
                    llm,
                    arm=True,
                )
            before = _metric_snapshot(llm)
            measured_seconds, token_ids = _run_generate(llm, prompts, sampling_params)
            measured_metrics = _metric_diff(before, _metric_snapshot(llm))
            forced_raw_measured_reset = (
                _reset_forced_raw_pages_through_runner(llm)
                if args.diagnose_forced_raw_lifecycle
                else None
            )
            forced_raw_measured_completed = (
                _collect_hybrid_raw_fallback_state(llm)
                if args.diagnose_forced_raw_lifecycle
                else None
            )

            profile_instrumentation_installed = False
            if not args.diagnose_outlier_pool:
                collector.install(args.backend)
                profile_instrumentation_installed = True
            try:
                forced_raw_profile_armed = (
                    _control_forced_raw_diagnostic(llm, arm=True)
                    if args.diagnose_forced_raw_lifecycle
                    else None
                )
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
            finally:
                if profile_instrumentation_installed:
                    collector.restore()
            profile_metrics = _metric_diff(
                profile_metrics_before,
                _metric_snapshot(llm),
            )
            forced_raw_profile_reset = (
                _reset_forced_raw_pages_through_runner(llm)
                if args.diagnose_forced_raw_lifecycle
                else None
            )
            forced_raw_profile_completed = (
                _collect_hybrid_raw_fallback_state(llm)
                if args.diagnose_forced_raw_lifecycle
                else None
            )
            forced_raw_lifecycle = None
            if args.diagnose_forced_raw_lifecycle:
                assert forced_raw_measured_armed is not None
                assert forced_raw_measured_completed is not None
                assert forced_raw_profile_armed is not None
                assert forced_raw_profile_completed is not None
                measured_proof = _forced_raw_phase_result(
                    forced_raw_measured_armed,
                    forced_raw_measured_completed,
                )
                profile_proof = _forced_raw_phase_result(
                    forced_raw_profile_armed,
                    forced_raw_profile_completed,
                )
                measured_proof["reset"] = forced_raw_measured_reset
                profile_proof["reset"] = forced_raw_profile_reset
                forced_raw_lifecycle = {
                    "enabled": True,
                    "mode": "test_force_next_valid_hybrid_q1_raw_page",
                    "performance_valid_for_tps": False,
                    "measured": measured_proof,
                    "profile": profile_proof,
                    "verified": (
                        measured_proof["verified"] and profile_proof["verified"]
                    ),
                }
                if not forced_raw_lifecycle["verified"]:
                    raise RuntimeError(
                        "ByteV2 forced raw lifecycle diagnostic did not "
                        f"verify: {forced_raw_lifecycle}"
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
                "prompt_source": (
                    "sharegpt" if args.sharegpt_rows_dir else "synthetic_repeat"
                ),
                "prompt_lens": [len(ids) for ids in prompt_token_ids],
                "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
                "enforce_eager": args.enforce_eager,
                "compile_size_specialization": (args.compile_size_specialization),
                "performance_valid_for_tps": (not args.diagnose_forced_raw_lifecycle),
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
                "byte_v2_prefill_patterns": collector.byte_v2_pattern_result(),
                "byte_v2_prefill_shapes": collector.byte_v2_shape_result(),
                "step_profile": step_profile,
                "outlier_pool_diagnostic": (
                    pool_diagnostic.result() if pool_diagnostic is not None else None
                ),
                "kv_cache_plan": _collect_kv_cache_plan(llm),
                "hybrid_raw_fallback_state": (
                    forced_raw_profile_completed
                    if args.diagnose_forced_raw_lifecycle
                    else _collect_hybrid_raw_fallback_state(llm)
                    if args.collect_hybrid_state
                    else None
                ),
                "forced_raw_lifecycle_diagnostic": forced_raw_lifecycle,
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
        env["BYTE_V2_TEST_FORCE_RAW_PROMOTION"] = (
            "1" if args.diagnose_forced_raw_lifecycle else "0"
        )
        if args.diagnose_forced_raw_lifecycle:
            env["BYTE_V2_FA2_HYBRID_RAW_FALLBACK"] = "1"
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
    if args.sharegpt_rows_dir:
        cmd.extend(["--sharegpt-rows-dir", args.sharegpt_rows_dir])
        cmd.extend(["--sharegpt-seed", str(args.sharegpt_seed)])
        if args.sharegpt_target_lens:
            cmd.extend(
                [
                    "--sharegpt-target-lens",
                    *[str(value) for value in args.sharegpt_target_lens],
                ]
            )
    if args.disable_prefix_caching:
        cmd.append("--disable-prefix-caching")
    if args.diagnose_outlier_pool:
        cmd.append("--diagnose-outlier-pool")
    if args.collect_hybrid_state:
        cmd.append("--collect-hybrid-state")
    if args.diagnose_forced_raw_lifecycle:
        cmd.append("--diagnose-forced-raw-lifecycle")
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
            f"accept={result['spec_metrics']['acceptance_rate']} "
            f"performance_valid={result['performance_valid_for_tps']}",
            flush=True,
        )
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    print(
        "backend,spec_tokens,context_len,batch_size,seconds,tokens_per_second,"
        "performance_valid,acceptance_rate,mean_acceptance_length,reference_match"
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
            f"{row['performance_valid_for_tps']},"
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
