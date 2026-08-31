# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and benchmark stock Raw-FA2 against canonical Static-W16."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from static_w16_fixed_page import (
    DEFAULT_K_EXPONENT_BASE,
    DEFAULT_V_EXPONENT_BASE,
    FIXED_PAGE_BYTES,
    RAW_PAGE_BYTES,
    StaticW16PackedPages,
    pack_static_w16_fixed_pages,
    unpack_static_w16_fixed_pages,
)

BACKENDS = ("raw_eager", "w16_eager", "raw_graph", "w16_graph")
ORDERS = tuple(itertools.permutations(BACKENDS))
PROFILE_ENV_VARS = (
    "BYTE_V2_FA2_PROFILE_Q16_GQA_SPLIT",
    "BYTE_V2_FA2_PROFILE_RAW_KV_SMEM_ALIAS",
    "BYTE_V2_FA2_PROFILE_STATIC_W16_ACTIVE_QUERY_WARP",
    "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_7P1C",
    "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_OVERLAP",
    "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_TRACE",
    "BYTE_V2_FA2_PROFILE_STATIC_W16_SMEM_PADDING_BYTES",
    "BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa2-so", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-splits", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--calls-per-sample", type=int, default=20)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--input",
        choices=("synthetic-compact", "synthetic-escapes", "capture"),
        default="synthetic-compact",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--capture-layer", type=int)
    parser.add_argument("--w16-k-base", type=int, default=DEFAULT_K_EXPONENT_BASE)
    parser.add_argument("--w16-v-base", type=int, default=DEFAULT_V_EXPONENT_BASE)
    parser.add_argument("--expected-raw-fallback-pages", type=int, default=0)
    parser.add_argument("--permute-pages", action="store_true")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--allow-profile-env", action="store_true")
    parser.add_argument("--profile-backend", choices=("raw", "w16"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if not 1 <= args.num_splits <= 128:
        parser.error("--num-splits must be in [1, 128]")
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.calls_per_sample <= 0
        or args.repetition <= 0
    ):
        parser.error("warmup, iterations, calls-per-sample, or repetition invalid")
    if not 0 <= args.w16_k_base <= 240:
        parser.error("--w16-k-base must be in [0, 240]")
    if not 0 <= args.w16_v_base <= 240:
        parser.error("--w16-v-base must be in [0, 240]")
    if args.expected_raw_fallback_pages < 0:
        parser.error("--expected-raw-fallback-pages must be non-negative")
    if args.capture_layer is not None and args.capture_layer < 0:
        parser.error("--capture-layer must be non-negative")
    if args.input == "capture" and args.capture is None:
        parser.error("--input capture requires --capture")
    if args.input != "capture" and args.capture is not None:
        parser.error("--capture is valid only with --input capture")
    if args.input == "synthetic-escapes" and (
        args.w16_k_base < 2
        or args.w16_k_base > 237
        or args.w16_v_base < 2
        or args.w16_v_base > 237
    ):
        parser.error("synthetic escapes require K/V bases in [2, 237]")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fa2_extension(path: Path) -> Path:
    """Load the requested extension without importing an installed FA2 binary."""
    import vllm

    del vllm
    resolved = path.resolve(strict=True)
    module_name = "vllm.vllm_flash_attn._vllm_fa2_C"
    if module_name in sys.modules:
        raise RuntimeError(f"{module_name} was loaded before --fa2-so")
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FA2 extension {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    loaded = Path(module.__file__).resolve(strict=True)
    if loaded != resolved:
        raise RuntimeError(f"loaded {loaded}, expected {resolved}")
    for op_name in ("varlen_fwd", "static_w16_canonical_varlen_fwd"):
        if not hasattr(torch.ops._vllm_fa2_C, op_name):
            raise RuntimeError(f"required FA2 op is missing: {op_name}")
    return loaded


def _check_stock_environment(allow_profile_env: bool) -> dict[str, str]:
    values = {name: os.environ[name] for name in PROFILE_ENV_VARS if name in os.environ}
    if values and not allow_profile_env:
        rendered = ", ".join(f"{name}={value!r}" for name, value in values.items())
        raise RuntimeError(
            f"stock comparison requires every profile control to be unset: {rendered}"
        )
    return values


def _device_info(allow_non_a100: bool) -> dict[str, Any]:
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.accelerator.current_device_index()
    properties = torch.cuda.get_device_properties(device)
    is_a100 = "A100" in properties.name.upper() and (
        properties.major,
        properties.minor,
    ) == (8, 0)
    if not is_a100 and not allow_non_a100:
        raise RuntimeError(
            "this protocol requires an NVIDIA A100 (SM 8.0); got "
            f"{properties.name} SM {properties.major}.{properties.minor}. "
            "Use --allow-non-a100 only for a non-formal smoke test."
        )
    return {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "is_a100_sm80": is_a100,
    }


def _physical_page_order(num_pages: int, permute: bool) -> list[int]:
    if not permute:
        return list(range(num_pages))
    order: list[int] = []
    low = 0
    high = num_pages - 1
    while low <= high:
        order.append(high)
        high -= 1
        if low <= high:
            order.append(low)
            low += 1
    return order


def _synthetic_bits(
    count: int,
    *,
    base: int,
    escapes_per_page: int,
) -> torch.Tensor:
    index = torch.arange(count, dtype=torch.int32, device="cuda")
    local = index % (16 * 8 * 128)
    exponent = base + (index % 16)
    if escapes_per_page:
        escape_mask = local < escapes_per_page
        escape_exponent = torch.where(local % 2 == 0, base - 2, base + 18)
        exponent = torch.where(escape_mask, escape_exponent, exponent)
    sign = (index & 1) << 15
    mantissa = (index * 29 + 17) % 128
    bits = sign | (exponent << 7) | mantissa
    return bits.to(torch.int16).view(torch.bfloat16)


def _capture_tokens(
    path: Path,
    *,
    layer: int | None,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    capture = torch.load(resolved, map_location="cpu", weights_only=False)
    key = capture["evaluation_key"]
    value = capture["evaluation_value"]
    if key.ndim == 4:
        if layer is None or not 0 <= layer < key.shape[0]:
            raise ValueError("a layered capture requires a valid --capture-layer")
        key = key[layer]
        value = value[layer]
    elif layer is not None:
        raise ValueError("--capture-layer requires rank-4 captured K/V")
    if (
        key.ndim != 3
        or key.dtype != torch.bfloat16
        or tuple(key.shape[1:]) != (8, 128)
        or value.shape != key.shape
        or value.dtype != key.dtype
    ):
        raise ValueError("capture K/V must be BF16 [tokens, 8, 128]")
    repeats = (seq_len + key.shape[0] - 1) // key.shape[0]
    key = key.repeat((repeats, 1, 1))[:seq_len].contiguous().cuda()
    value = value.repeat((repeats, 1, 1))[:seq_len].contiguous().cuda()
    return (
        key,
        value,
        {
            "kind": "capture",
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "layer": layer,
        },
    )


def _input_tokens(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if args.input == "capture":
        assert args.capture is not None
        return _capture_tokens(
            args.capture,
            layer=args.capture_layer,
            seq_len=args.seq_len,
        )
    escapes = 32 if args.input == "synthetic-escapes" else 0
    count = args.seq_len * 8 * 128
    key = _synthetic_bits(
        count,
        base=args.w16_k_base,
        escapes_per_page=escapes,
    ).reshape(args.seq_len, 8, 128)
    value = _synthetic_bits(
        count,
        base=args.w16_v_base,
        escapes_per_page=escapes,
    ).reshape(args.seq_len, 8, 128)
    descriptor = {
        "kind": args.input,
        "generator": "bf16-bit-pattern-v1",
        "escapes_per_side_per_page": escapes,
        "k_exponent_base": args.w16_k_base,
        "v_exponent_base": args.w16_v_base,
    }
    descriptor["sha256"] = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True).encode()
    ).hexdigest()
    return key, value, descriptor


def _make_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    logical_key, logical_value, source = _input_tokens(args)
    num_pages = (args.seq_len + 15) // 16
    physical_order = _physical_page_order(num_pages, args.permute_pages)
    raw_key = torch.zeros(
        (num_pages, 16, 8, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_value = torch.zeros_like(raw_key)
    valid_rows = torch.zeros(num_pages, dtype=torch.int32, device="cuda")
    for logical_page, physical_page in enumerate(physical_order):
        start = logical_page * 16
        stop = min(start + 16, args.seq_len)
        rows = stop - start
        raw_key[physical_page, :rows].copy_(logical_key[start:stop])
        raw_value[physical_page, :rows].copy_(logical_value[start:stop])
        valid_rows[physical_page] = rows
    block_table = torch.tensor(
        physical_order,
        dtype=torch.int32,
        device="cuda",
    ).view(1, num_pages)
    query_index = torch.arange(32 * 128, dtype=torch.int32, device="cuda")
    query_bits = (
        ((query_index & 1) << 15)
        | ((124 + query_index % 4) << 7)
        | ((query_index * 13 + 5) % 128)
    )
    query = query_bits.to(torch.int16).view(torch.bfloat16).reshape(1, 32, 128)
    tensors: dict[str, Any] = {
        "query": query,
        "key": raw_key,
        "value": raw_value,
        "block_table": block_table,
        "seq_lens": torch.tensor([args.seq_len], dtype=torch.int32, device="cuda"),
        "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "cu_seqlens_k": torch.zeros(2, dtype=torch.int32, device="cuda"),
        "scale": 128**-0.5,
        "valid_rows": valid_rows,
    }
    source["permuted_pages"] = args.permute_pages
    return tensors, source


def _common_args(tensors: dict[str, Any], num_splits: int) -> tuple[Any, ...]:
    return (
        tensors["cu_seqlens_q"],
        tensors["cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        1,
        int(tensors["seq_lens"].max().item()),
        0.0,
        tensors["scale"],
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        num_splits,
        None,
    )


def _make_calls(
    tensors: dict[str, Any],
    packed: StaticW16PackedPages,
    *,
    num_splits: int,
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    if packed.raw_sidecar is None or packed.page_to_raw_slot is None:
        raise RuntimeError("Static-W16 call requires fallback tensors")
    query = tensors["query"]
    raw_output = torch.empty_like(query)
    w16_output = torch.empty_like(query)
    common = _common_args(tensors, num_splits)

    def raw() -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.ops._vllm_fa2_C.varlen_fwd(
            query,
            tensors["key"],
            tensors["value"],
            raw_output,
            *common,
        )
        return result[0], result[1]

    def w16() -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.ops._vllm_fa2_C.static_w16_canonical_varlen_fwd(
            query,
            packed.cache,
            packed.raw_sidecar,
            packed.page_to_raw_slot,
            w16_output,
            *common,
        )
        return result[0], result[1]

    return {"raw": raw, "w16": w16}


def _correctness(
    candidate: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, int | float]:
    candidate_out, candidate_lse = candidate
    reference_out, reference_lse = reference
    return {
        "output_bit_mismatch": int(
            (candidate_out.view(torch.int16) != reference_out.view(torch.int16))
            .sum()
            .item()
        ),
        "lse_bit_mismatch": int(
            (candidate_lse.view(torch.int32) != reference_lse.view(torch.int32))
            .sum()
            .item()
        ),
        "output_max_abs": float(
            (candidate_out.float() - reference_out.float()).abs().max().item()
        ),
        "lse_max_abs": float(
            (candidate_lse.float() - reference_lse.float()).abs().max().item()
        ),
    }


def _require_bitwise(name: str, result: dict[str, int | float]) -> None:
    if result["output_bit_mismatch"] or result["lse_bit_mismatch"]:
        raise RuntimeError(f"{name} failed the Raw-FA2 bitwise gate: {result}")


def _capture_graph(
    call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, torch.Tensor]]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    return graph, result


def _summary(samples: list[float]) -> dict[str, int | float]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min_us": ordered[0],
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "max_us": ordered[-1],
    }


def _git_provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    result: dict[str, Any] = {"repo": str(repo)}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["commit"] = None
        result["dirty"] = None
    return result


def _profile_one(
    calls: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]],
    backend: str,
    warmup: int,
) -> None:
    for _ in range(warmup):
        calls[backend]()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    calls[backend]()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _time_calls(
    calls: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]],
    *,
    warmup: int,
    iterations: int,
    calls_per_sample: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = calls["raw"]()
    eager_correctness = _correctness(calls["w16"](), reference)
    torch.accelerator.synchronize()
    _require_bitwise("eager W16", eager_correctness)

    raw_graph, raw_graph_result = _capture_graph(calls["raw"])
    w16_graph, w16_graph_result = _capture_graph(calls["w16"])
    graph_calls: dict[str, Callable[[], Any]] = {
        "raw_graph": raw_graph.replay,
        "w16_graph": w16_graph.replay,
    }
    graph_calls["raw_graph"]()
    graph_calls["w16_graph"]()
    torch.accelerator.synchronize()
    graph_correctness = {
        "raw_vs_eager_raw": _correctness(raw_graph_result, reference),
        "w16_vs_eager_raw": _correctness(w16_graph_result, reference),
    }
    for name, result in graph_correctness.items():
        _require_bitwise(name, result)

    timed_calls: dict[str, Callable[[], Any]] = {
        "raw_eager": calls["raw"],
        "w16_eager": calls["w16"],
        **graph_calls,
    }
    for iteration in range(warmup):
        for backend in ORDERS[iteration % len(ORDERS)]:
            timed_calls[backend]()
    torch.accelerator.synchronize()

    events: dict[str, list[tuple[torch.Event, torch.Event]]] = {
        name: [] for name in BACKENDS
    }
    for iteration in range(iterations):
        order = ORDERS[iteration % len(ORDERS)]
        for backend in order:
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_sample):
                timed_calls[backend]()
            end.record()
            events[backend].append((start, end))
    torch.accelerator.synchronize()
    samples = {
        backend: [
            start.elapsed_time(end) * 1000.0 / calls_per_sample
            for start, end in backend_events
        ]
        for backend, backend_events in events.items()
    }
    timings = {name: _summary(values) for name, values in samples.items()}
    return (
        timings,
        samples,
        {
            "eager": eager_correctness,
            "graph": graph_correctness,
        },
    )


def main() -> None:
    """Run the correctness-gated benchmark or one profiler launch."""
    args = parse_args()
    profile_env = _check_stock_environment(args.allow_profile_env)
    extension = _load_fa2_extension(args.fa2_so)
    device = _device_info(args.allow_non_a100)
    tensors, input_source = _make_inputs(args)
    packed = pack_static_w16_fixed_pages(
        tensors["key"],
        tensors["value"],
        valid_rows=tensors["valid_rows"],
        k_exponent_base=args.w16_k_base,
        v_exponent_base=args.w16_v_base,
        canonical_metadata=True,
    )
    if packed.stats.raw_fallback_pages != args.expected_raw_fallback_pages:
        raise RuntimeError(
            "unexpected raw fallback page count: expected "
            f"{args.expected_raw_fallback_pages}, got "
            f"{packed.stats.raw_fallback_pages}"
        )
    decoded_key, decoded_value = unpack_static_w16_fixed_pages(packed)
    wire_bitwise = torch.equal(
        decoded_key.view(torch.int16), tensors["key"].view(torch.int16)
    ) and torch.equal(
        decoded_value.view(torch.int16), tensors["value"].view(torch.int16)
    )
    del decoded_key, decoded_value
    if not wire_bitwise:
        raise RuntimeError("canonical Static-W16 wire failed the BF16 bitwise gate")

    calls = _make_calls(tensors, packed, num_splits=args.num_splits)
    raw_reference = calls["raw"]()
    eager_correctness = _correctness(calls["w16"](), raw_reference)
    torch.accelerator.synchronize()
    _require_bitwise("eager W16", eager_correctness)

    timings: dict[str, Any] | None = None
    samples: dict[str, Any] | None = None
    correctness: dict[str, Any] = {"eager": eager_correctness}
    if args.profile_backend is not None:
        _profile_one(calls, args.profile_backend, args.warmup)
    else:
        timings, samples, correctness = _time_calls(
            calls,
            warmup=args.warmup,
            iterations=args.iterations,
            calls_per_sample=args.calls_per_sample,
        )

    raw_bytes = tensors["key"].numel() * tensors["key"].element_size() * 2
    payload = {
        "schema_version": 1,
        "experiment": "bytev2_static_w16_a100_raw_fa2_ab",
        "mode": "profile" if args.profile_backend else "benchmark",
        "profile_backend": args.profile_backend,
        "seq_len": args.seq_len,
        "query_len": 1,
        "batch_size": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_splits": args.num_splits,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "calls_per_sample": args.calls_per_sample,
        "repetition": args.repetition,
        "order": "all 24 eager/graph permutations, repeated",
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fa2_extension": {
            "path": str(extension),
            "sha256": _sha256_file(extension),
        },
        "git": _git_provenance(),
        "input": input_source,
        "profile_environment": profile_env,
        "w16_k_exponent_base": args.w16_k_base,
        "w16_v_exponent_base": args.w16_v_base,
        "expected_raw_fallback_pages": args.expected_raw_fallback_pages,
        "wire_bitwise": wire_bitwise,
        "correctness": correctness,
        "pack_stats": packed.stats.to_dict(),
        "storage": {
            "raw_kv_bytes": raw_bytes,
            "w16_compact_bytes": packed.cache.numel(),
            "raw_bytes_per_page": RAW_PAGE_BYTES,
            "w16_bytes_per_page": FIXED_PAGE_BYTES,
            "compact_reduction_percent": 100.0
            * (1.0 - packed.cache.numel() / raw_bytes),
        },
        "timings": timings,
        "samples_us": samples,
    }
    if timings is not None:
        raw_us = timings["raw_graph"]["median_us"]
        w16_us = timings["w16_graph"]["median_us"]
        payload["ratios"] = {
            "graph_w16_over_raw": w16_us / raw_us,
            "graph_w16_vs_raw_percent": 100.0 * (w16_us / raw_us - 1.0),
            "eager_w16_over_raw": (
                timings["w16_eager"]["median_us"] / timings["raw_eager"]["median_us"]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
