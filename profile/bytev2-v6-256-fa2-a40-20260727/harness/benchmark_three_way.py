# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark raw, ByteV2, and fixed-page SplitZip through the same FA2 path."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from splitzip_fixed_page import pack_splitzip_fixed_pages  # noqa: E402

from scripts import byte_v2_fa2_oracle as oracle  # noqa: E402
from vllm.v1.attention.backends.byte_v2_ops import (  # noqa: E402
    byte_v2_reshape_and_cache,
)

BACKENDS = ("raw", "bytev2", "splitzip")
ORDERS = tuple(itertools.permutations(BACKENDS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=198)
    parser.add_argument(
        "--calls-per-sample",
        type=int,
        default=1,
        help="Wrap this many identical calls in each CUDA-event interval.",
    )
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument(
        "--capture",
        type=Path,
        default=(
            REPO_ROOT / "profile/bytev2-splitzip-codec-a40-20260726/raw/"
            "llama_layer0_cal2048_eval4096.pt"
        ),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seq_len <= 0 or args.seq_len % 16:
        parser.error("--seq-len must be a positive multiple of 16")
    if args.query_len <= 0 or args.query_len > args.seq_len:
        parser.error("--query-len must be in [1, seq-len]")
    if not 0 <= args.num_splits <= 128:
        parser.error("--num-splits must be in [0, 128]")
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.calls_per_sample <= 0
        or args.repetition <= 0
    ):
        parser.error("warmup, iterations, and repetition are invalid")
    return args


def _repeat_capture(
    tensor: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    repeats = (tokens + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats, 1, 1))[:tokens].contiguous()


def _replace_with_capture(
    tensors: dict[str, torch.Tensor | float | int],
    capture_path: Path,
    seq_len: int,
) -> None:
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    key_cpu = _repeat_capture(capture["evaluation_key"], seq_len)
    value_cpu = _repeat_capture(capture["evaluation_value"], seq_len)
    if (
        key_cpu.dtype != torch.bfloat16
        or value_cpu.dtype != torch.bfloat16
        or key_cpu.shape[1:] != (8, 128)
        or value_cpu.shape != key_cpu.shape
    ):
        raise ValueError("capture must contain BF16 evaluation K/V [N, 8, 128]")

    key = tensors["key"]
    value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    key.copy_(key_cpu.to(device=key.device).view_as(key))
    value.copy_(value_cpu.to(device=value.device).view_as(value))
    byte_cache.zero_()
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=key.device)
    byte_v2_reshape_and_cache(
        key.view(seq_len, 8, 128),
        value.view(seq_len, 8, 128),
        byte_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    torch.accelerator.synchronize()


def _make_calls(
    tensors: dict[str, torch.Tensor | float | int],
    splitzip_cache: torch.Tensor,
    *,
    query_len: int,
    num_splits: int,
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    query = tensors["query"]
    max_seq_len = tensors["max_seq_len"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(max_seq_len, int)
    outputs = {name: torch.empty_like(query) for name in BACKENDS}
    common: tuple[Any, ...] = (
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        query_len,
        max_seq_len,
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

    def raw() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.varlen_fwd(
            query,
            tensors["key"],
            tensors["value"],
            outputs["raw"],
            *common,
        )

    def bytev2() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.byte_v2_varlen_fwd(
            query,
            tensors["byte_cache"],
            None,
            outputs["bytev2"],
            *common,
        )

    def splitzip() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.splitzip_varlen_fwd(
            query,
            splitzip_cache,
            outputs["splitzip"],
            *common,
        )

    return {"raw": raw, "bytev2": bytev2, "splitzip": splitzip}


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min_us": ordered[0],
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "max_us": ordered[-1],
    }


def _correctness(
    candidate: tuple[torch.Tensor, torch.Tensor],
    raw: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, int | float]:
    candidate_out, candidate_lse = candidate
    raw_out, raw_lse = raw
    return {
        "output_bit_mismatch": int(
            (candidate_out.view(torch.int16) != raw_out.view(torch.int16)).sum()
        ),
        "lse_bit_mismatch": int(
            (candidate_lse.view(torch.int32) != raw_lse.view(torch.int32)).sum()
        ),
        "output_max_abs": float((candidate_out.float() - raw_out.float()).abs().max()),
        "lse_max_abs": float((candidate_lse - raw_lse).abs().max()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not oracle.fa2_oracle_ops_are_available():
        raise RuntimeError("raw and ByteV2 FA2 ops are unavailable")
    if not hasattr(torch.ops._vllm_fa2_C, "splitzip_varlen_fwd"):
        raise RuntimeError("SplitZip FA2 reader op is unavailable")

    tensors = oracle.make_inputs(
        (args.seq_len,),
        query_len=args.query_len,
        permute_pages=False,
    )
    if not args.synthetic:
        _replace_with_capture(tensors, args.capture, args.seq_len)
    key = tensors["key"]
    value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    splitzip_cache, pack_stats = pack_splitzip_fixed_pages(key, value)
    torch.accelerator.synchronize()

    overflow_offset = 4
    byte_overflow_pages = int(
        (
            byte_cache[:, overflow_offset : overflow_offset + 4]
            .contiguous()
            .view(torch.int32)
            != 0
        ).sum()
    )
    if byte_overflow_pages:
        raise RuntimeError(f"ByteV2 writer overflowed {byte_overflow_pages} pages")

    calls = _make_calls(
        tensors,
        splitzip_cache,
        query_len=args.query_len,
        num_splits=args.num_splits,
    )
    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for iteration in range(args.warmup):
        for backend in ORDERS[iteration % len(ORDERS)]:
            results[backend] = calls[backend]()
    torch.accelerator.synchronize()

    correctness = {
        backend: _correctness(results[backend], results["raw"])
        for backend in ("bytev2", "splitzip")
    }
    for backend, result in correctness.items():
        if result["output_bit_mismatch"] or result["lse_bit_mismatch"]:
            raise RuntimeError(f"{backend} failed the raw FA2 bitwise gate: {result}")

    events: dict[str, list[tuple[torch.Event, torch.Event]]] = {
        name: [] for name in BACKENDS
    }
    for iteration in range(args.iterations):
        for backend in ORDERS[iteration % len(ORDERS)]:
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            for _ in range(args.calls_per_sample):
                results[backend] = calls[backend]()
            end.record()
            events[backend].append((start, end))
    torch.accelerator.synchronize()

    timings = {
        backend: _summary(
            [
                start.elapsed_time(end) * 1000.0 / args.calls_per_sample
                for start, end in events[backend]
            ]
        )
        for backend in BACKENDS
    }
    raw_median = float(timings["raw"]["median_us"])
    ratios = {
        backend: {
            "latency_over_raw": float(timings[backend]["median_us"]) / raw_median,
            "latency_gap_percent": (
                float(timings[backend]["median_us"]) / raw_median - 1.0
            )
            * 100.0,
        }
        for backend in ("bytev2", "splitzip")
    }
    raw_bytes = key.numel() * key.element_size() * 2
    properties = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    )
    payload = {
        "seq_len": args.seq_len,
        "query_len": args.query_len,
        "num_splits": args.num_splits,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "calls_per_sample": args.calls_per_sample,
        "repetition": args.repetition,
        "input": "synthetic" if args.synthetic else str(args.capture),
        "gpu": properties.name,
        "order": "all six backend permutations, repeated",
        "timings": timings,
        "ratios": ratios,
        "correctness": correctness,
        "pack_stats": pack_stats.to_dict(),
        "storage": {
            "raw_bytes": raw_bytes,
            "bytev2_fixed_page_bytes": byte_cache.numel(),
            "splitzip_fixed_page_bytes": splitzip_cache.numel(),
            "bytev2_saving_percent": (1.0 - byte_cache.numel() / raw_bytes) * 100.0,
            "splitzip_saving_percent": (1.0 - splitzip_cache.numel() / raw_bytes)
            * 100.0,
        },
    }
    serialized = json.dumps(payload, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
