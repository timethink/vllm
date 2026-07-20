#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize ByteV2/raw FA2 main and combine kernels from Nsys traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384],
    )
    parser.add_argument("--expected-instances", type=int, default=200)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def load_events(path: Path, backend: str) -> tuple[list[dict[str, Any]], ...]:
    main_pattern = (
        "flash_fwd_splitkv_byte_v2_kernel<"
        if backend == "byte"
        else "flash_fwd_splitkv_kernel<"
    )
    main_events = []
    combine_events = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            name = row["Name"]
            if main_pattern in name or "flash_fwd_splitkv_combine_kernel<" in name:
                event = {
                    "start_ns": int(row["Start (ns)"]),
                    "duration_ns": int(row["Duration (ns)"]),
                    "grid": [int(row[key]) for key in ("GrdX", "GrdY", "GrdZ")],
                }
                if main_pattern in name:
                    main_events.append(event)
                else:
                    combine_events.append(event)
    return (
        sorted(main_events, key=lambda event: event["start_ns"]),
        sorted(combine_events, key=lambda event: event["start_ns"]),
    )


def summarize_trace(
    path: Path,
    backend: str,
    expected_instances: int,
) -> dict[str, Any]:
    main_events, combine_events = load_events(path, backend)
    if len(main_events) != expected_instances:
        raise RuntimeError(
            f"{path}: expected {expected_instances} main kernels, "
            f"found {len(main_events)}"
        )
    if len(combine_events) != expected_instances:
        raise RuntimeError(
            f"{path}: expected {expected_instances} combine kernels, "
            f"found {len(combine_events)}"
        )

    pairs = []
    for index, (main, combine) in enumerate(zip(main_events, combine_events)):
        main_end_ns = main["start_ns"] + main["duration_ns"]
        combine_end_ns = combine["start_ns"] + combine["duration_ns"]
        if combine["start_ns"] < main_end_ns:
            raise RuntimeError(f"{path}: pair {index} overlaps unexpectedly")
        if index + 1 < len(main_events):
            next_main_start_ns = main_events[index + 1]["start_ns"]
            if combine_end_ns > next_main_start_ns:
                raise RuntimeError(f"{path}: pair {index} crosses next main kernel")
        pairs.append(
            {
                "main_us": main["duration_ns"] / 1000.0,
                "combine_us": combine["duration_ns"] / 1000.0,
                "active_us": (main["duration_ns"] + combine["duration_ns"]) / 1000.0,
                "launch_gap_us": (combine["start_ns"] - main_end_ns) / 1000.0,
                "envelope_us": (combine_end_ns - main["start_ns"]) / 1000.0,
            }
        )

    # The first call can include lazy CUDA runtime work. Medians are robust to it;
    # steady means explicitly omit it.
    steady_pairs = pairs[1:]
    result: dict[str, Any] = {
        "instances": len(pairs),
        "steady_instances": len(steady_pairs),
        "main_grid": main_events[-1]["grid"],
        "combine_grid": combine_events[-1]["grid"],
    }
    for name in ("main_us", "combine_us", "active_us", "launch_gap_us", "envelope_us"):
        all_values = [pair[name] for pair in pairs]
        steady_values = [pair[name] for pair in steady_pairs]
        result[f"median_{name}"] = statistics.median(all_values)
        result[f"steady_mean_{name}"] = statistics.fmean(steady_values)
        result[f"steady_p05_{name}"] = percentile(steady_values, 0.05)
        result[f"steady_p95_{name}"] = percentile(steady_values, 0.95)
    return result


def main() -> None:
    args = parse_args()
    analysis_dir = args.run_dir / "analysis"
    summary: dict[str, Any] = {
        "trace_tool": "Nsight Systems 2025.5.2",
        "iterations_per_trace": args.expected_instances,
        "first_iteration_excluded_from_steady_means": True,
        "contexts": {},
    }
    csv_rows = []
    for seq_len in args.seq_lens:
        backend_summaries = {}
        for backend in ("byte", "raw"):
            trace_path = (
                analysis_dir / f"nsys_{backend}_seq{seq_len}_trace_cuda_gpu_trace.csv"
            )
            backend_summaries[backend] = summarize_trace(
                trace_path,
                backend,
                args.expected_instances,
            )

        byte = backend_summaries["byte"]
        raw = backend_summaries["raw"]
        comparison = {}
        for name in ("main_us", "combine_us", "active_us", "envelope_us"):
            byte_value = byte[f"median_{name}"]
            raw_value = raw[f"median_{name}"]
            comparison[f"median_{name}_gap"] = byte_value - raw_value
            comparison[f"median_{name}_gap_percent"] = (
                byte_value / raw_value - 1.0
            ) * 100.0
        summary["contexts"][str(seq_len)] = {
            "byte": byte,
            "raw": raw,
            "comparison": comparison,
        }
        for backend, values in backend_summaries.items():
            csv_rows.append(
                {
                    "seq_len": seq_len,
                    "backend": backend,
                    **values,
                }
            )

    json_path = analysis_dir / "nsys_attention_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = analysis_dir / "nsys_attention_summary.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
