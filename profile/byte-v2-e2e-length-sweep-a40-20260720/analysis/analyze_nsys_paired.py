#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze interleaved ByteV2/raw FA2 Nsys traces."""

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


def classify(name: str) -> str | None:
    if "flash_fwd_splitkv_byte_v2_kernel<" in name:
        return "byte"
    if "flash_fwd_splitkv_kernel<" in name:
        return "raw"
    if "flash_fwd_splitkv_combine_kernel<" in name:
        return "combine"
    return None


def load_samples(path: Path) -> dict[str, list[dict[str, Any]]]:
    events = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            kind = classify(row["Name"])
            if kind is None:
                continue
            events.append(
                {
                    "kind": kind,
                    "start_ns": int(row["Start (ns)"]),
                    "duration_ns": int(row["Duration (ns)"]),
                    "grid": [int(row[key]) for key in ("GrdX", "GrdY", "GrdZ")],
                }
            )
    events.sort(key=lambda event: event["start_ns"])

    samples: dict[str, list[dict[str, Any]]] = {"byte": [], "raw": []}
    pending_main = None
    for event in events:
        if event["kind"] != "combine":
            if pending_main is not None:
                raise RuntimeError(f"{path}: main kernel was not followed by combine")
            pending_main = event
            continue
        if pending_main is None:
            raise RuntimeError(f"{path}: combine kernel has no preceding main")
        main_end_ns = pending_main["start_ns"] + pending_main["duration_ns"]
        combine_end_ns = event["start_ns"] + event["duration_ns"]
        if event["start_ns"] < main_end_ns:
            raise RuntimeError(f"{path}: main and combine kernels overlap")
        samples[pending_main["kind"]].append(
            {
                "main_us": pending_main["duration_ns"] / 1000.0,
                "combine_us": event["duration_ns"] / 1000.0,
                "active_us": (pending_main["duration_ns"] + event["duration_ns"])
                / 1000.0,
                "launch_gap_us": (event["start_ns"] - main_end_ns) / 1000.0,
                "envelope_us": (combine_end_ns - pending_main["start_ns"]) / 1000.0,
                "main_grid": pending_main["grid"],
                "combine_grid": event["grid"],
                "start_ns": pending_main["start_ns"],
            }
        )
        pending_main = None
    if pending_main is not None:
        raise RuntimeError(f"{path}: final main kernel has no combine")
    return samples


def summarize_backend(samples: list[dict[str, Any]]) -> dict[str, Any]:
    steady = samples[1:]
    result: dict[str, Any] = {
        "instances": len(samples),
        "steady_instances": len(steady),
        "main_grid": samples[-1]["main_grid"],
        "combine_grid": samples[-1]["combine_grid"],
    }
    for metric in (
        "main_us",
        "combine_us",
        "active_us",
        "launch_gap_us",
        "envelope_us",
    ):
        result[f"median_{metric}"] = statistics.median(
            sample[metric] for sample in samples
        )
        result[f"steady_mean_{metric}"] = statistics.fmean(
            sample[metric] for sample in steady
        )
    return result


def summarize_comparison(
    byte_samples: list[dict[str, Any]],
    raw_samples: list[dict[str, Any]],
) -> dict[str, float]:
    if len(byte_samples) != len(raw_samples):
        raise RuntimeError("ByteV2/raw instance counts differ")
    comparisons: dict[str, float] = {}
    for byte, raw in zip(byte_samples, raw_samples):
        if byte["start_ns"] >= raw["start_ns"]:
            raise RuntimeError("expected ByteV2 to precede raw in each iteration")
    for metric in ("main_us", "combine_us", "active_us", "envelope_us"):
        steady_pairs = zip(byte_samples[1:], raw_samples[1:])
        gaps = []
        gap_percents = []
        for byte, raw in steady_pairs:
            gaps.append(byte[metric] - raw[metric])
            gap_percents.append((byte[metric] / raw[metric] - 1.0) * 100.0)
        comparisons[f"paired_median_{metric}_gap"] = statistics.median(gaps)
        comparisons[f"paired_mean_{metric}_gap"] = statistics.fmean(gaps)
        comparisons[f"paired_median_{metric}_gap_percent"] = statistics.median(
            gap_percents
        )
        comparisons[f"paired_mean_{metric}_gap_percent"] = statistics.fmean(
            gap_percents
        )
    return comparisons


def main() -> None:
    args = parse_args()
    analysis_dir = args.run_dir / "analysis"
    summary: dict[str, Any] = {
        "trace_tool": "Nsight Systems 2025.5.2",
        "iterations_per_trace": args.expected_instances,
        "order_within_iteration": ["byte", "raw"],
        "first_iteration_excluded_from_paired_statistics": True,
        "contexts": {},
    }
    csv_rows = []
    for seq_len in args.seq_lens:
        trace_path = analysis_dir / f"nsys_pair_seq{seq_len}_trace_cuda_gpu_trace.csv"
        samples = load_samples(trace_path)
        for backend in ("byte", "raw"):
            if len(samples[backend]) != args.expected_instances:
                raise RuntimeError(
                    f"{trace_path}: expected {args.expected_instances} {backend} "
                    f"instances, found {len(samples[backend])}"
                )
        byte_summary = summarize_backend(samples["byte"])
        raw_summary = summarize_backend(samples["raw"])
        comparison = summarize_comparison(samples["byte"], samples["raw"])
        summary["contexts"][str(seq_len)] = {
            "byte": byte_summary,
            "raw": raw_summary,
            "comparison": comparison,
        }
        csv_rows.append(
            {
                "seq_len": seq_len,
                "splits": byte_summary["main_grid"][1],
                "byte_median_main_us": byte_summary["median_main_us"],
                "raw_median_main_us": raw_summary["median_main_us"],
                "byte_median_combine_us": byte_summary["median_combine_us"],
                "raw_median_combine_us": raw_summary["median_combine_us"],
                "byte_median_envelope_us": byte_summary["median_envelope_us"],
                "raw_median_envelope_us": raw_summary["median_envelope_us"],
                **comparison,
            }
        )

    json_path = analysis_dir / "nsys_paired_attention_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = analysis_dir / "nsys_paired_attention_summary.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
