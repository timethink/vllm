#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize steady decode CUDA Graph replays from Nsys SQLite exports."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=8)
    return parser.parse_args()


def classify_kernel(name: str, backend: str) -> str:
    if "flash_fwd_splitkv_byte_v2_kernel<" in name:
        return "attention_main"
    if "flash_fwd_splitkv_kernel<" in name:
        return "attention_main"
    if "flash_fwd_splitkv_combine_kernel<" in name:
        return "attention_combine"
    if "byte_v2_hydrate_append_single_token_raw_staging_kernel<" in name:
        return "update_stage"
    if "byte_v2_clear_page_metadata_from_staging_kernel" in name:
        return "update_clear"
    if "byte_v2_commit_raw_staging_to_cache_kernel<" in name:
        return "update_commit"
    if "byte_v2_release_raw_staging_and_update_flags_kernel" in name:
        return "update_release"
    if "reshape_and_cache_flash_kernel<" in name:
        return "update_raw"
    return f"other_{backend}"


def discover_decode_graph(connection: sqlite3.Connection, backend: str) -> int:
    main_pattern = (
        "%flash_fwd_splitkv_byte_v2_kernel%"
        if backend == "byte"
        else "%flash_fwd_splitkv_kernel%"
    )
    row = connection.execute(
        """
        SELECT k.graphId, COUNT(*) AS instances
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.demangledName
        WHERE k.graphId IS NOT NULL AND s.value LIKE ?
        GROUP BY k.graphId
        ORDER BY instances DESC
        LIMIT 1
        """,
        (main_pattern,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"no decode graph found for {backend}")
    return int(row[0])


def load_replays(
    path: Path,
    backend: str,
) -> tuple[int, list[dict[str, Any]]]:
    connection = sqlite3.connect(path)
    try:
        graph_id = discover_decode_graph(connection, backend)
        rows = connection.execute(
            """
            SELECT k.correlationId, k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
            WHERE k.graphId = ?
            ORDER BY k.start
            """,
            (graph_id,),
        ).fetchall()
    finally:
        connection.close()

    grouped: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for correlation_id, start_ns, end_ns, name in rows:
        grouped[int(correlation_id)].append((int(start_ns), int(end_ns), name))

    replays = []
    for correlation_id, kernels in grouped.items():
        kernels.sort()
        by_class: dict[str, list[float]] = defaultdict(list)
        for start_ns, end_ns, name in kernels:
            by_class[classify_kernel(name, backend)].append(
                (end_ns - start_ns) / 1000.0
            )
        first_start_ns = kernels[0][0]
        last_end_ns = max(kernel[1] for kernel in kernels)
        replays.append(
            {
                "correlation_id": correlation_id,
                "start_ns": first_start_ns,
                "kernel_count": len(kernels),
                "graph_envelope_us": (last_end_ns - first_start_ns) / 1000.0,
                "all_kernel_active_us": sum(
                    (end_ns - start_ns) / 1000.0 for start_ns, end_ns, _ in kernels
                ),
                "classes": dict(by_class),
            }
        )
    replays.sort(key=lambda replay: replay["start_ns"])
    return graph_id, replays


def summarize_backend(path: Path, backend: str) -> dict[str, Any]:
    graph_id, replays = load_replays(path, backend)
    if len(replays) < 2:
        raise RuntimeError(f"{path}: expected validation plus steady replays")
    # The first replay is graph validation during engine initialization. The two
    # following clusters are the seven measured and seven profiler decode steps.
    steady = replays[1:]
    expected_classes = (
        (
            "attention_main",
            "attention_combine",
            "update_stage",
            "update_clear",
            "update_commit",
            "update_release",
        )
        if backend == "byte"
        else ("attention_main", "attention_combine", "update_raw")
    )
    for replay in steady:
        for name in expected_classes:
            instances = len(replay["classes"].get(name, []))
            if instances != 32:
                raise RuntimeError(
                    f"{path}: expected 32 {name} nodes per replay, found {instances}"
                )

    per_replay = []
    for replay in steady:
        class_totals = {name: sum(replay["classes"][name]) for name in expected_classes}
        attention_us = (
            class_totals["attention_main"] + class_totals["attention_combine"]
        )
        update_us = sum(
            value for name, value in class_totals.items() if name.startswith("update_")
        )
        relevant_us = attention_us + update_us
        per_replay.append(
            {
                "graph_envelope_us": replay["graph_envelope_us"],
                "all_kernel_active_us": replay["all_kernel_active_us"],
                "attention_us": attention_us,
                "update_us": update_us,
                "relevant_us": relevant_us,
                "other_kernel_active_us": (
                    replay["all_kernel_active_us"] - relevant_us
                ),
                **{f"{name}_us": value for name, value in class_totals.items()},
            }
        )

    metrics = list(per_replay[0])
    summary: dict[str, Any] = {
        "graph_id": graph_id,
        "total_replays": len(replays),
        "steady_replays": len(steady),
        "validation_replays_excluded": 1,
        "kernel_count_per_replay": int(
            statistics.median(replay["kernel_count"] for replay in steady)
        ),
    }
    for metric in metrics:
        values = [replay[metric] for replay in per_replay]
        summary[f"median_{metric}"] = statistics.median(values)
        summary[f"mean_{metric}"] = statistics.fmean(values)
    summary["median_attention_per_layer_us"] = summary["median_attention_us"] / 32.0
    summary["median_update_per_layer_us"] = summary["median_update_us"] / 32.0
    return summary


def main() -> None:
    args = parse_args()
    reports_dir = args.run_dir / "reports"
    analysis_dir = args.run_dir / "analysis"
    backends = {}
    for backend in ("byte", "raw"):
        sqlite_path = (
            reports_dir
            / f"nsys_engine_{backend}_seq{args.seq_len}_out{args.output_tokens}.sqlite"
        )
        backends[backend] = summarize_backend(sqlite_path, backend)

    comparison = {}
    for metric in (
        "graph_envelope_us",
        "all_kernel_active_us",
        "attention_us",
        "update_us",
        "relevant_us",
        "other_kernel_active_us",
    ):
        byte_value = backends["byte"][f"median_{metric}"]
        raw_value = backends["raw"][f"median_{metric}"]
        comparison[f"median_{metric}_gap"] = byte_value - raw_value
        comparison[f"median_{metric}_gap_percent"] = (
            byte_value / raw_value - 1.0
        ) * 100.0

    summary = {
        "seq_len": args.seq_len,
        "output_tokens": args.output_tokens,
        "layers": 32,
        "trace_tool": "Nsight Systems 2025.5.2",
        "byte": backends["byte"],
        "raw": backends["raw"],
        "comparison": comparison,
    }
    json_path = analysis_dir / "nsys_engine_graph_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = analysis_dir / "nsys_engine_graph_summary.csv"
    rows = [{"backend": backend, **values} for backend, values in backends.items()]
    fieldnames = ["backend"] + sorted(
        {name for row in rows for name in row if name != "backend"}
    )
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
