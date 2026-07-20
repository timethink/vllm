#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a compact A40 NCU comparison from archived key metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METRICS = {
    "duration_us": ("gpu__time_duration.sum", 1e-3),
    "registers_per_thread": ("launch__registers_per_thread", 1.0),
    "shared_memory_kib": ("launch__shared_mem_per_block", 1.0 / 1024.0),
    "shared_memory_block_limit": ("launch__occupancy_limit_shared_mem", 1.0),
    "register_block_limit": ("launch__occupancy_limit_registers", 1.0),
    "waves_per_sm": ("launch__waves_per_multiprocessor", 1.0),
    "dram_read_mib": ("dram__bytes_read.sum", 1.0 / (1024.0 * 1024.0)),
    "dram_read_gbps": ("dram__bytes_read.sum.per_second", 1e-9),
    "l1_hit_percent": ("l1tex__t_sector_hit_rate.pct", 1.0),
    "l2_hit_percent": ("lts__t_sector_hit_rate.pct", 1.0),
    "instructions_per_sm": ("smsp__inst_executed.avg", 1.0),
    "global_load_instructions": ("smsp__sass_inst_executed_op_global_ld.sum", 1.0),
    "shared_load_instructions": ("smsp__sass_inst_executed_op_shared_ld.sum", 1.0),
    "shared_store_instructions": ("smsp__sass_inst_executed_op_shared_st.sum", 1.0),
    "eligible_warps_per_cycle": ("smsp__warps_eligible.avg.per_cycle_active", 1.0),
    "issue_active_percent": (
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        1.0,
    ),
    "long_scoreboard_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        1.0,
    ),
    "long_scoreboard_samples": (
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        1.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())
    result = {}
    for output_name, (metric_name, scale) in METRICS.items():
        value = raw[metric_name]
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{path}: {metric_name} is unavailable")
        result[output_name] = float(value) * scale
    return result


def main() -> None:
    args = parse_args()
    analysis_dir = args.run_dir / "analysis"
    backends = {
        backend: load_metrics(analysis_dir / f"metrics_key_{backend}_seq1024.json")
        for backend in ("byte", "raw")
    }
    comparison: dict[str, Any] = {}
    for name in METRICS:
        byte_value = backends["byte"][name]
        raw_value = backends["raw"][name]
        comparison[f"{name}_gap"] = byte_value - raw_value
        comparison[f"{name}_gap_percent"] = (
            (byte_value / raw_value - 1.0) * 100.0 if raw_value else None
        )
    summary = {
        "gpu": "NVIDIA A40 (SM86)",
        "seq_len": 1024,
        "query_len": 1,
        "byte": backends["byte"],
        "raw": backends["raw"],
        "comparison": comparison,
        "interpretation_note": (
            "NCU replay duration is diagnostic; uninstrumented E2E and paired "
            "Nsys medians are the performance decision metrics."
        ),
    }
    json_path = analysis_dir / "ncu_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = analysis_dir / "ncu_comparison_summary.csv"
    rows = [{"backend": backend, **metrics} for backend, metrics in backends.items()]
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
