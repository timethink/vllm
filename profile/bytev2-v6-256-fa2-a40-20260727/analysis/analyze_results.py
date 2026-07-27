#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate the ByteV2 V6-256 A40 experiment without plotting dependencies."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
RUN_DIR = ANALYSIS_DIR.parent
PROFILE_DIR = RUN_DIR.parent
REPORTS_DIR = RUN_DIR / "reports"
POOL_REPORT = (
    PROFILE_DIR / "bytev2-v6-pool-demand-a40-20260727" / "reports" / "pool_demand.json"
)
EXPECTED_SEQ_LENS = (4096, 16384, 65536)
BACKENDS = ("raw", "bytev2", "splitzip")
BACKEND_LABELS = {
    "raw": "Raw FA2",
    "bytev2": "ByteV2 V6-256",
    "splitzip": "SplitZip V5 envelope",
}
COLORS = {
    "raw": "#4c78a8",
    "bytev2": "#e45756",
    "splitzip": "#54a24b",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: Input JSON path.

    Returns:
        Parsed JSON object.
    """
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, data: Any) -> None:
    """Write deterministic, human-readable JSON.

    Args:
        path: Output JSON path.
        data: JSON-serializable value.
    """
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")


def median(values: list[float]) -> float:
    """Return a float median and reject empty input.

    Args:
        values: Numeric observations.

    Returns:
        Median value.
    """
    if not values:
        raise ValueError("cannot take the median of an empty list")
    return float(statistics.median(values))


def relative_percent(value: float, reference: float) -> float:
    """Compute a signed percentage change.

    Args:
        value: New value.
        reference: Baseline value.

    Returns:
        ``(value / reference - 1) * 100``.
    """
    return (value / reference - 1.0) * 100.0


def collect_event_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect paired CUDA-event measurements by sequence length.

    Returns:
        CSV rows and a nested JSON summary.
    """
    paths = sorted(REPORTS_DIR.glob("event_s*_split*_rep*.json"))
    records = [load_json(path) | {"_path": path} for path in paths]
    found_seq_lens = {int(record["seq_len"]) for record in records}
    if found_seq_lens != set(EXPECTED_SEQ_LENS):
        raise ValueError(
            f"event sequence lengths {sorted(found_seq_lens)} do not match "
            f"{list(EXPECTED_SEQ_LENS)}"
        )

    by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seq[int(record["seq_len"])].append(record)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for seq_len in EXPECTED_SEQ_LENS:
        repetitions = sorted(
            by_seq[seq_len],
            key=lambda item: int(item["repetition"]),
        )
        raw_medians = [
            float(record["timings"]["raw"]["median_us"]) for record in repetitions
        ]
        seq_summary: dict[str, Any] = {
            "num_splits": int(repetitions[0]["num_splits"]),
            "repetitions": len(repetitions),
            "samples_per_repetition": int(repetitions[0]["iterations"]),
            "calls_per_sample": int(repetitions[0]["calls_per_sample"]),
            "backends": {},
        }
        seq_summary["total_calls_per_repetition"] = (
            seq_summary["samples_per_repetition"] * seq_summary["calls_per_sample"]
        )
        for backend in BACKENDS:
            rep_medians = [
                float(record["timings"][backend]["median_us"]) for record in repetitions
            ]
            paired_gaps = [
                relative_percent(value, raw)
                for value, raw in zip(rep_medians, raw_medians, strict=True)
            ]
            correctness = [
                record["correctness"].get(backend)
                for record in repetitions
                if backend != "raw"
            ]
            backend_summary = {
                "label": BACKEND_LABELS[backend],
                "median_of_repetition_medians_us": median(rep_medians),
                "mean_of_repetition_medians_us": float(statistics.fmean(rep_medians)),
                "min_repetition_median_us": min(rep_medians),
                "max_repetition_median_us": max(rep_medians),
                "median_paired_gap_vs_raw_percent": median(paired_gaps),
                "min_paired_gap_vs_raw_percent": min(paired_gaps),
                "max_paired_gap_vs_raw_percent": max(paired_gaps),
            }
            if correctness:
                backend_summary["all_output_bitwise_equal"] = all(
                    item["output_bit_mismatch"] == 0 and item["lse_bit_mismatch"] == 0
                    for item in correctness
                )
            seq_summary["backends"][backend] = backend_summary
            rows.append(
                {
                    "seq_len": seq_len,
                    "num_splits": seq_summary["num_splits"],
                    "backend": backend,
                    "label": BACKEND_LABELS[backend],
                    "repetitions": len(rep_medians),
                    "samples_per_repetition": seq_summary["samples_per_repetition"],
                    "calls_per_sample": seq_summary["calls_per_sample"],
                    "total_calls_per_repetition": seq_summary[
                        "total_calls_per_repetition"
                    ],
                    "median_of_repetition_medians_us": backend_summary[
                        "median_of_repetition_medians_us"
                    ],
                    "mean_of_repetition_medians_us": backend_summary[
                        "mean_of_repetition_medians_us"
                    ],
                    "min_repetition_median_us": min(rep_medians),
                    "max_repetition_median_us": max(rep_medians),
                    "median_paired_gap_vs_raw_percent": median(paired_gaps),
                }
            )
        summary[str(seq_len)] = seq_summary

    return rows, summary


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file.

    Args:
        path: JSONL input path.

    Returns:
        Parsed objects, excluding blank lines.
    """
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def token_digest(token_ids: Any) -> str:
    """Hash token IDs using a canonical JSON representation.

    Args:
        token_ids: Nested token ID sequence.

    Returns:
        SHA-256 hexadecimal digest.
    """
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_e2e_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect production ABBA E2E results and planner capacities.

    Returns:
        Per-run CSV rows and an aggregate JSON summary.
    """
    rows = []
    trace_objects_skipped = 0
    for path in sorted((REPORTS_DIR / "e2e_production").glob("*.jsonl")):
        for record in read_json_lines(path):
            if not record.get("request_metrics") or "backend" not in record:
                trace_objects_skipped += 1
                continue
            backend = "raw" if record["backend"] == "flash_attn" else "bytev2"
            request = record["request_metrics"][0]
            output_tokens = int(record["output_tokens"])
            decode_intervals = max(output_tokens - 1, 1)
            hybrid = record.get("hybrid_raw_fallback_state") or {}
            rows.append(
                {
                    "file": path.name,
                    "abba_position": path.name.split("_r1_")[1][0].upper(),
                    "backend": backend,
                    "label": BACKEND_LABELS[backend],
                    "context_len": int(record["context_len"]),
                    "output_tokens": output_tokens,
                    "ttft_ms": float(request["first_token_latency"]) * 1000,
                    "decode_seconds": float(request["decode_seconds"]),
                    "tpot_ms": (
                        float(request["decode_seconds"]) / decode_intervals * 1000
                    ),
                    "engine_e2e_seconds": float(request["engine_e2e_seconds"]),
                    "capacity_tokens": int(record["kv_cache_plan"]["capacity_tokens"]),
                    "num_blocks": int(record["kv_cache_plan"]["num_blocks"]),
                    "performance_valid": bool(record["performance_valid_for_tps"]),
                    "preemptions": int(
                        record["measured_scheduler_counters"]["num_preemptions"]
                    ),
                    "token_ids_sha256": token_digest(record["token_ids"]),
                    "hybrid_fatal": hybrid.get("fatal"),
                    "hybrid_raw_page_count": hybrid.get("raw_page_count"),
                }
            )

    rows.sort(key=lambda row: row["abba_position"])
    if [row["backend"] for row in rows] != [
        "raw",
        "bytev2",
        "bytev2",
        "raw",
    ]:
        raise ValueError("expected raw/ByteV2/ByteV2/raw ABBA E2E order")

    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    aggregate: dict[str, Any] = {}
    for backend in ("raw", "bytev2"):
        backend_rows = by_backend[backend]
        aggregate[backend] = {
            "label": BACKEND_LABELS[backend],
            "runs": len(backend_rows),
            "median_ttft_ms": median([float(row["ttft_ms"]) for row in backend_rows]),
            "median_tpot_ms": median([float(row["tpot_ms"]) for row in backend_rows]),
            "median_engine_e2e_seconds": median(
                [float(row["engine_e2e_seconds"]) for row in backend_rows]
            ),
            "capacity_tokens": int(
                median([float(row["capacity_tokens"]) for row in backend_rows])
            ),
            "all_performance_valid": all(
                row["performance_valid"] for row in backend_rows
            ),
            "total_preemptions": sum(int(row["preemptions"]) for row in backend_rows),
        }

    raw = aggregate["raw"]
    bytev2 = aggregate["bytev2"]
    aggregate["comparison"] = {
        "v6_tpot_gap_vs_raw_percent": relative_percent(
            bytev2["median_tpot_ms"], raw["median_tpot_ms"]
        ),
        "v6_ttft_gap_vs_raw_percent": relative_percent(
            bytev2["median_ttft_ms"], raw["median_ttft_ms"]
        ),
        "v6_capacity_gain_vs_raw_percent": relative_percent(
            bytev2["capacity_tokens"], raw["capacity_tokens"]
        ),
        "all_four_token_sequences_identical": (
            len({row["token_ids_sha256"] for row in rows}) == 1
        ),
        "abba_order": [row["backend"] for row in rows],
    }
    aggregate["trace_objects_skipped"] = trace_objects_skipped
    return rows, aggregate


def collect_pool_summary() -> dict[str, Any]:
    """Extract the pool-demand evidence relevant to V6-256.

    Returns:
        Pool-demand and fixed-format summary.
    """
    report = load_json(POOL_REPORT)
    splits = {}
    for split_name, split in report["splits"].items():
        capacity = split["full_page_capacities"]["256"]
        splits[split_name] = {
            "pages": int(split["pages"]),
            "demand": split["full_page_demand"],
            "pool_256": capacity,
            "escape_fraction": float(split["escape_fraction"]),
            "max_tile_outliers": int(split["max_tile_outliers"]),
        }
    return {
        "capture_sha256": report["capture"]["sha256"],
        "capture_model": report["capture"]["model"],
        "splits": splits,
        "formats": {
            "raw": {
                "page_bytes": 65536,
                "saving_vs_raw_percent": 0.0,
            },
            "v5_1024_format_reference": {
                "page_bytes": report["format_candidates"]["v5_1024"]["page_bytes"],
                "saving_vs_raw_percent": (
                    report["format_candidates"]["v5_1024"]["saving_vs_raw"] * 100
                ),
                "comparison_scope": (
                    "format-size reference only; not a same-binary latency A/B"
                ),
            },
            "v6_256": {
                "page_bytes": report["format_candidates"]["v6_256"]["page_bytes"],
                "saving_vs_raw_percent": (
                    report["format_candidates"]["v6_256"]["saving_vs_raw"] * 100
                ),
            },
        },
        "limitations": report["limitations"],
        "raw_fallback_still_required": True,
    }


def collect_ncu_summary() -> dict[str, Any]:
    """Extract comparable NCU metrics and derived changes.

    Returns:
        Raw/V6 metric subset and derived differences.
    """
    raw = load_json(ANALYSIS_DIR / "metrics_key_raw.json")
    v6 = load_json(ANALYSIS_DIR / "metrics_key_v6.json")
    keys = {
        "duration_ns": "gpu__time_duration.sum",
        "dram_bytes_read": "dram__bytes_read.sum",
        "gpu_compute_memory_throughput_percent": (
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        ),
        "dram_read_throughput_percent": (
            "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"
        ),
        "sm_throughput_percent": ("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "registers_per_thread": "launch__registers_per_thread",
        "dynamic_shared_bytes": "launch__shared_mem_per_block",
        "theoretical_occupancy_percent": ("sm__maximum_warps_per_active_cycle_pct"),
        "achieved_occupancy_percent": (
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        ),
        "long_scoreboard_per_issue": (
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
        ),
        "short_scoreboard_per_issue": (
            "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
        ),
    }
    raw_subset = {name: raw[key] for name, key in keys.items()}
    v6_subset = {name: v6[key] for name, key in keys.items()}
    return {
        "raw": raw_subset,
        "v6_256": v6_subset,
        "comparison": {
            "dram_read_reduction_percent": (
                1 - v6_subset["dram_bytes_read"] / raw_subset["dram_bytes_read"]
            )
            * 100,
            "ncu_replay_duration_gap_percent": relative_percent(
                v6_subset["duration_ns"], raw_subset["duration_ns"]
            ),
            "long_scoreboard_reduction_percent": (
                1
                - v6_subset["long_scoreboard_per_issue"]
                / raw_subset["long_scoreboard_per_issue"]
            )
            * 100,
        },
        "caveat": (
            "NCU replay duration is diagnostic and must not replace "
            "CUDA-event or E2E latency."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a CSV table.

    Args:
        path: Output CSV path.
        rows: Rows with a common schema.
    """
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def svg_text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: str = "normal",
    fill: str = "#222",
) -> str:
    """Create an escaped SVG text element.

    Args:
        x: Horizontal coordinate.
        y: Vertical coordinate.
        content: Text content.
        size: Font size in pixels.
        anchor: SVG text anchor.
        weight: Font weight.
        fill: Text color.

    Returns:
        Serialized SVG element.
    """
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(content)}</text>"
    )


def write_latency_svg(
    path: Path,
    event_summary: dict[str, Any],
) -> None:
    """Render sequence-length versus attention latency as standalone SVG.

    Args:
        path: Output SVG path.
        event_summary: Aggregated CUDA-event results.
    """
    width, height = 960, 600
    left, right, top, bottom = 90, 925, 75, 500
    y_max = 500.0

    def sx(seq_len: int) -> float:
        minimum = math.log2(EXPECTED_SEQ_LENS[0])
        maximum = math.log2(EXPECTED_SEQ_LENS[-1])
        return left + (math.log2(seq_len) - minimum) / (maximum - minimum) * (
            right - left
        )

    def sy(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            34,
            "Q1 split-K attention latency on NVIDIA A40",
            size=21,
            anchor="middle",
            weight="bold",
        ),
        svg_text(
            width / 2,
            58,
            "point = median of independent repetition medians; whisker = range",
            size=12,
            anchor="middle",
            fill="#555",
        ),
    ]
    for value in (0, 100, 200, 300, 400, 500):
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" '
            'y2="{:.1f}" stroke="#ddd" stroke-width="1"/>'.format(y)
        )
        parts.append(svg_text(left - 12, y + 5, str(value), anchor="end", size=12))
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            'stroke="#333"/>',
            svg_text(
                24,
                (top + bottom) / 2,
                "Latency (us)",
                size=14,
                anchor="middle",
            ).replace(
                "<text ",
                (f'<text transform="rotate(-90 24 {(top + bottom) / 2:.1f})" '),
                1,
            ),
            svg_text(
                (left + right) / 2,
                555,
                "KV sequence length",
                size=14,
                anchor="middle",
            ),
        ]
    )
    for seq_len in EXPECTED_SEQ_LENS:
        x = sx(seq_len)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" '
            f'y2="{bottom + 6}" stroke="#333"/>'
        )
        parts.append(
            svg_text(
                x,
                bottom + 24,
                f"{seq_len // 1024}K",
                anchor="middle",
                size=12,
            )
        )

    marker_offsets = {"raw": -6.0, "bytev2": 0.0, "splitzip": 6.0}
    for backend in BACKENDS:
        points = []
        for seq_len in EXPECTED_SEQ_LENS:
            data = event_summary[str(seq_len)]["backends"][backend]
            points.append(
                (
                    sx(seq_len) + marker_offsets[backend],
                    sy(data["median_of_repetition_medians_us"]),
                    sy(data["min_repetition_median_us"]),
                    sy(data["max_repetition_median_us"]),
                )
            )
        path_data = " ".join(
            ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
            for index, (x, y, _, _) in enumerate(points)
        )
        color = COLORS[backend]
        parts.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y, y_min, y_max_point in points:
            top_whisker = min(y_min, y_max_point)
            bottom_whisker = max(y_min, y_max_point)
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top_whisker:.1f}" '
                    f'x2="{x:.1f}" y2="{bottom_whisker:.1f}" '
                    f'stroke="{color}" stroke-width="1.5"/>',
                    f'<line x1="{x - 4:.1f}" y1="{top_whisker:.1f}" '
                    f'x2="{x + 4:.1f}" y2="{top_whisker:.1f}" '
                    f'stroke="{color}"/>',
                    f'<line x1="{x - 4:.1f}" y1="{bottom_whisker:.1f}" '
                    f'x2="{x + 4:.1f}" y2="{bottom_whisker:.1f}" '
                    f'stroke="{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                    f'fill="{color}" stroke="white" stroke-width="1"/>',
                ]
            )
    legend_x = 585
    for index, backend in enumerate(BACKENDS):
        y = 100 + index * 25
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" '
            f'y2="{y}" stroke="{COLORS[backend]}" stroke-width="3"/>'
        )
        parts.append(
            svg_text(
                legend_x + 36,
                y + 5,
                BACKEND_LABELS[backend],
                size=12,
            )
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_capacity_svg(
    path: Path,
    pool_summary: dict[str, Any],
    e2e_summary: dict[str, Any],
) -> None:
    """Render fixed page footprint and measured planner capacity.

    Args:
        path: Output SVG path.
        pool_summary: Fixed-format sizes from the pool scan.
        e2e_summary: E2E planner capacities.
    """
    width, height = 1050, 610
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            35,
            "ByteV2 V6-256 storage and 20 GB E2E planner capacity",
            size=21,
            anchor="middle",
            weight="bold",
        ),
    ]

    panels = [
        (70, 80, 435, 410, "Fixed bytes per 16-token page"),
        (585, 80, 950, 410, "Planner KV capacity (tokens)"),
    ]
    for left, top, right, bottom, title in panels:
        parts.extend(
            [
                f'<line x1="{left}" y1="{bottom}" x2="{right}" '
                f'y2="{bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{bottom}" stroke="#333"/>',
                svg_text(
                    (left + right) / 2,
                    70,
                    title,
                    anchor="middle",
                    size=15,
                    weight="bold",
                ),
            ]
        )

    formats = pool_summary["formats"]
    page_bars = [
        ("Raw", formats["raw"]["page_bytes"], "#4c78a8"),
        (
            "V5-1024 ref.",
            formats["v5_1024_format_reference"]["page_bytes"],
            "#999999",
        ),
        ("V6-256", formats["v6_256"]["page_bytes"], "#e45756"),
    ]
    page_left, page_top, page_bottom = 70, 80, 410
    page_max = 70000.0
    for tick in (0, 20000, 40000, 60000):
        y = page_bottom - tick / page_max * (page_bottom - page_top)
        parts.append(
            f'<line x1="{page_left}" y1="{y:.1f}" x2="435" '
            f'y2="{y:.1f}" stroke="#e5e5e5"/>'
        )
        parts.append(
            svg_text(
                page_left - 8,
                y + 4,
                f"{tick // 1000}K",
                anchor="end",
                size=11,
            )
        )
    for index, (label, value, color) in enumerate(page_bars):
        x = 105 + index * 105
        bar_height = value / page_max * (page_bottom - page_top)
        y = page_bottom - bar_height
        parts.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="62" '
                f'height="{bar_height:.1f}" fill="{color}"/>',
                svg_text(
                    x + 31,
                    y - 8,
                    f"{value:,}",
                    anchor="middle",
                    size=11,
                    weight="bold",
                ),
                svg_text(
                    x + 31,
                    page_bottom + 22,
                    label,
                    anchor="middle",
                    size=11,
                ),
            ]
        )

    raw_capacity = e2e_summary["raw"]["capacity_tokens"]
    v6_capacity = e2e_summary["bytev2"]["capacity_tokens"]
    cap_bars = [
        ("Raw FA2", raw_capacity, "#4c78a8"),
        ("ByteV2 V6", v6_capacity, "#e45756"),
    ]
    cap_left, cap_top, cap_bottom = 585, 80, 410
    cap_max = 210000.0
    for tick in (0, 50000, 100000, 150000, 200000):
        y = cap_bottom - tick / cap_max * (cap_bottom - cap_top)
        parts.append(
            f'<line x1="{cap_left}" y1="{y:.1f}" x2="950" '
            f'y2="{y:.1f}" stroke="#e5e5e5"/>'
        )
        parts.append(
            svg_text(
                cap_left - 8,
                y + 4,
                f"{tick // 1000}K",
                anchor="end",
                size=11,
            )
        )
    for index, (label, value, color) in enumerate(cap_bars):
        x = 655 + index * 150
        bar_height = value / cap_max * (cap_bottom - cap_top)
        y = cap_bottom - bar_height
        parts.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="82" '
                f'height="{bar_height:.1f}" fill="{color}"/>',
                svg_text(
                    x + 41,
                    y - 8,
                    f"{value:,}",
                    anchor="middle",
                    size=12,
                    weight="bold",
                ),
                svg_text(
                    x + 41,
                    cap_bottom + 22,
                    label,
                    anchor="middle",
                    size=11,
                ),
            ]
        )
    capacity_gain = e2e_summary["comparison"]["v6_capacity_gain_vs_raw_percent"]
    parts.extend(
        [
            svg_text(
                767,
                470,
                f"Measured planner capacity gain: +{capacity_gain:.2f}%",
                anchor="middle",
                size=15,
                weight="bold",
                fill="#b33b34",
            ),
            svg_text(
                252,
                470,
                (
                    "V6 fixed-page saving: "
                    f"{formats['v6_256']['saving_vs_raw_percent']:.4f}%"
                ),
                anchor="middle",
                size=14,
                weight="bold",
            ),
            svg_text(
                width / 2,
                525,
                (
                    "V5 is a format-size reference only; no historical V5 "
                    "latency is used as a same-binary comparison."
                ),
                anchor="middle",
                size=12,
                fill="#555",
            ),
            svg_text(
                width / 2,
                550,
                (
                    "Planner capacity includes ByteV2 raw-fallback sidecar "
                    "and staging workspace; KV cache memory budget = 20 GB."
                ),
                anchor="middle",
                size=12,
                fill="#555",
            ),
            svg_text(
                width / 2,
                575,
                (
                    "Pool scan saw zero V6-256 overflow in the sampled "
                    "3,072 pages; production raw fallback remains required."
                ),
                anchor="middle",
                size=12,
                fill="#555",
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all tabular, JSON, and SVG analysis artifacts."""
    event_rows, event_summary = collect_event_results()
    e2e_rows, e2e_summary = collect_e2e_results()
    pool_summary = collect_pool_summary()
    ncu_summary = collect_ncu_summary()
    summary = {
        "schema_version": 1,
        "experiment": "ByteV2 V6-256 FA2 reader on NVIDIA A40",
        "inputs": {
            "event_glob": "reports/event_s{4096,16384,65536}_split*_rep*.json",
            "e2e_glob": "reports/e2e_production/*.jsonl",
            "pool_demand": str(POOL_REPORT.relative_to(PROFILE_DIR.parent)),
            "ncu_metrics": [
                "analysis/metrics_key_raw.json",
                "analysis/metrics_key_v6.json",
            ],
        },
        "event_latency": event_summary,
        "e2e": e2e_summary,
        "pool_demand": pool_summary,
        "ncu": ncu_summary,
        "comparison_policy": {
            "v6_vs_raw": "same-run paired data",
            "v5": (
                "format-size reference only; historical V5 latency is not "
                "used for same-binary causal claims"
            ),
        },
    }
    write_csv(ANALYSIS_DIR / "event_latency.csv", event_rows)
    write_csv(ANALYSIS_DIR / "e2e_summary.csv", e2e_rows)
    write_json(ANALYSIS_DIR / "summary.json", summary)
    write_latency_svg(
        ANALYSIS_DIR / "latency_vs_seq.svg",
        event_summary,
    )
    write_capacity_svg(
        ANALYSIS_DIR / "capacity_storage.svg",
        pool_summary,
        e2e_summary,
    )

    print(
        json.dumps(
            {
                "event_rows": len(event_rows),
                "e2e_rows": len(e2e_rows),
                "token_sequences_identical": e2e_summary["comparison"][
                    "all_four_token_sequences_identical"
                ],
                "v6_tpot_gap_vs_raw_percent": e2e_summary["comparison"][
                    "v6_tpot_gap_vs_raw_percent"
                ],
                "v6_capacity_gain_vs_raw_percent": e2e_summary["comparison"][
                    "v6_capacity_gain_vs_raw_percent"
                ],
                "ncu_dram_read_reduction_percent": ncu_summary["comparison"][
                    "dram_read_reduction_percent"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
