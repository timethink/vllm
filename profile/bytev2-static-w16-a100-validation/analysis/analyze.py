# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate correctness-gated A100 Raw-FA2/Static-W16 measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-repetitions", type=int, default=3)
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--allow-fallback-pages", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.expected_repetitions <= 0:
        parser.error("--expected-repetitions must be positive")
    if not args.results.is_dir():
        parser.error(f"results directory does not exist: {args.results}")
    for output in (args.output_json, args.output_md):
        if output.exists() and not args.force:
            parser.error(f"refusing to overwrite {output}; pass --force")
    return args


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _bitwise_ok(record: dict[str, Any]) -> bool:
    if record.get("wire_bitwise") is not True:
        return False
    correctness = record.get("correctness", {})
    eager = correctness.get("eager", {})
    if eager.get("output_bit_mismatch") or eager.get("lse_bit_mismatch"):
        return False
    graph = correctness.get("graph", {})
    if set(graph) != {"raw_vs_eager_raw", "w16_vs_eager_raw"}:
        return False
    return all(
        not result.get("output_bit_mismatch") and not result.get("lse_bit_mismatch")
        for result in graph.values()
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "values_us": values,
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "min_us": min(values),
        "max_us": max(values),
        "range_us": max(values) - min(values),
    }


def _percent(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def _speedup(candidate: float, reference: float) -> float:
    return 100.0 * (reference / candidate - 1.0)


def _validate_record(
    path: Path,
    record: dict[str, Any],
    *,
    allow_non_a100: bool,
    allow_fallback_pages: bool,
) -> None:
    if record.get("schema_version") != 1:
        raise RuntimeError(f"unsupported schema in {path}")
    if record.get("experiment") != "bytev2_static_w16_a100_raw_fa2_ab":
        raise RuntimeError(f"wrong experiment in {path}")
    if record.get("mode") != "benchmark":
        raise RuntimeError(f"non-benchmark record in {path}")
    if not _bitwise_ok(record):
        raise RuntimeError(f"bitwise correctness gate failed in {path}")
    if record.get("profile_environment"):
        raise RuntimeError(f"profile controls were active in {path}")
    device = record.get("device", {})
    if not allow_non_a100 and device.get("is_a100_sm80") is not True:
        raise RuntimeError(f"formal result is not from an A100 SM80: {path}")
    fallback_pages = record.get("pack_stats", {}).get("raw_fallback_pages")
    if not allow_fallback_pages and fallback_pages != 0:
        raise RuntimeError(f"raw fallback pages are present in {path}")
    for key in ("seq_len", "num_splits", "repetition"):
        if not isinstance(record.get(key), int) or record[key] <= 0:
            raise RuntimeError(f"invalid {key} in {path}")
    timings = record.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError(f"timings are missing in {path}")
    for backend in ("raw_graph", "w16_graph"):
        if timings.get(backend, {}).get("median_us", 0) <= 0:
            raise RuntimeError(f"invalid {backend} timing in {path}")


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# A100 Raw-FA2 versus Static-W16 results",
        "",
        "Negative `W16 vs Raw` means W16 is faster. Every included run passed ",
        "the wire, eager output/LSE, and CUDA Graph output/LSE bitwise gates.",
        "",
        "## Same-split results",
        "",
        "| Context | Splits | Raw graph (us) | W16 graph (us) | "
        "W16 vs Raw | Repetitions |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seq_len, seq_result in summary["sequence_results"].items():
        for split, row in seq_result["splits"].items():
            lines.append(
                f"| {int(seq_len):,} | {split} | "
                f"{row['raw_graph']['median_us']:.4f} | "
                f"{row['w16_graph']['median_us']:.4f} | "
                f"{row['w16_vs_raw_percent']:+.3f}% | "
                f"{row['repetitions']} |"
            )
    lines.extend(
        [
            "",
            "## Independently tuned best split",
            "",
            "| Context | Raw best split/time | W16 best split/time | "
            "W16 vs tuned Raw | W16 speedup |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seq_len, seq_result in summary["sequence_results"].items():
        best = seq_result["best"]
        lines.append(
            f"| {int(seq_len):,} | {best['raw_split']} / "
            f"{best['raw_us']:.4f} us | {best['w16_split']} / "
            f"{best['w16_us']:.4f} us | "
            f"{best['w16_vs_raw_percent']:+.3f}% | "
            f"{best['w16_speedup_percent']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- GPU: `{summary['provenance']['gpu']}`",
            f"- Compute capability: `{summary['provenance']['compute_capability']}`",
            f"- FA2 extension SHA-256: `{summary['provenance']['extension_sha256']}`",
            f"- Input: `{summary['provenance']['input_kind']}` / "
            f"`{summary['provenance']['input_sha256']}`",
            f"- Files checked: `{summary['files_checked']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Validate all records, aggregate process medians, and write reports."""
    args = parse_args()
    paths = sorted(args.results.rglob("*.json"))
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        record = _load(path)
        if record.get("experiment") != "bytev2_static_w16_a100_raw_fa2_ab":
            continue
        _validate_record(
            path,
            record,
            allow_non_a100=args.allow_non_a100,
            allow_fallback_pages=args.allow_fallback_pages,
        )
        records.append((path, record))
    if not records:
        raise RuntimeError(f"no benchmark records found under {args.results}")

    provenance_sets: dict[str, set[str]] = defaultdict(set)
    grouped: dict[tuple[int, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(
        list
    )
    for path, record in records:
        grouped[(record["seq_len"], record["num_splits"])].append((path, record))
        provenance_sets["gpu"].add(record["device"]["name"])
        provenance_sets["compute_capability"].add(
            record["device"]["compute_capability"]
        )
        provenance_sets["extension_sha256"].add(record["fa2_extension"]["sha256"])
        provenance_sets["input_kind"].add(record["input"]["kind"])
        provenance_sets["input_sha256"].add(record["input"]["sha256"])
    for name, values in provenance_sets.items():
        if len(values) != 1:
            raise RuntimeError(f"mixed {name} values: {sorted(values)}")

    sequence_rows: dict[int, dict[int, Any]] = defaultdict(dict)
    for (seq_len, split), items in sorted(grouped.items()):
        repetitions = sorted(record["repetition"] for _, record in items)
        expected = list(range(1, args.expected_repetitions + 1))
        if repetitions != expected:
            raise RuntimeError(
                f"expected repetitions {expected} for seq={seq_len}, split={split}; "
                f"got {repetitions}"
            )
        raw = [record["timings"]["raw_graph"]["median_us"] for _, record in items]
        w16 = [record["timings"]["w16_graph"]["median_us"] for _, record in items]
        raw_median = statistics.median(raw)
        w16_median = statistics.median(w16)
        sequence_rows[seq_len][split] = {
            "raw_graph": _distribution(raw),
            "w16_graph": _distribution(w16),
            "w16_vs_raw_percent": _percent(w16_median, raw_median),
            "w16_speedup_percent": _speedup(w16_median, raw_median),
            "paired_ratio_values": [
                w16_value / raw_value
                for raw_value, w16_value in zip(raw, w16, strict=True)
            ],
            "repetitions": len(items),
            "files": [str(path) for path, _ in items],
        }

    sequence_results: dict[str, Any] = {}
    for seq_len, rows in sorted(sequence_rows.items()):
        raw_split, raw_row = min(
            rows.items(), key=lambda item: item[1]["raw_graph"]["median_us"]
        )
        w16_split, w16_row = min(
            rows.items(), key=lambda item: item[1]["w16_graph"]["median_us"]
        )
        raw_us = raw_row["raw_graph"]["median_us"]
        w16_us = w16_row["w16_graph"]["median_us"]
        sequence_results[str(seq_len)] = {
            "splits": {str(split): row for split, row in sorted(rows.items())},
            "best": {
                "raw_split": raw_split,
                "raw_us": raw_us,
                "w16_split": w16_split,
                "w16_us": w16_us,
                "w16_vs_raw_percent": _percent(w16_us, raw_us),
                "w16_speedup_percent": _speedup(w16_us, raw_us),
            },
        }

    provenance = {name: next(iter(values)) for name, values in provenance_sets.items()}
    summary = {
        "schema_version": 1,
        "experiment": "bytev2_static_w16_a100_aggregate",
        "files_checked": len(records),
        "expected_repetitions": args.expected_repetitions,
        "provenance": provenance,
        "sequence_results": sequence_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
