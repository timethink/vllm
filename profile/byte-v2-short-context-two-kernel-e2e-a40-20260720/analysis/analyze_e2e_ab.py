#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and aggregate the short-context three-mode E2E experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROUNDS = (1, 2, 3)
MODES = ("four_kernel", "two_kernel", "raw")
CONTEXTS = (128, 512, 1024)
OUTPUT_TOKENS = 256
RETENTION_GATE_RATIO = 1.005
BACKEND_BY_MODE = {
    "four_kernel": "byte_v2",
    "two_kernel": "byte_v2",
    "raw": "flash_attn",
}
LATIN_SQUARE_ORDER = {
    1: ("four_kernel", "two_kernel", "raw"),
    2: ("two_kernel", "raw", "four_kernel"),
    3: ("raw", "four_kernel", "two_kernel"),
}

Row = dict[str, Any]
RowsByContext = dict[int, Row]
LoadedRows = dict[tuple[int, str], RowsByContext]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--precheck",
        action="store_true",
        help=(
            "Validate every currently available input and report missing files "
            "without writing the final summary."
        ),
    )
    return parser.parse_args()


def expected_path(reports_dir: Path, round_index: int, mode: str) -> Path:
    return reports_dir / f"run{round_index}_{mode}.jsonl"


def _require_equal(
    row: Row,
    field: str,
    expected: Any,
    *,
    source: str,
) -> None:
    if field not in row:
        raise RuntimeError(f"{source}: missing field {field!r}")
    actual = row[field]
    matches = actual is expected if isinstance(expected, bool) else actual == expected
    if not matches:
        raise RuntimeError(f"{source}: {field}={actual!r}, expected {expected!r}")


def _validate_token_ids(row: Row, *, source: str) -> None:
    token_ids = row.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != 1:
        raise RuntimeError(f"{source}: token_ids must contain one request")
    request_tokens = token_ids[0]
    if not isinstance(request_tokens, list) or len(request_tokens) != OUTPUT_TOKENS:
        raise RuntimeError(
            f"{source}: expected exactly {OUTPUT_TOKENS} generated tokens"
        )
    if any(
        not isinstance(token, int) or isinstance(token, bool)
        for token in request_tokens
    ):
        raise RuntimeError(f"{source}: token_ids contains a non-integer token")
    if row["output_tokens"] != sum(len(tokens) for tokens in token_ids):
        raise RuntimeError(f"{source}: output_tokens disagrees with token_ids")


def _validate_row(
    row: Row,
    *,
    path: Path,
    line_number: int,
    mode: str,
) -> None:
    source = f"{path}:{line_number}"
    context_len = row.get("context_len")
    if context_len not in CONTEXTS:
        raise RuntimeError(
            f"{source}: context_len={context_len!r}, expected one of {CONTEXTS}"
        )

    expected_fields = {
        "backend": BACKEND_BY_MODE[mode],
        "spec_tokens": 0,
        "verify_query_len": 1,
        "prompt_source": "synthetic_repeat",
        "prompt_lens": [context_len],
        "batch_size": 1,
        "max_tokens": OUTPUT_TOKENS,
        "enforce_eager": False,
        "compile_size_specialization": True,
        "output_tokens": OUTPUT_TOKENS,
        "profile_token_ids_match": True,
    }
    for field, expected in expected_fields.items():
        _require_equal(row, field, expected, source=source)

    measured_seconds = row.get("measured_seconds")
    throughput = row.get("output_tokens_per_second")
    if (
        not isinstance(measured_seconds, (int, float))
        or not math.isfinite(measured_seconds)
        or measured_seconds <= 0
    ):
        raise RuntimeError(f"{source}: measured_seconds must be finite and positive")
    if (
        not isinstance(throughput, (int, float))
        or not math.isfinite(throughput)
        or throughput <= 0
    ):
        raise RuntimeError(
            f"{source}: output_tokens_per_second must be finite and positive"
        )
    recomputed_throughput = OUTPUT_TOKENS / measured_seconds
    if not math.isclose(
        throughput,
        recomputed_throughput,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{source}: throughput {throughput} disagrees with "
            f"{recomputed_throughput} recomputed from measured_seconds"
        )

    spec_metrics = row.get("spec_metrics")
    if not isinstance(spec_metrics, dict):
        raise RuntimeError(f"{source}: spec_metrics must be an object")
    if spec_metrics.get("acceptance_rate") is not None:
        raise RuntimeError(f"{source}: Q1 acceptance_rate must be null")
    if spec_metrics.get("mean_acceptance_length") != 1.0:
        raise RuntimeError(f"{source}: Q1 mean_acceptance_length must be 1.0")
    _validate_token_ids(row, source=source)


def load_rows(path: Path, *, mode: str) -> RowsByContext:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"could not read {path}: {error}") from error
    if not lines:
        raise RuntimeError(f"{path}: empty input")

    rows: RowsByContext = {}
    observed_order = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RuntimeError(f"{path}:{line_number}: blank JSONL row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{path}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number}: row must be a JSON object")
        _validate_row(row, path=path, line_number=line_number, mode=mode)
        context_len = row["context_len"]
        if context_len in rows:
            raise RuntimeError(f"{path}: duplicate context {context_len}")
        rows[context_len] = row
        observed_order.append(context_len)

    if tuple(observed_order) != CONTEXTS:
        raise RuntimeError(
            f"{path}: context order {tuple(observed_order)} differs from {CONTEXTS}; "
            "the prompt depends on context_index"
        )
    return rows


def load_available(reports_dir: Path) -> tuple[LoadedRows, list[Path]]:
    loaded: LoadedRows = {}
    missing = []
    for round_index in ROUNDS:
        for mode in MODES:
            path = expected_path(reports_dir, round_index, mode)
            if not path.is_file():
                missing.append(path)
                continue
            loaded[(round_index, mode)] = load_rows(path, mode=mode)
    return loaded, missing


def validate_cross_file_tokens(loaded: LoadedRows) -> dict[int, str]:
    token_hashes = {}
    for context_len in CONTEXTS:
        values = [
            (round_index, mode, rows[context_len]["token_ids"])
            for (round_index, mode), rows in sorted(loaded.items())
        ]
        if not values:
            continue
        canonical_round, canonical_mode, canonical_tokens = values[0]
        for round_index, mode, token_ids in values[1:]:
            if token_ids != canonical_tokens:
                raise RuntimeError(
                    "token mismatch at context "
                    f"{context_len}: run{round_index}_{mode} differs from "
                    f"run{canonical_round}_{canonical_mode}"
                )
        payload = json.dumps(
            canonical_tokens,
            separators=(",", ":"),
        ).encode("utf-8")
        token_hashes[context_len] = hashlib.sha256(payload).hexdigest()
    return token_hashes


def precheck(reports_dir: Path) -> None:
    loaded, missing = load_available(reports_dir)
    if not loaded:
        raise RuntimeError(f"no expected inputs are present under {reports_dir}")
    token_hashes = validate_cross_file_tokens(loaded)
    result = {
        "status": "incomplete" if missing else "complete",
        "validated_file_count": len(loaded),
        "expected_file_count": len(ROUNDS) * len(MODES),
        "validated_row_count": len(loaded) * len(CONTEXTS),
        "available_files": [
            expected_path(reports_dir, round_index, mode).name
            for round_index, mode in sorted(loaded)
        ],
        "missing_files": [path.name for path in missing],
        "available_tokens_exact": True,
        "token_sha256_by_context": {
            str(context_len): digest
            for context_len, digest in sorted(token_hashes.items())
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def round_result(
    round_index: int,
    context_len: int,
    loaded: LoadedRows,
) -> dict[str, Any]:
    four = loaded[(round_index, "four_kernel")][context_len]
    two = loaded[(round_index, "two_kernel")][context_len]
    raw = loaded[(round_index, "raw")][context_len]
    four_tps = four["output_tokens_per_second"]
    two_tps = two["output_tokens_per_second"]
    raw_tps = raw["output_tokens_per_second"]
    two_over_four = two_tps / four_tps
    four_over_raw = four_tps / raw_tps
    two_over_raw = two_tps / raw_tps
    return {
        "round": round_index,
        "declared_mode_order": list(LATIN_SQUARE_ORDER[round_index]),
        "four_kernel_seconds": four["measured_seconds"],
        "two_kernel_seconds": two["measured_seconds"],
        "raw_seconds": raw["measured_seconds"],
        "four_kernel_tokens_per_second": four_tps,
        "two_kernel_tokens_per_second": two_tps,
        "raw_tokens_per_second": raw_tps,
        "two_over_four_ratio": two_over_four,
        "two_over_four_gain_percent": (two_over_four - 1.0) * 100.0,
        "four_over_raw_ratio": four_over_raw,
        "four_over_raw_gap_percent": (four_over_raw - 1.0) * 100.0,
        "two_over_raw_ratio": two_over_raw,
        "two_over_raw_gap_percent": (two_over_raw - 1.0) * 100.0,
        "raw_gap_closure_percentage_points": (two_over_raw - four_over_raw) * 100.0,
        "tokens_exact": True,
        "profile_tokens_exact": True,
    }


def aggregate_context(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "four_kernel_seconds",
        "two_kernel_seconds",
        "raw_seconds",
        "four_kernel_tokens_per_second",
        "two_kernel_tokens_per_second",
        "raw_tokens_per_second",
        "two_over_four_ratio",
        "two_over_four_gain_percent",
        "four_over_raw_ratio",
        "four_over_raw_gap_percent",
        "two_over_raw_ratio",
        "two_over_raw_gap_percent",
        "raw_gap_closure_percentage_points",
    )
    statistics_by_metric = {
        field: metric_stats([result[field] for result in rounds])
        for field in numeric_fields
    }
    candidate_stats = statistics_by_metric["two_over_four_ratio"]
    return {
        "statistics": statistics_by_metric,
        "median_passes_0_5_percent_retention_gate": (
            candidate_stats["median"] >= RETENTION_GATE_RATIO
        ),
        "all_rounds_non_regressing": candidate_stats["min"] >= 1.0,
        "rounds_at_or_above_retention_gate": sum(
            result["two_over_four_ratio"] >= RETENTION_GATE_RATIO for result in rounds
        ),
        "rounds": rounds,
    }


def csv_rows_for_context(
    context_len: int,
    context_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for result in context_summary["rounds"]:
        rows.append(
            {
                "context_len": context_len,
                "row_kind": "round",
                **result,
                "retention_gate_pass": (
                    result["two_over_four_ratio"] >= RETENTION_GATE_RATIO
                ),
            }
        )
    for statistic in ("median", "min", "max"):
        row: dict[str, Any] = {
            "context_len": context_len,
            "row_kind": statistic,
            "round": "",
            "declared_mode_order": "",
            "tokens_exact": True,
            "profile_tokens_exact": True,
            "retention_gate_pass": (
                context_summary["median_passes_0_5_percent_retention_gate"]
                if statistic == "median"
                else ""
            ),
        }
        for field, values in context_summary["statistics"].items():
            row[field] = values[statistic]
        rows.append(row)
    return rows


def write_summary(run_dir: Path, loaded: LoadedRows) -> None:
    token_hashes = validate_cross_file_tokens(loaded)
    contexts = {}
    csv_rows = []
    for context_len in CONTEXTS:
        rounds = [
            round_result(round_index, context_len, loaded) for round_index in ROUNDS
        ]
        context_summary = aggregate_context(rounds)
        context_summary["token_sha256"] = token_hashes[context_len]
        contexts[str(context_len)] = context_summary
        csv_rows.extend(csv_rows_for_context(context_len, context_summary))

    all_contexts_pass = all(
        value["median_passes_0_5_percent_retention_gate"] for value in contexts.values()
    )
    all_rounds_non_regressing = all(
        value["all_rounds_non_regressing"] for value in contexts.values()
    )
    summary = {
        "configuration": {
            "rounds": list(ROUNDS),
            "modes": list(MODES),
            "contexts": list(CONTEXTS),
            "batch_size": 1,
            "max_tokens": OUTPUT_TOKENS,
            "spec_tokens": 0,
            "enforce_eager": False,
            "compile_size_specialization": True,
            "declared_latin_square_order": {
                str(round_index): list(order)
                for round_index, order in LATIN_SQUARE_ORDER.items()
            },
        },
        "correctness": {
            "all_measured_tokens_exact_across_modes_and_rounds": True,
            "all_profile_replays_match_measured_tokens": True,
            "output_tokens_per_run": OUTPUT_TOKENS,
            "token_sha256_by_context": {
                str(context_len): digest for context_len, digest in token_hashes.items()
            },
        },
        "retention": {
            "threshold_percent": (RETENTION_GATE_RATIO - 1.0) * 100.0,
            "threshold_ratio": RETENTION_GATE_RATIO,
            "primary_metric": "median per-round two_over_four_ratio per context",
            "all_contexts_pass_median_gate": all_contexts_pass,
            "all_rounds_non_regressing": all_rounds_non_regressing,
            "retained_as_default": (all_contexts_pass and all_rounds_non_regressing),
        },
        "contexts": contexts,
    }

    analysis_dir = run_dir / "analysis"
    summary_path = analysis_dir / "e2e_ab_summary.json"
    csv_path = analysis_dir / "e2e_ab_summary.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(summary_path)
    print(csv_path)


def main() -> None:
    args = parse_args()
    reports_dir = args.run_dir / "reports"
    if args.precheck:
        precheck(reports_dir)
        return

    loaded, missing = load_available(reports_dir)
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise RuntimeError(f"missing required inputs: {missing_names}")
    write_summary(args.run_dir, loaded)


if __name__ == "__main__":
    main()
