# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and aggregate compact, hybrid, and raw ByteV2 E2E sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

VARIANTS = ("compact", "hybrid", "raw")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=_positive_int, default=3)
    return parser.parse_args()


def _report_path(run_dir: Path, round_index: int, variant: str) -> Path:
    name = f"round{round_index}_{variant}.jsonl"
    candidates = (run_dir / name, run_dir / "reports" / name)
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"missing {name}; looked in {run_dir} and {run_dir / 'reports'}"
        )
    if len(existing) > 1:
        raise ValueError(f"ambiguous input: both locations contain {name}")
    return existing[0]


def _as_nonnegative_int(value: Any, *, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: {field} must be a non-negative integer")
    return value


def _as_positive_float(value: Any, *, field: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{source}: {field} must be finite and positive")
    return result


def _output_token_count(token_ids: Any, *, source: str) -> int:
    if not isinstance(token_ids, list):
        raise ValueError(f"{source}: token_ids must be a list of output lists")
    count = 0
    for batch_index, output in enumerate(token_ids):
        if not isinstance(output, list):
            raise ValueError(
                f"{source}: token_ids[{batch_index}] must be an output list"
            )
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in output
        ):
            raise ValueError(f"{source}: token_ids must contain integer token IDs")
        count += len(output)
    return count


def _load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        source = f"{path}:{line_number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{source}: each JSONL record must be an object")
        context_len = _as_nonnegative_int(
            row.get("context_len"), field="context_len", source=source
        )
        if context_len == 0:
            raise ValueError(f"{source}: context_len must be positive")
        if context_len in rows:
            raise ValueError(f"{path}: duplicate context_len {context_len}")

        output_tokens = _as_nonnegative_int(
            row.get("output_tokens"), field="output_tokens", source=source
        )
        actual_output_tokens = _output_token_count(row.get("token_ids"), source=source)
        if output_tokens != actual_output_tokens:
            raise ValueError(
                f"{source}: output_tokens={output_tokens} does not match "
                f"the {actual_output_tokens} recorded token IDs"
            )
        _as_positive_float(
            row.get("output_tokens_per_second"),
            field="output_tokens_per_second",
            source=source,
        )
        if row.get("profile_token_ids_match") is not True:
            raise ValueError(f"{source}: profiler replay token IDs are not exact")
        rows[context_len] = row
    if not rows:
        raise ValueError(f"{path}: report contains no records")
    return rows


def _validated_plan(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    plan = row.get("kv_cache_plan")
    if not isinstance(plan, dict) or plan.get("available") is not True:
        raise ValueError(f"{source}: kv_cache_plan must be available")

    result: dict[str, Any] = {"available": True}
    for field in (
        "num_blocks",
        "compact_tensor_bytes",
        "raw_fallback_sidecar_bytes",
        "raw_staging_workspace_bytes",
        "total_planned_bytes",
        "num_byte_v2_layers",
        "raw_fallback_slots_per_layer",
        "raw_staging_slots",
    ):
        result[field] = _as_nonnegative_int(
            plan.get(field), field=f"kv_cache_plan.{field}", source=source
        )
    if result["num_blocks"] == 0 or result["compact_tensor_bytes"] == 0:
        raise ValueError(f"{source}: kv_cache_plan must allocate blocks and tensors")
    planned_sum = (
        result["compact_tensor_bytes"]
        + result["raw_fallback_sidecar_bytes"]
        + result["raw_staging_workspace_bytes"]
    )
    if result["total_planned_bytes"] != planned_sum:
        raise ValueError(
            f"{source}: kv_cache_plan.total_planned_bytes does not equal its "
            "compact, sidecar, and workspace components"
        )
    return result


def _stable_plan(
    rows: dict[int, dict[str, Any]],
    *,
    round_index: int,
    variant: str,
) -> dict[str, Any]:
    plans = [
        _validated_plan(
            row, source=f"round {round_index} {variant} context {context_len}"
        )
        for context_len, row in sorted(rows.items())
    ]
    if any(plan != plans[0] for plan in plans[1:]):
        raise ValueError(
            f"round {round_index} {variant}: kv_cache_plan changes by context"
        )
    plan = plans[0]
    sidecar_bytes = plan["raw_fallback_sidecar_bytes"]
    workspace_bytes = plan["raw_staging_workspace_bytes"]
    if variant == "hybrid":
        if sidecar_bytes == 0 or workspace_bytes == 0:
            raise ValueError(
                f"round {round_index} hybrid: sidecar and workspace must be planned"
            )
    elif sidecar_bytes != 0 or workspace_bytes != 0:
        raise ValueError(f"round {round_index} {variant}: unexpected hybrid allocation")
    return plan


def _validate_hybrid_state(
    row: dict[str, Any], *, round_index: int, context_len: int
) -> dict[str, int]:
    source = f"round {round_index} hybrid context {context_len}"
    state = row.get("hybrid_raw_fallback_state")
    if not isinstance(state, dict):
        raise ValueError(f"{source}: hybrid raw-fallback state was not collected")
    if state.get("enabled") is not True or state.get("fully_initialized") is not True:
        raise ValueError(
            f"{source}: hybrid raw-fallback state is not fully initialized"
        )
    raw_pages = _as_nonnegative_int(
        state.get("raw_page_count"), field="raw_page_count", source=source
    )
    free_count = _as_nonnegative_int(
        state.get("free_count"), field="free_count", source=source
    )
    slot_count = _as_nonnegative_int(
        state.get("slot_count"), field="slot_count", source=source
    )
    fatal = _as_nonnegative_int(state.get("fatal"), field="fatal", source=source)
    if fatal != 0:
        raise ValueError(f"{source}: hybrid raw-fallback fatal={fatal}")
    if free_count != slot_count:
        raise ValueError(
            f"{source}: free_count={free_count} does not equal slot_count={slot_count}"
        )
    if raw_pages + free_count != slot_count:
        raise ValueError(
            f"{source}: raw pages and free slots do not match slot capacity"
        )
    return {
        "round": round_index,
        "context_len": context_len,
        "raw_page_count": raw_pages,
        "free_count": free_count,
        "slot_count": slot_count,
        "fatal": fatal,
    }


def _median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def _memory_comparison(
    plans: dict[str, dict[str, Any]], *, round_index: int
) -> dict[str, Any]:
    compact = plans["compact"]
    hybrid = plans["hybrid"]
    raw = plans["raw"]
    raw_bytes, remainder = divmod(raw["total_planned_bytes"], raw["num_blocks"])
    if remainder:
        raise ValueError(
            f"round {round_index} raw: planned bytes are not block-divisible"
        )
    raw_same_capacity_bytes = raw_bytes * hybrid["num_blocks"]
    hybrid_bytes = hybrid["total_planned_bytes"]
    memory_saving_ratio = 1.0 - hybrid_bytes / raw_same_capacity_bytes
    capacity_gain_ratio = hybrid["num_blocks"] / raw["num_blocks"] - 1.0
    return {
        "round": round_index,
        "plans": plans,
        "raw_bytes_per_block": raw_bytes,
        "same_capacity_num_blocks": hybrid["num_blocks"],
        "same_capacity_raw_bytes": raw_same_capacity_bytes,
        "same_capacity_hybrid_bytes": hybrid_bytes,
        "same_capacity_memory_saving_ratio": memory_saving_ratio,
        "same_capacity_memory_saving_percent": memory_saving_ratio * 100.0,
        "fixed_budget_raw_num_blocks": raw["num_blocks"],
        "fixed_budget_hybrid_num_blocks": hybrid["num_blocks"],
        "fixed_budget_capacity_gain_ratio": capacity_gain_ratio,
        "fixed_budget_capacity_gain_percent": capacity_gain_ratio * 100.0,
        "fixed_budget_compact_num_blocks": compact["num_blocks"],
    }


def analyze_run(
    run_dir: Path, rounds: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze a paired compact/hybrid/raw E2E sweep.

    Args:
        run_dir: Directory containing reports directly or in reports/.
        rounds: Number of paired rounds to load.

    Returns:
        The JSON summary and one CSV row per context length.

    Raises:
        FileNotFoundError: If any expected report is missing.
        ValueError: If correctness, state, or allocation validation fails.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    reports: list[dict[str, dict[int, dict[str, Any]]]] = []
    memory_rounds = []
    state_observations = []
    contexts: set[int] | None = None
    for round_index in range(1, rounds + 1):
        round_reports = {
            variant: _load_rows(_report_path(run_dir, round_index, variant))
            for variant in VARIANTS
        }
        round_contexts = set(round_reports["compact"])
        for variant in VARIANTS[1:]:
            if set(round_reports[variant]) != round_contexts:
                raise ValueError(
                    f"round {round_index}: {variant} context set differs from compact"
                )
        if contexts is None:
            contexts = round_contexts
        elif contexts != round_contexts:
            raise ValueError(f"round {round_index}: context set differs across rounds")

        plans = {
            variant: _stable_plan(
                round_reports[variant],
                round_index=round_index,
                variant=variant,
            )
            for variant in VARIANTS
        }
        memory_rounds.append(_memory_comparison(plans, round_index=round_index))
        for context_len, row in sorted(round_reports["hybrid"].items()):
            state_observations.append(
                _validate_hybrid_state(
                    row, round_index=round_index, context_len=context_len
                )
            )
        reports.append(round_reports)

    assert contexts is not None
    context_summaries: dict[str, Any] = {}
    csv_rows = []
    all_cross_round_exact = True
    for context_len in sorted(contexts):
        baseline_tokens: Any = None
        output_tokens: int | None = None
        round_values = []
        cross_round_exact = True
        for round_index, round_reports in enumerate(reports, 1):
            rows = {
                variant: round_reports[variant][context_len] for variant in VARIANTS
            }
            tokens = [rows[variant]["token_ids"] for variant in VARIANTS]
            if any(value != tokens[0] for value in tokens[1:]):
                raise ValueError(
                    f"round {round_index} context {context_len}: token IDs differ"
                )
            counts = [rows[variant]["output_tokens"] for variant in VARIANTS]
            if any(value != counts[0] for value in counts[1:]):
                raise ValueError(
                    f"round {round_index} context {context_len}: output counts differ"
                )
            if output_tokens is None:
                output_tokens = counts[0]
            elif output_tokens != counts[0]:
                raise ValueError(
                    f"context {context_len}: output count differs across rounds"
                )
            if baseline_tokens is None:
                baseline_tokens = tokens[0]
            elif tokens[0] != baseline_tokens:
                raise ValueError(
                    f"context {context_len}: token IDs differ across rounds"
                )

            tps = {
                variant: float(rows[variant]["output_tokens_per_second"])
                for variant in VARIANTS
            }
            hybrid_over_compact = tps["hybrid"] / tps["compact"]
            round_values.append(
                {
                    "round": round_index,
                    "tokens_per_second": tps,
                    "ratios": {
                        "compact_over_raw": tps["compact"] / tps["raw"],
                        "hybrid_over_raw": tps["hybrid"] / tps["raw"],
                        "hybrid_over_compact": hybrid_over_compact,
                    },
                    "hybrid_vs_compact_tps_gap_percent": (hybrid_over_compact - 1.0)
                    * 100.0,
                    "hybrid_over_compact_time_overhead_percent": (
                        tps["compact"] / tps["hybrid"] - 1.0
                    )
                    * 100.0,
                }
            )

        all_cross_round_exact &= cross_round_exact
        median_tps = {
            variant: _median(
                [value["tokens_per_second"][variant] for value in round_values]
            )
            for variant in VARIANTS
        }
        median_ratios = {
            name: _median([value["ratios"][name] for value in round_values])
            for name in ("compact_over_raw", "hybrid_over_raw", "hybrid_over_compact")
        }
        median_tps_gap = _median(
            [value["hybrid_vs_compact_tps_gap_percent"] for value in round_values]
        )
        median_time_overhead = _median(
            [
                value["hybrid_over_compact_time_overhead_percent"]
                for value in round_values
            ]
        )
        context_summary = {
            "output_tokens": output_tokens,
            "median_tokens_per_second": median_tps,
            "median_paired_ratios": median_ratios,
            "median_hybrid_vs_compact_tps_gap_percent": median_tps_gap,
            "median_hybrid_over_compact_time_overhead_percent": (median_time_overhead),
            "token_exact": True,
            "profile_replay_exact": True,
            "output_count_exact": True,
            "cross_round_token_exact": cross_round_exact,
            "rounds": round_values,
        }
        context_summaries[str(context_len)] = context_summary
        csv_rows.append(
            {
                "context_len": context_len,
                "round_count": rounds,
                "output_tokens": output_tokens,
                "compact_median_tokens_per_second": median_tps["compact"],
                "hybrid_median_tokens_per_second": median_tps["hybrid"],
                "raw_median_tokens_per_second": median_tps["raw"],
                "compact_over_raw_median_ratio": median_ratios["compact_over_raw"],
                "hybrid_over_raw_median_ratio": median_ratios["hybrid_over_raw"],
                "hybrid_over_compact_median_ratio": median_ratios[
                    "hybrid_over_compact"
                ],
                "hybrid_vs_compact_median_tps_gap_percent": median_tps_gap,
                "hybrid_over_compact_median_time_overhead_percent": (
                    median_time_overhead
                ),
                "token_exact": True,
                "profile_replay_exact": True,
                "output_count_exact": True,
                "cross_round_token_exact": cross_round_exact,
            }
        )

    memory_median = {
        "same_capacity_memory_saving_ratio": _median(
            [value["same_capacity_memory_saving_ratio"] for value in memory_rounds]
        ),
        "same_capacity_memory_saving_percent": _median(
            [value["same_capacity_memory_saving_percent"] for value in memory_rounds]
        ),
        "fixed_budget_capacity_gain_ratio": _median(
            [value["fixed_budget_capacity_gain_ratio"] for value in memory_rounds]
        ),
        "fixed_budget_capacity_gain_percent": _median(
            [value["fixed_budget_capacity_gain_percent"] for value in memory_rounds]
        ),
        "hybrid_num_blocks": _median(
            [value["fixed_budget_hybrid_num_blocks"] for value in memory_rounds]
        ),
        "raw_num_blocks": _median(
            [value["fixed_budget_raw_num_blocks"] for value in memory_rounds]
        ),
    }
    raw_page_counts = [value["raw_page_count"] for value in state_observations]
    summary = {
        "round_count": rounds,
        "variants": list(VARIANTS),
        "contexts": context_summaries,
        "correctness": {
            "all_same_context_token_exact": True,
            "all_profile_replays_exact": True,
            "all_output_counts_exact": True,
            "cross_round_token_exact": all_cross_round_exact,
        },
        "hybrid_state": {
            "observation_count": len(state_observations),
            "all_fatal_zero": True,
            "all_free_equals_slots": True,
            "raw_page_count_sum": sum(raw_page_counts),
            "raw_page_count_max": max(raw_page_counts),
            "observations": state_observations,
        },
        "memory": {"median": memory_median, "rounds": memory_rounds},
    }
    return summary, csv_rows


def write_analysis(
    run_dir: Path, summary: dict[str, Any], csv_rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Write summary.json and comparison.csv under the run analysis directory."""
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison_path = analysis_dir / "comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary_path, comparison_path


def main() -> None:
    args = parse_args()
    summary, csv_rows = analyze_run(args.run_dir, args.rounds)
    summary_path, comparison_path = write_analysis(args.run_dir, summary, csv_rows)
    print(summary_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
