# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

from scripts import byte_v2_hybrid_e2e_analysis as analysis


def _plan(variant: str) -> dict:
    if variant == "raw":
        num_blocks = 80
        compact_bytes = num_blocks * 65_536
        sidecar_bytes = workspace_bytes = 0
    elif variant == "hybrid":
        num_blocks = 100
        compact_bytes = num_blocks * 52_096
        sidecar_bytes = 1_000
        workspace_bytes = 2_000
    else:
        num_blocks = 101
        compact_bytes = num_blocks * 52_096
        sidecar_bytes = workspace_bytes = 0
    return {
        "available": True,
        "num_blocks": num_blocks,
        "compact_tensor_bytes": compact_bytes,
        "raw_fallback_sidecar_bytes": sidecar_bytes,
        "raw_staging_workspace_bytes": workspace_bytes,
        "total_planned_bytes": compact_bytes + sidecar_bytes + workspace_bytes,
        "num_byte_v2_layers": 32 if variant != "raw" else 0,
        "raw_fallback_slots_per_layer": 1 if variant == "hybrid" else 0,
        "raw_staging_slots": 128 if variant == "hybrid" else 0,
    }


def _row(context_len: int, variant: str, round_index: int) -> dict:
    tps = {
        1: {"compact": 90.0, "hybrid": 89.0, "raw": 100.0},
        2: {"compact": 92.0, "hybrid": 91.0, "raw": 100.0},
    }[round_index][variant]
    return {
        "context_len": context_len,
        "output_tokens": 2,
        "output_tokens_per_second": tps,
        "token_ids": [[context_len, 7]],
        "profile_token_ids_match": True,
        "kv_cache_plan": _plan(variant),
        "hybrid_raw_fallback_state": (
            {
                "enabled": True,
                "fully_initialized": True,
                "raw_page_count": 0,
                "free_count": 64,
                "slot_count": 64,
                "fatal": 0,
            }
            if variant == "hybrid"
            else None
        ),
    }


def _write_reports(run_dir: Path) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir()
    for round_index in (1, 2):
        for variant in analysis.VARIANTS:
            rows = [
                _row(context_len, variant, round_index) for context_len in (64, 128)
            ]
            path = reports_dir / f"round{round_index}_{variant}.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )


def _rewrite_row(run_dir: Path, round_index: int, variant: str, mutate) -> None:
    path = run_dir / "reports" / f"round{round_index}_{variant}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_run_writes_correctness_performance_and_memory(tmp_path: Path):
    _write_reports(tmp_path)

    summary, csv_rows = analysis.analyze_run(tmp_path, 2)
    summary_path, comparison_path = analysis.write_analysis(tmp_path, summary, csv_rows)

    context = summary["contexts"]["64"]
    assert context["median_tokens_per_second"] == {
        "compact": 91.0,
        "hybrid": 90.0,
        "raw": 100.0,
    }
    assert context["median_paired_ratios"]["compact_over_raw"] == pytest.approx(0.91)
    assert context["median_paired_ratios"]["hybrid_over_raw"] == pytest.approx(0.90)
    assert context["cross_round_token_exact"] is True
    assert summary["correctness"] == {
        "all_same_context_token_exact": True,
        "all_profile_replays_exact": True,
        "all_output_counts_exact": True,
        "cross_round_token_exact": True,
    }
    assert summary["hybrid_state"]["raw_page_count_sum"] == 0
    assert summary["hybrid_state"]["observation_count"] == 4
    memory = summary["memory"]["median"]
    expected_saving = 1.0 - (100 * 52_096 + 3_000) / (100 * 65_536)
    assert memory["same_capacity_memory_saving_ratio"] == pytest.approx(expected_saving)
    assert memory["fixed_budget_capacity_gain_ratio"] == pytest.approx(0.25)

    assert json.loads(summary_path.read_text()) == summary
    with comparison_path.open(newline="") as input_file:
        written_rows = list(csv.DictReader(input_file))
    assert [int(row["context_len"]) for row in written_rows] == [64, 128]


def test_analyze_run_rejects_token_mismatch(tmp_path: Path):
    _write_reports(tmp_path)
    _rewrite_row(
        tmp_path,
        1,
        "hybrid",
        lambda row: row["token_ids"][0].__setitem__(1, 8),
    )

    with pytest.raises(ValueError, match="token IDs differ"):
        analysis.analyze_run(tmp_path, 2)


def test_analyze_run_rejects_failed_profile_replay(tmp_path: Path):
    _write_reports(tmp_path)
    _rewrite_row(
        tmp_path,
        1,
        "compact",
        lambda row: row.__setitem__("profile_token_ids_match", False),
    )

    with pytest.raises(ValueError, match="profiler replay token IDs are not exact"):
        analysis.analyze_run(tmp_path, 2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("fatal", 1, "fatal=1"), ("free_count", 63, "does not equal slot_count")],
)
def test_analyze_run_rejects_invalid_hybrid_state(
    tmp_path: Path, field: str, value: int, message: str
):
    _write_reports(tmp_path)

    def mutate(row):
        row["hybrid_raw_fallback_state"][field] = value

    _rewrite_row(tmp_path, 1, "hybrid", mutate)

    with pytest.raises(ValueError, match=message):
        analysis.analyze_run(tmp_path, 2)


def test_analyze_run_rejects_cross_round_token_mismatch(tmp_path: Path):
    _write_reports(tmp_path)
    for variant in analysis.VARIANTS:
        _rewrite_row(
            tmp_path,
            2,
            variant,
            lambda row: row["token_ids"][0].__setitem__(1, 9),
        )

    with pytest.raises(ValueError, match="token IDs differ across rounds"):
        analysis.analyze_run(tmp_path, 2)
