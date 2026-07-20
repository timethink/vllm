#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate paired ByteV2/raw E2E context-sweep results."""

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
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {
        row["context_len"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def median(values: list[dict[str, Any]], name: str) -> float:
    return statistics.median(value[name] for value in values)


def main() -> None:
    args = parse_args()
    reports_dir = args.run_dir / "reports"
    analysis_dir = args.run_dir / "analysis"
    rounds = []
    contexts: set[int] | None = None

    for round_index in range(1, args.rounds + 1):
        byte_rows = load_rows(reports_dir / f"e2e_run{round_index}_byte.jsonl")
        raw_rows = load_rows(reports_dir / f"e2e_run{round_index}_raw.jsonl")
        if byte_rows.keys() != raw_rows.keys():
            raise RuntimeError(f"round {round_index} context sets differ")
        round_contexts = set(byte_rows)
        if contexts is None:
            contexts = round_contexts
        elif contexts != round_contexts:
            raise RuntimeError("context sets differ between rounds")

        paired = {}
        for context_len in sorted(round_contexts):
            byte = byte_rows[context_len]
            raw = raw_rows[context_len]
            token_exact = byte["token_ids"] == raw["token_ids"]
            profile_exact = (
                byte["profile_token_ids_match"] and raw["profile_token_ids_match"]
            )
            output_exact = byte["output_tokens"] == raw["output_tokens"] == 256
            if not token_exact or not profile_exact or not output_exact:
                raise RuntimeError(
                    f"round {round_index} context {context_len} is not exact"
                )
            ratio = byte["output_tokens_per_second"] / raw["output_tokens_per_second"]
            byte_decode_residual = byte["measured_seconds"] - byte["warmup_seconds"]
            raw_decode_residual = raw["measured_seconds"] - raw["warmup_seconds"]
            paired[context_len] = {
                "byte_seconds": byte["measured_seconds"],
                "raw_seconds": raw["measured_seconds"],
                "byte_prefill_proxy_seconds": byte["warmup_seconds"],
                "raw_prefill_proxy_seconds": raw["warmup_seconds"],
                "prefill_proxy_gap_ms": (byte["warmup_seconds"] - raw["warmup_seconds"])
                * 1000.0,
                "byte_decode_residual_seconds": byte_decode_residual,
                "raw_decode_residual_seconds": raw_decode_residual,
                "decode_residual_gap_ms": (byte_decode_residual - raw_decode_residual)
                * 1000.0,
                "decode_residual_gap_us_per_step": (
                    byte_decode_residual - raw_decode_residual
                )
                * 1_000_000.0
                / 255.0,
                "byte_tokens_per_second": byte["output_tokens_per_second"],
                "raw_tokens_per_second": raw["output_tokens_per_second"],
                "paired_ratio": ratio,
                "paired_gap_percent": (ratio - 1.0) * 100.0,
                "wall_gap_ms": (byte["measured_seconds"] - raw["measured_seconds"])
                * 1000.0,
                "token_exact": token_exact,
                "profile_exact": profile_exact,
            }
        rounds.append(paired)

    assert contexts is not None
    summary = {
        "round_count": args.rounds,
        "output_tokens": 256,
        "contexts": {},
        "all_pairs_token_exact": True,
        "all_profiles_token_exact": True,
        "cross_round_token_exact": True,
    }
    csv_rows = []
    for context_len in sorted(contexts):
        values = [paired[context_len] for paired in rounds]
        byte_tokens = []
        raw_tokens = []
        for round_index in range(1, args.rounds + 1):
            byte_rows = load_rows(reports_dir / f"e2e_run{round_index}_byte.jsonl")
            raw_rows = load_rows(reports_dir / f"e2e_run{round_index}_raw.jsonl")
            byte_tokens.append(byte_rows[context_len]["token_ids"])
            raw_tokens.append(raw_rows[context_len]["token_ids"])
        cross_round_exact = all(
            tokens == byte_tokens[0] for tokens in byte_tokens[1:] + raw_tokens
        )
        summary["cross_round_token_exact"] &= cross_round_exact

        context_summary = {
            "median_byte_seconds": median(values, "byte_seconds"),
            "median_raw_seconds": median(values, "raw_seconds"),
            "median_byte_prefill_proxy_seconds": median(
                values, "byte_prefill_proxy_seconds"
            ),
            "median_raw_prefill_proxy_seconds": median(
                values, "raw_prefill_proxy_seconds"
            ),
            "median_prefill_proxy_gap_ms": median(values, "prefill_proxy_gap_ms"),
            "median_byte_decode_residual_seconds": median(
                values, "byte_decode_residual_seconds"
            ),
            "median_raw_decode_residual_seconds": median(
                values, "raw_decode_residual_seconds"
            ),
            "median_decode_residual_gap_ms": median(values, "decode_residual_gap_ms"),
            "median_decode_residual_gap_us_per_step": median(
                values, "decode_residual_gap_us_per_step"
            ),
            "median_byte_tokens_per_second": median(values, "byte_tokens_per_second"),
            "median_raw_tokens_per_second": median(values, "raw_tokens_per_second"),
            "median_paired_ratio": median(values, "paired_ratio"),
            "median_paired_gap_percent": median(values, "paired_gap_percent"),
            "median_wall_gap_ms": median(values, "wall_gap_ms"),
            "min_paired_ratio": min(value["paired_ratio"] for value in values),
            "max_paired_ratio": max(value["paired_ratio"] for value in values),
            "cross_round_token_exact": cross_round_exact,
            "runs": values,
        }
        summary["contexts"][str(context_len)] = context_summary
        for round_index, value in enumerate(values, start=1):
            csv_rows.append({"context_len": context_len, "round": round_index, **value})
        csv_rows.append(
            {
                "context_len": context_len,
                "round": "median",
                "byte_seconds": context_summary["median_byte_seconds"],
                "raw_seconds": context_summary["median_raw_seconds"],
                "byte_prefill_proxy_seconds": context_summary[
                    "median_byte_prefill_proxy_seconds"
                ],
                "raw_prefill_proxy_seconds": context_summary[
                    "median_raw_prefill_proxy_seconds"
                ],
                "prefill_proxy_gap_ms": context_summary["median_prefill_proxy_gap_ms"],
                "byte_decode_residual_seconds": context_summary[
                    "median_byte_decode_residual_seconds"
                ],
                "raw_decode_residual_seconds": context_summary[
                    "median_raw_decode_residual_seconds"
                ],
                "decode_residual_gap_ms": context_summary[
                    "median_decode_residual_gap_ms"
                ],
                "decode_residual_gap_us_per_step": context_summary[
                    "median_decode_residual_gap_us_per_step"
                ],
                "byte_tokens_per_second": context_summary[
                    "median_byte_tokens_per_second"
                ],
                "raw_tokens_per_second": context_summary[
                    "median_raw_tokens_per_second"
                ],
                "paired_ratio": context_summary["median_paired_ratio"],
                "paired_gap_percent": context_summary["median_paired_gap_percent"],
                "wall_gap_ms": context_summary["median_wall_gap_ms"],
                "token_exact": True,
                "profile_exact": True,
            }
        )

    summary_path = analysis_dir / "e2e_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = analysis_dir / "e2e_summary.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(summary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
