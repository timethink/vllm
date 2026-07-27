#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan ByteV2 V5/V6 outlier-pool demand on an all-layer KV capture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

PAGE_TOKENS = 16
KV_HEADS = 8
HEAD_DIM = 128
DIM_TILE = 16
DIM_TILES = HEAD_DIM // DIM_TILE
TILES_PER_SIDE = KV_HEADS * DIM_TILES
TILES_PER_PAGE = 2 * TILES_PER_SIDE
RAW_PAGE_BYTES = 65_536
V5_METADATA_BYTES = 896
V6_128_METADATA_BYTES = 640
DENSE_PAYLOAD_BYTES = 49_152
OUTLIER_ENTRY_BYTES = 2
POOL_CAPACITIES = (128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    """Return the pool-demand quantiles used by the format gate."""

    values_f64 = values.to(torch.float64)
    return {
        "min": float(values_f64.min()),
        "mean": float(values_f64.mean()),
        "p50": float(torch.quantile(values_f64, 0.50)),
        "p90": float(torch.quantile(values_f64, 0.90)),
        "p95": float(torch.quantile(values_f64, 0.95)),
        "p99": float(torch.quantile(values_f64, 0.99)),
        "p999": float(torch.quantile(values_f64, 0.999)),
        "max": float(values_f64.max()),
    }


def capacity_results(demand: torch.Tensor) -> dict[str, dict[str, float | int]]:
    """Count pages that require authoritative raw promotion at each capacity."""

    pages = demand.numel()
    results: dict[str, dict[str, float | int]] = {}
    for capacity in POOL_CAPACITIES:
        overflow = int((demand > capacity).sum())
        results[str(capacity)] = {
            "overflow_pages": overflow,
            "overflow_rate": overflow / pages,
            "exact_capacity_pages": int((demand == capacity).sum()),
            "headroom_at_max": capacity - int(demand.max()),
        }
    return results


def page_bytes(metadata_bytes: int, pool_entries: int) -> int:
    """Return fixed compact-page bytes for a metadata/pool choice."""

    return metadata_bytes + DENSE_PAYLOAD_BYTES + pool_entries * OUTLIER_ENTRY_BYTES


def tiles_by_row(tensor: torch.Tensor) -> torch.Tensor:
    """Return high7 values as [layer, page, tile, row, dim]."""

    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16, got {tensor.dtype}")
    if tensor.ndim != 4 or tuple(tensor.shape[2:]) != (KV_HEADS, HEAD_DIM):
        raise ValueError(
            f"expected [layers, tokens, 8, 128], got {tuple(tensor.shape)}"
        )
    layers, tokens, _, _ = tensor.shape
    if tokens % PAGE_TOKENS:
        raise ValueError("the all-layer capture must contain complete pages")
    pages = tokens // PAGE_TOKENS
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return (
        ((bits >> 8) & 0x7F)
        .view(
            layers,
            pages,
            PAGE_TOKENS,
            KV_HEADS,
            DIM_TILES,
            DIM_TILE,
        )
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(layers, pages, TILES_PER_SIDE, PAGE_TOKENS, DIM_TILE)
    )


def allocation_table() -> torch.Tensor:
    """Return the production power-of-two segment capacity for counts 0..256."""

    table = torch.zeros(PAGE_TOKENS * DIM_TILE + 1, dtype=torch.int64)
    for count in range(1, table.numel()):
        table[count] = 1 << (count - 1).bit_length()
    return table


def scan_split(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Scan full-page and every valid-row prefix for one capture split."""

    key_tiles = tiles_by_row(key)
    value_tiles = tiles_by_row(value)
    if key_tiles.shape != value_tiles.shape:
        raise ValueError("K/V tile shapes differ")

    combined = torch.cat((key_tiles, value_tiles), dim=2)
    layers, pages_per_layer, _, _, _ = combined.shape
    alloc_table = allocation_table()
    prefix_demands: list[list[torch.Tensor]] = [[] for _ in range(PAGE_TOKENS)]
    prefix_exact: list[list[torch.Tensor]] = [[] for _ in range(PAGE_TOKENS)]
    layer_rows: list[dict[str, Any]] = []
    top_pages: list[dict[str, Any]] = []
    full_tile_outliers: list[torch.Tensor] = []

    for layer in range(layers):
        layer_tiles = combined[layer]
        flat_tiles = layer_tiles.reshape(
            pages_per_layer * TILES_PER_PAGE,
            PAGE_TOKENS,
            DIM_TILE,
        )
        histogram = torch.zeros(
            (flat_tiles.shape[0], 128),
            dtype=torch.int32,
        )
        full_counts = None
        full_demand = None
        full_exact = None
        for row in range(PAGE_TOKENS):
            row_values = flat_tiles[:, row, :].to(torch.int64)
            histogram.scatter_add_(
                1,
                row_values,
                torch.ones_like(row_values, dtype=torch.int32),
            )
            best_window = histogram.unfold(1, 8, 1).sum(dim=2).max(dim=1).values
            counts = ((row + 1) * DIM_TILE - best_window).reshape(
                pages_per_layer,
                TILES_PER_PAGE,
            )
            exact = counts.sum(dim=1).to(torch.int64)
            allocated = alloc_table[counts.to(torch.int64)].sum(dim=1)
            prefix_exact[row].append(exact)
            prefix_demands[row].append(allocated)
            if row == PAGE_TOKENS - 1:
                full_counts = counts
                full_demand = allocated
                full_exact = exact

        assert full_counts is not None
        assert full_demand is not None
        assert full_exact is not None
        full_tile_outliers.append(full_counts.reshape(-1))
        key_counts = full_counts[:, :TILES_PER_SIDE]
        value_counts = full_counts[:, TILES_PER_SIDE:]
        layer_row: dict[str, Any] = {
            "split": split,
            "layer": layer,
            "pages": pages_per_layer,
            "escape_entries": int(full_exact.sum()),
            "key_escape_entries": int(key_counts.sum()),
            "value_escape_entries": int(value_counts.sum()),
            "max_tile_outliers": int(full_counts.max()),
            "demand": quantiles(full_demand),
            "capacities": capacity_results(full_demand),
        }
        layer_rows.append(layer_row)

        top_count = min(8, pages_per_layer)
        top_values, top_indices = torch.topk(full_demand, top_count)
        for demand, page_idx in zip(
            top_values.tolist(),
            top_indices.tolist(),
            strict=True,
        ):
            top_pages.append(
                {
                    "split": split,
                    "layer": layer,
                    "page": page_idx,
                    "allocated_demand": int(demand),
                    "exact_outliers": int(full_exact[page_idx]),
                    "max_tile_outliers": int(full_counts[page_idx].max()),
                }
            )

    full_demand_all = torch.cat(prefix_demands[-1])
    full_exact_all = torch.cat(prefix_exact[-1])
    full_tile_outliers_all = torch.cat(full_tile_outliers)
    values = layers * pages_per_layer * TILES_PER_PAGE * PAGE_TOKENS * DIM_TILE
    prefix_summary = {}
    for row in range(PAGE_TOKENS):
        demand = torch.cat(prefix_demands[row])
        exact = torch.cat(prefix_exact[row])
        prefix_summary[str(row + 1)] = {
            "instances": demand.numel(),
            "valid_rows": row + 1,
            "exact_outliers": int(exact.sum()),
            "demand": quantiles(demand),
            "capacities": capacity_results(demand),
        }

    top_pages.sort(
        key=lambda item: (
            item["allocated_demand"],
            item["exact_outliers"],
        ),
        reverse=True,
    )
    summary: dict[str, Any] = {
        "split": split,
        "layers": layers,
        "tokens_per_layer": key.shape[1],
        "pages_per_layer": pages_per_layer,
        "pages": layers * pages_per_layer,
        "values": values,
        "escape_entries": int(full_exact_all.sum()),
        "escape_fraction": float(full_exact_all.sum()) / values,
        "max_tile_outliers": int(full_tile_outliers_all.max()),
        "tile_outlier_histogram": torch.bincount(
            full_tile_outliers_all.to(torch.int64),
            minlength=int(full_tile_outliers_all.max()) + 1,
        ).tolist(),
        "full_page_demand": quantiles(full_demand_all),
        "full_page_capacities": capacity_results(full_demand_all),
        "all_valid_row_prefixes": prefix_summary,
        "top_pages": top_pages[:32],
    }
    return summary, layer_rows


def main() -> None:
    """Run the all-layer V6 pool-demand scan."""

    args = parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    split_summaries = {}
    layer_rows = []
    for split in ("calibration", "evaluation"):
        summary, rows = scan_split(
            capture[f"{split}_key"],
            capture[f"{split}_value"],
            split=split,
        )
        split_summaries[split] = summary
        layer_rows.extend(rows)

    format_candidates = {
        "v5_1024": {
            "metadata_bytes": V5_METADATA_BYTES,
            "pool_entries": 1024,
            "page_bytes": page_bytes(V5_METADATA_BYTES, 1024),
        },
        "v6_512": {
            "metadata_bytes": V5_METADATA_BYTES,
            "pool_entries": 512,
            "page_bytes": page_bytes(V5_METADATA_BYTES, 512),
        },
        "v6_256": {
            "metadata_bytes": V5_METADATA_BYTES,
            "pool_entries": 256,
            "page_bytes": page_bytes(V5_METADATA_BYTES, 256),
        },
        "v6_128_current_metadata_not_abi_legal": {
            "metadata_bytes": V5_METADATA_BYTES,
            "pool_entries": 128,
            "page_bytes": page_bytes(V5_METADATA_BYTES, 128),
        },
        "v6_128_u8_metadata_target": {
            "metadata_bytes": V6_128_METADATA_BYTES,
            "pool_entries": 128,
            "page_bytes": page_bytes(V6_128_METADATA_BYTES, 128),
        },
    }
    for candidate in format_candidates.values():
        candidate["saving_vs_raw"] = 1.0 - candidate["page_bytes"] / RAW_PAGE_BYTES
        candidate["extra_saving_vs_v5_bytes"] = (
            page_bytes(V5_METADATA_BYTES, 1024) - candidate["page_bytes"]
        )

    output = {
        "schema_version": 1,
        "capture": {
            "path": str(args.capture.resolve()),
            "sha256": sha256_file(args.capture),
            "model": capture.get("model"),
            "dtype": capture.get("dtype"),
            "layers": capture.get("layers"),
            "num_kv_heads": capture.get("num_kv_heads"),
            "head_dim": capture.get("head_dim"),
            "capture_backend": capture.get("capture_backend"),
            "prompts": capture.get("prompts"),
        },
        "demand_contract": {
            "tile": "adaptive contiguous high7 window of width 8",
            "tile_outliers": "valid elements outside the best window",
            "segment_allocation": "0 if count=0 else next_power_of_two(count)",
            "page_demand": "sum of all 128 K/V tile segment allocations",
            "overflow": "page_demand > fixed pool entries",
            "fallback": "authoritative raw sidecar before compact publication",
        },
        "format_candidates": format_candidates,
        "splits": split_summaries,
        "limitations": [
            "one Llama-3.1-8B model",
            "one 512-token calibration and one 1024-token evaluation capture",
            "batch one natural-text prefill KV only",
            "no code, multilingual, long-context, multi-batch, or decode-tail sample",
            "zero observed overflow does not remove the raw fallback requirement",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.layer_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.layer_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "layer",
                "pages",
                "escape_entries",
                "key_escape_entries",
                "value_escape_entries",
                "max_tile_outliers",
                "demand_mean",
                "demand_p99",
                "demand_max",
                "overflow_128",
                "overflow_256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in layer_rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "layer": row["layer"],
                    "pages": row["pages"],
                    "escape_entries": row["escape_entries"],
                    "key_escape_entries": row["key_escape_entries"],
                    "value_escape_entries": row["value_escape_entries"],
                    "max_tile_outliers": row["max_tile_outliers"],
                    "demand_mean": row["demand"]["mean"],
                    "demand_p99": row["demand"]["p99"],
                    "demand_max": row["demand"]["max"],
                    "overflow_128": row["capacities"]["128"]["overflow_pages"],
                    "overflow_256": row["capacities"]["256"]["overflow_pages"],
                }
            )

    compact = {
        split: {
            "pages": summary["pages"],
            "escape_fraction": summary["escape_fraction"],
            "demand": summary["full_page_demand"],
            "overflow_128": summary["full_page_capacities"]["128"]["overflow_pages"],
            "overflow_256": summary["full_page_capacities"]["256"]["overflow_pages"],
        }
        for split, summary in split_summaries.items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
