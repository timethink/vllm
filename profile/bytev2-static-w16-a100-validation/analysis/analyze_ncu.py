# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract launch-resource evidence from Raw/W16 Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_METRICS = (
    "Registers Per Thread",
    "Driver Shared Memory Per Block",
    "Dynamic Shared Memory Per Block",
    "Block Limit Registers",
    "Block Limit Shared Mem",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Waves Per SM",
)
OPTIONAL_METRICS = (
    "Static Shared Memory Per Block",
    "Shared Memory Configuration Size",
    "Block Limit Warps",
    "Block Limit Blocks",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--ncu-bin", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.ncu_dir.is_dir():
        parser.error(f"NCU directory does not exist: {args.ncu_dir}")
    for output in (args.output_json, args.output_md):
        if output.exists() and not args.force:
            parser.error(f"refusing to overwrite {output}; pass --force")
    return args


def _find_ncu(explicit: Path | None) -> Path:
    candidates = (
        explicit,
        Path("/opt/nvidia/nsight-compute/2025.4.1/ncu"),
        Path("/opt/nvidia/nsight-compute/2025.4.0/ncu"),
        Path(shutil.which("ncu")) if shutil.which("ncu") else None,
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("Nsight Compute executable not found; pass --ncu-bin")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _number(value: str) -> float | int:
    parsed = float(value.replace(",", ""))
    return int(parsed) if parsed.is_integer() else parsed


def _report_metrics(ncu: Path, report: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ncu), "--import", str(report), "--page", "details", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    header = '"ID","Process ID"'
    start = completed.stdout.find(header)
    if start < 0:
        raise RuntimeError(f"cannot find NCU CSV header in {report}")
    rows = list(csv.DictReader(io.StringIO(completed.stdout[start:])))
    if not rows:
        raise RuntimeError(f"no NCU metric rows in {report}")
    wanted = set(REQUIRED_METRICS) | set(OPTIONAL_METRICS)
    metrics: dict[str, float | int] = {}
    units: dict[str, str] = {}
    for row in rows:
        name = row.get("Metric Name")
        if name not in wanted or name in metrics:
            continue
        metrics[name] = _number(row["Metric Value"])
        units[name] = row["Metric Unit"]
    missing = sorted(set(REQUIRED_METRICS) - metrics.keys())
    if missing:
        raise RuntimeError(f"missing NCU metrics in {report}: {missing}")
    return {
        "kernel": rows[0]["Kernel Name"],
        "metrics": metrics,
        "units": units,
        "report": str(report),
    }


def _validate_driver(
    path: Path,
    *,
    backend: str,
    allow_non_a100: bool,
) -> dict[str, Any]:
    record = _load(path)
    if record.get("mode") != "profile" or record.get("profile_backend") != backend:
        raise RuntimeError(f"wrong profile backend in {path}")
    if record.get("wire_bitwise") is not True:
        raise RuntimeError(f"wire gate failed in {path}")
    correctness = record.get("correctness", {}).get("eager", {})
    output_mismatch = correctness.get("output_bit_mismatch")
    lse_mismatch = correctness.get("lse_bit_mismatch")
    if output_mismatch or lse_mismatch:
        raise RuntimeError(f"attention correctness gate failed in {path}")
    if record.get("profile_environment"):
        raise RuntimeError(f"profile controls were active in {path}")
    if not allow_non_a100 and record.get("device", {}).get("is_a100_sm80") is not True:
        raise RuntimeError(f"formal profile is not from A100 SM80: {path}")
    return record


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# A100 launch-resource comparison",
        "",
        "| Backend | Registers/thread | Dynamic shared/block | "
        "Register block limit | Shared block limit | Theoretical occupancy | "
        "Achieved occupancy | Waves/SM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in ("raw", "w16"):
        row = summary["backends"][backend]
        metrics = row["metrics"]
        units = row["units"]
        dynamic = metrics["Dynamic Shared Memory Per Block"]
        dynamic_unit = units["Dynamic Shared Memory Per Block"]
        theoretical = metrics["Theoretical Occupancy"]
        theoretical_unit = units["Theoretical Occupancy"]
        achieved = metrics["Achieved Occupancy"]
        achieved_unit = units["Achieved Occupancy"]
        lines.append(
            f"| {backend} | {metrics['Registers Per Thread']} | "
            f"{dynamic} {dynamic_unit} | {metrics['Block Limit Registers']} | "
            f"{metrics['Block Limit Shared Mem']} | "
            f"{theoretical} {theoretical_unit} | "
            f"{achieved} {achieved_unit} | {metrics['Waves Per SM']} |"
        )
    lines.extend(
        [
            "",
            "Both drivers passed BF16 wire and attention output/LSE bitwise gates.",
            "The lower of the register and shared-memory block limits determines ",
            "the resource ceiling; do not infer resident CTAs from shared "
            "memory alone.",
            "",
            f"FA2 extension SHA-256: `{summary['extension_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Validate profile drivers and extract comparable NCU launch metrics."""
    args = parse_args()
    ncu = _find_ncu(args.ncu_bin)
    results: dict[str, Any] = {}
    extension_hashes: set[str] = set()
    devices: set[str] = set()
    for backend in ("raw", "w16"):
        report = args.ncu_dir / f"{backend}.ncu-rep"
        driver_path = args.ncu_dir / f"{backend}_driver.json"
        if not report.is_file() or not driver_path.is_file():
            raise FileNotFoundError(f"missing {backend} NCU report or driver JSON")
        driver = _validate_driver(
            driver_path,
            backend=backend,
            allow_non_a100=args.allow_non_a100,
        )
        extension_hashes.add(driver["fa2_extension"]["sha256"])
        devices.add(driver["device"]["name"])
        results[backend] = _report_metrics(ncu, report)
        results[backend]["driver"] = str(driver_path)
        results[backend]["bitwise"] = True
    if len(extension_hashes) != 1 or len(devices) != 1:
        raise RuntimeError("Raw and W16 profiles used mixed binaries or devices")
    if "flash_fwd_splitkv_byte_v2_kernel" in results["raw"]["kernel"]:
        raise RuntimeError("Raw report captured the W16 kernel")
    if "flash_fwd_splitkv_byte_v2_kernel" not in results["w16"]["kernel"]:
        raise RuntimeError("W16 report did not capture the W16 main kernel")

    summary = {
        "schema_version": 1,
        "experiment": "bytev2_static_w16_a100_ncu_resources",
        "device": next(iter(devices)),
        "extension_sha256": next(iter(extension_hashes)),
        "ncu": str(ncu),
        "backends": results,
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
