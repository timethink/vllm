# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the current safe ByteV2 single-token raw-staging update."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from vllm.v1.attention.backends.byte_v2_layout import (
    ByteV2PageLayoutV5,
    ByteV2RawStagingLayout,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_reshape_and_cache,
    byte_v2_update_cache_raw_staging,
)

TILE_POLICY = (16, 16, 16, 64, 128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, choices=range(16), default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fused-stage", action="store_true")
    parser.add_argument(
        "--comparison",
        action="store_true",
        help=(
            "Benchmark the baseline and fused-stage candidate in one process, "
            "alternating their execution order between trials."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()

    torch.manual_seed(20260719 + args.row)
    kv_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    if args.row:
        initial_key = torch.randn(
            args.row,
            8,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        initial_value = torch.randn_like(initial_key)
        initial_slots = torch.arange(args.row, dtype=torch.int64, device="cuda")
        byte_v2_reshape_and_cache(
            initial_key,
            initial_value,
            kv_cache,
            initial_slots,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )

    key = torch.randn(1, 8, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.tensor([args.row], dtype=torch.int64, device="cuda")
    raw_staging = torch.empty(
        (1, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_to_staging_slot = torch.full(
        (1,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    staging_to_physical_block = torch.full(
        (1,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    valid_rows = torch.zeros((1,), dtype=torch.int32, device="cuda")
    next_staging_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    page_unsafe_flags = torch.zeros((1,), dtype=torch.int32, device="cuda")

    def update(fused_stage: bool) -> None:
        byte_v2_update_cache_raw_staging(
            key,
            value,
            raw_staging,
            kv_cache,
            slot_mapping,
            block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            next_staging_slot,
            overflow,
            page_unsafe_flags,
            tile_policy=TILE_POLICY,
            fuse_metadata_clear=True,
            warp_parallel_histogram=True,
            fuse_single_token_staging=fused_stage,
        )

    if args.comparison and args.profile:
        raise ValueError("--comparison and --profile cannot be used together")
    if args.comparison and args.fused_stage:
        raise ValueError("--comparison already benchmarks both staging modes")

    if args.comparison:
        for warmup_index in range(args.warmup):
            warmup_order = (False, True) if warmup_index % 2 == 0 else (True, False)
            for fused_stage in warmup_order:
                update(fused_stage)
    else:
        for _ in range(args.warmup):
            update(args.fused_stage)
    torch.accelerator.synchronize()

    if args.profile:
        torch.cuda.nvtx.range_push(f"byte_v2_safe_n1_update_row{args.row}")
        for _ in range(args.iterations):
            update(args.fused_stage)
        torch.cuda.nvtx.range_pop()
        torch.accelerator.synchronize()
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "row": args.row,
                    "iterations": args.iterations,
                    "profile": True,
                    "fused_stage": args.fused_stage,
                },
                sort_keys=True,
            )
        )
        return

    def measure(fused_stage: bool) -> tuple[float, float]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.accelerator.synchronize()
        wall_start_ns = time.perf_counter_ns()
        start.record()
        for _ in range(args.iterations):
            update(fused_stage)
        end.record()
        end.synchronize()
        wall_end_ns = time.perf_counter_ns()
        event_us = start.elapsed_time(end) * 1000.0 / args.iterations
        wall_us = (wall_end_ns - wall_start_ns) / 1000.0 / args.iterations
        return event_us, wall_us

    if args.comparison:
        samples_us = {
            "baseline": {"event": [], "wall": []},
            "candidate": {"event": [], "wall": []},
        }
        rounds = []
        for trial_index in range(args.trials):
            order = (
                ("baseline", False),
                ("candidate", True),
            )
            if trial_index % 2:
                order = tuple(reversed(order))

            round_result = {
                "round": trial_index,
                "order": [name for name, _ in order],
            }
            for name, fused_stage in order:
                event_us, wall_us = measure(fused_stage)
                samples_us[name]["event"].append(event_us)
                samples_us[name]["wall"].append(wall_us)
                round_result[name] = {"event": event_us, "wall": wall_us}
            rounds.append(round_result)

        median_us = {
            name: {
                timer: statistics.median(samples)
                for timer, samples in mode_samples.items()
            }
            for name, mode_samples in samples_us.items()
        }
        candidate_delta_vs_baseline_pct = {
            timer: (median_us["candidate"][timer] / median_us["baseline"][timer] - 1.0)
            * 100.0
            for timer in ("event", "wall")
        }
        result = {
            "device": torch.cuda.get_device_name(),
            "row": args.row,
            "iterations": args.iterations,
            "trials": args.trials,
            "comparison": True,
            "candidate_delta_vs_baseline_pct": (candidate_delta_vs_baseline_pct),
            "median_us": median_us,
            "rounds": rounds,
            "samples_us": samples_us,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    event_samples_us = []
    wall_samples_us = []
    for _ in range(args.trials):
        event_us, wall_us = measure(args.fused_stage)
        event_samples_us.append(event_us)
        wall_samples_us.append(wall_us)

    result = {
        "device": torch.cuda.get_device_name(),
        "row": args.row,
        "iterations": args.iterations,
        "trials": args.trials,
        "fused_stage": args.fused_stage,
        "median_us": {
            "event": statistics.median(event_samples_us),
            "wall": statistics.median(wall_samples_us),
        },
        "samples_us": {
            "event": event_samples_us,
            "wall": wall_samples_us,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
