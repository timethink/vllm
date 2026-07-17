# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a saved ByteV2 real-cache capture without loading a model."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    import torch

    from vllm.v1.attention.backends.byte_v2_ops import (
        byte_v2_paged_decode_attention_split_k_guarded,
    )

    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")

    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    query = capture["query"].to(device)
    kv_cache = capture["kv_cache"].to(device)
    page_unsafe_flags = capture["page_unsafe_flags"].to(device)
    block_tables = capture["block_tables"].to(device)
    seq_lens = capture["seq_lens"].to(device)
    reference = capture["output"].to(device)
    output = torch.empty_like(reference)
    exp_sums = torch.empty(
        capture["exp_sums_shape"], dtype=torch.float32, device=device
    )
    max_logits = torch.empty(
        capture["max_logits_shape"], dtype=torch.float32, device=device
    )
    tmp_out = torch.empty(capture["tmp_out_shape"], dtype=torch.float32, device=device)

    def replay() -> None:
        byte_v2_paged_decode_attention_split_k_guarded(
            output,
            exp_sums,
            max_logits,
            tmp_out,
            query,
            kv_cache,
            page_unsafe_flags,
            block_tables,
            seq_lens,
            **capture["kwargs"],
        )

    for _ in range(args.warmup):
        replay()
    torch.accelerator.synchronize()

    elapsed_ms: list[float] = []
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()

    replay()
    torch.accelerator.synchronize()
    result = {
        "capture": str(args.capture),
        "iters": args.iters,
        "median_us": statistics.median(elapsed_ms) * 1000.0,
        "mean_us": statistics.fmean(elapsed_ms) * 1000.0,
        "min_us": min(elapsed_ms) * 1000.0,
        "max_abs_diff": float((output.float() - reference.float()).abs().max().item()),
    }
    print("BYTE_V2_CAPTURE_REPLAY " + json.dumps(result, sort_keys=True))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
