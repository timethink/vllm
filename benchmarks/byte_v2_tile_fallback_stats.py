# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect real ByteV2 tile-level fallback statistics from a vLLM run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _make_prompt_token_ids(model: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        add_special_tokens=False,
    )
    if not base:
        base = [0]
    repeats = (prompt_len + len(base) - 1) // len(base)
    return (base * repeats)[:prompt_len]


def _collect_rpc(llm: Any, method: str) -> Any:
    try:
        return llm.collective_rpc(method)
    except Exception as exc:  # pragma: no cover - diagnostic script.
        return {"error": repr(exc)}


def _merge_tile_worker_stats(worker_stats: Any) -> dict[str, Any]:
    if not isinstance(worker_stats, list):
        return {"error": "worker stats are not a list", "raw": worker_stats}
    numeric_sum_keys = (
        "enabled_layers",
        "total_full_active_blocks",
        "total_full_raw_fallback_blocks",
        "total_partial_raw_fallback_blocks",
        "total_full_tiles",
        "total_raw_full_tiles",
        "total_full_bad_tiles",
        "total_full_good_tiles_inside_raw_fallback_blocks",
        "total_sum_bad_tile_misses",
        "bad_tiles_misses_le_1",
        "bad_tiles_misses_le_2",
        "bad_tiles_misses_le_4",
        "bad_tiles_misses_le_8",
        "bad_tiles_misses_gt_8",
        "invalid_raw_fallback_slots",
    )
    merged: dict[str, Any] = {
        "num_workers": len(worker_stats),
    }
    for key in numeric_sum_keys:
        merged[key] = sum(int(stats.get(key, 0)) for stats in worker_stats)

    total_full_tiles = int(merged["total_full_tiles"])
    total_raw_full_tiles = int(merged["total_raw_full_tiles"])
    total_bad_tiles = int(merged["total_full_bad_tiles"])
    total_misses = int(merged["total_sum_bad_tile_misses"])
    merged["total_full_tile_fallback_ratio"] = (
        total_bad_tiles / total_full_tiles if total_full_tiles > 0 else 0.0
    )
    merged["total_bad_tile_ratio_within_raw_fallback_blocks"] = (
        total_bad_tiles / total_raw_full_tiles
        if total_raw_full_tiles > 0
        else 0.0
    )
    merged["mean_misses_per_bad_tile"] = (
        total_misses / total_bad_tiles if total_bad_tiles > 0 else 0.0
    )
    merged["max_misses_per_bad_tile"] = max(
        (int(stats.get("max_misses_per_bad_tile", 0)) for stats in worker_stats),
        default=0,
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE", "1")
    os.environ.setdefault("VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL", "1")
    os.environ.setdefault("VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE", "0")

    import torch

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    max_model_len = args.max_model_len or args.prompt_len + args.decode_len + 16
    max_num_batched_tokens = args.max_num_batched_tokens or (
        args.batch_size * (args.prompt_len + args.decode_len)
    )
    prompt_ids = _make_prompt_token_ids(args.model, args.prompt_len)
    prompts = [
        TokensPrompt(prompt_token_ids=list(prompt_ids))
        for _ in range(args.batch_size)
    ]

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        kv_cache_dtype="byte_v2",
        block_size=args.block_size,
        enforce_eager=True,
        enable_prefix_caching=not args.disable_prefix_caching,
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        max_tokens=args.decode_len,
        min_tokens=args.decode_len,
        ignore_eos=True,
        temperature=0.0,
        detokenize=False,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sparse_stats = _collect_rpc(llm, "get_byte_v2_sparse_fallback_stats")
    tile_worker_stats = _collect_rpc(llm, "get_byte_v2_tile_fallback_stats")
    result = {
        "model": args.model,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
        "output_tokens": sum(
            len(request_output.outputs[0].token_ids)
            for request_output in outputs
        ),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE",
                "VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL",
                "VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO",
                "VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS",
                "VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE",
            )
            if os.environ.get(key) is not None
        },
        "sparse_fallback_worker_stats": sparse_stats,
        "tile_fallback_worker_stats": tile_worker_stats,
        "tile_fallback_stats": _merge_tile_worker_stats(tile_worker_stats),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
