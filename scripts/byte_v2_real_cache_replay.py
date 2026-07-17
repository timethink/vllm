# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay one ByteV2 decode layer against a real model-generated KV cache."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any


def _prepend_venv_bin_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    venv_bin = repo_root / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/mnt/sdb/yxz/ByteV2/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--prompt-token-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--capture-layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-capture", default=None)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap replay iterations in cudaProfilerStart/Stop for NCU.",
    )
    return parser.parse_args()


def _make_prompt_token_ids(model: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    seed_text = (
        "Deterministic decoding gives regression tests a stable signal. "
        "Long context attention should preserve the same factual details. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced no token ids")
    repeats = (prompt_len + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:prompt_len]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _save_replay_capture(captured: dict[str, Any], path: str) -> None:
    import torch

    block_tables = captured["block_tables"].detach().cpu()
    physical_blocks = torch.unique(block_tables[block_tables >= 0].to(torch.long))
    remapped_block_tables = block_tables.clone()
    for logical_block, physical_block in enumerate(physical_blocks.tolist()):
        remapped_block_tables[block_tables == physical_block] = logical_block

    payload = {
        "output": captured["output"].detach().cpu(),
        "query": captured["query"].detach().cpu(),
        "kv_cache": captured["kv_cache"]
        .index_select(0, physical_blocks.to(captured["kv_cache"].device))
        .detach()
        .cpu(),
        "page_unsafe_flags": captured["page_unsafe_flags"]
        .index_select(0, physical_blocks.to(captured["page_unsafe_flags"].device))
        .detach()
        .cpu(),
        "block_tables": remapped_block_tables,
        "seq_lens": captured["seq_lens"].detach().cpu(),
        "exp_sums_shape": captured["exp_sums_shape"],
        "max_logits_shape": captured["max_logits_shape"],
        "tmp_out_shape": captured["tmp_out_shape"],
        "kwargs": captured["kwargs"],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    _prepend_venv_bin_to_path()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.v1.attention.backends import byte_v2_attn
    from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayoutV4
    from vllm.v1.attention.backends.byte_v2_ops import byte_v2_collect_cache_stats

    args = parse_args()
    if args.capture_layer < 0:
        raise ValueError("--capture-layer must be non-negative")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")

    original = byte_v2_attn.byte_v2_paged_decode_attention_split_k_guarded
    calls_by_seq_len: dict[int, int] = {}
    captured: dict[str, Any] = {}

    def capture_wrapper(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        **kwargs,
    ):
        original(
            output,
            exp_sums,
            max_logits,
            tmp_out,
            query,
            kv_cache,
            page_unsafe_flags,
            block_tables,
            seq_lens,
            **kwargs,
        )
        max_seq_len = int(kwargs["max_seq_len"])
        call_index = calls_by_seq_len.get(max_seq_len, 0)
        calls_by_seq_len[max_seq_len] = call_index + 1
        if (
            not captured
            and max_seq_len >= args.prompt_token_len
            and call_index == args.capture_layer
        ):
            captured.update(
                output=output.detach().clone(),
                query=query.detach().clone(),
                kv_cache=kv_cache,
                page_unsafe_flags=page_unsafe_flags,
                block_tables=block_tables.detach().clone(),
                seq_lens=seq_lens.detach().clone(),
                exp_sums_shape=tuple(exp_sums.shape),
                max_logits_shape=tuple(max_logits.shape),
                tmp_out_shape=tuple(tmp_out.shape),
                kwargs=dict(kwargs),
                max_seq_len=max_seq_len,
            )

    byte_v2_attn.byte_v2_paged_decode_attention_split_k_guarded = capture_wrapper
    max_model_len = args.prompt_token_len + args.max_tokens + 32
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        block_size=16,
        enable_prefix_caching=False,
        attention_config={"backend": "BYTE_V2"},
    )
    try:
        prompt = TokensPrompt(
            prompt_token_ids=_make_prompt_token_ids(args.model, args.prompt_token_len)
        )
        outputs = llm.generate(
            [prompt],
            SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        torch.accelerator.synchronize()
        if not captured:
            raise RuntimeError(
                "no guarded split-K call was captured; check the ByteV2 fast-path env"
            )
        if args.save_capture is not None:
            _save_replay_capture(captured, args.save_capture)

        output = torch.empty_like(captured["output"])
        exp_sums = torch.empty(
            captured["exp_sums_shape"],
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty(
            captured["max_logits_shape"],
            dtype=torch.float32,
            device=output.device,
        )
        tmp_out = torch.empty(
            captured["tmp_out_shape"],
            dtype=torch.float32,
            device=output.device,
        )
        replay_kwargs = dict(captured["kwargs"])

        def replay() -> None:
            original(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                captured["query"],
                captured["kv_cache"],
                captured["page_unsafe_flags"],
                captured["block_tables"],
                captured["seq_lens"],
                **replay_kwargs,
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
        max_abs_diff = float(
            (output.float() - captured["output"].float()).abs().max().item()
        )

        stats = torch.zeros((4,), dtype=torch.int32, device=output.device)
        byte_v2_collect_cache_stats(
            stats,
            captured["kv_cache"],
            captured["block_tables"],
            captured["seq_lens"],
            max_seq_len=captured["max_seq_len"],
            tile_policy=(16, 16, 16, 64, 128, 128),
        )
        stats_cpu = stats.cpu().tolist()
        blocks_per_seq = (captured["max_seq_len"] + 15) // 16
        physical_blocks = (
            captured["block_tables"][:, :blocks_per_seq]
            .detach()
            .flatten()
            .to(torch.long)
        )
        physical_blocks = torch.unique(physical_blocks[physical_blocks >= 0])
        page_flags = captured["page_unsafe_flags"].index_select(0, physical_blocks)
        unsafe_pages = int((page_flags != 0).sum().item())
        layout = ByteV2PageLayoutV4()
        total_tiles = (
            int(stats_cpu[3])
            * layout.num_kv_heads
            * (
                layout.tile_policy.codec_tiles_per_k_page
                + layout.tile_policy.codec_tiles_per_v_page
            )
        )
        unsafe_tiles = int(stats_cpu[1]) + int(stats_cpu[2])

        result = {
            "prompt_token_len": args.prompt_token_len,
            "generated_token_ids": list(outputs[0].outputs[0].token_ids),
            "capture_layer": args.capture_layer,
            "captured_max_seq_len": captured["max_seq_len"],
            "partition_size": int(replay_kwargs["partition_size"]),
            "tile_policy": list(replay_kwargs["tile_policy"]),
            "workspace_shapes": {
                "exp_sums": captured["exp_sums_shape"],
                "max_logits": captured["max_logits_shape"],
                "tmp_out": captured["tmp_out_shape"],
            },
            "replay": {
                "median_us": statistics.median(elapsed_ms) * 1000.0,
                "mean_us": statistics.fmean(elapsed_ms) * 1000.0,
                "min_us": min(elapsed_ms) * 1000.0,
                "p90_us": _percentile(elapsed_ms, 0.90) * 1000.0,
                "iters": args.iters,
                "max_abs_diff": max_abs_diff,
            },
            "cache": {
                "pages": int(stats_cpu[3]),
                "unsafe_pages": unsafe_pages,
                "unsafe_page_ratio": unsafe_pages / max(int(stats_cpu[3]), 1),
                "fallback_tiles": int(stats_cpu[1]),
                "outlier_tiles": int(stats_cpu[2]),
                "total_tiles": total_tiles,
                "unsafe_tile_ratio": unsafe_tiles / max(total_tiles, 1),
            },
        }
        print("BYTE_V2_REAL_CACHE_REPLAY " + json.dumps(result, sort_keys=True))
        if args.output_json is not None:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        byte_v2_attn.byte_v2_paged_decode_attention_split_k_guarded = original


if __name__ == "__main__":
    main()
