# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark isolated prefill attention kernels for ByteV2 experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
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
        "--seq-lens",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="Per-sequence prefill lengths to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fa-version", type=int, default=None)
    parser.add_argument(
        "--include-raw-flash",
        action="store_true",
        help="Also benchmark non-paged vLLM flash_attn_varlen over raw K/V.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="profiles/byte_v2_prefill_bench.jsonl",
        help="Path for machine-readable benchmark results.",
    )
    return parser.parse_args()


def _dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


@contextmanager
def _force_torch_sdpa_flash() -> Iterator[None]:
    import torch

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        yield


def _sdpa_prefill_exact(
    q: Any,
    k: Any,
    v: Any,
    output: Any,
    cu_seqlens: Any,
    *,
    scale: float,
    force_flash: bool,
) -> Any:
    import torch

    context = _force_torch_sdpa_flash() if force_flash else nullcontext()
    q_per_kv = q.shape[1] // k.shape[1]
    with context:
        for seq_idx in range(cu_seqlens.shape[0] - 1):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
            k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
            v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
                enable_gqa=q_per_kv != 1,
            )
            output[start:end].copy_(attn_output.squeeze(0).transpose(0, 1))
    return output


def _sdpa_prefill_batched_flash(
    q: Any,
    k: Any,
    v: Any,
    output: Any,
    *,
    batch_size: int,
    seq_len: int,
    scale: float,
) -> Any:
    import torch

    q_per_kv = q.shape[1] // k.shape[1]
    q_bhsd = q.view(batch_size, seq_len, q.shape[1], q.shape[2]).transpose(1, 2)
    k_bhsd = k.view(batch_size, seq_len, k.shape[1], k.shape[2]).transpose(1, 2)
    v_bhsd = v.view(batch_size, seq_len, v.shape[1], v.shape[2]).transpose(1, 2)
    with _force_torch_sdpa_flash():
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
            enable_gqa=q_per_kv != 1,
        )
    output.copy_(attn_output.transpose(1, 2).reshape_as(output))
    return output


def _vllm_flash_prefill_raw(
    q: Any,
    k: Any,
    v: Any,
    output: Any,
    cu_seqlens: Any,
    *,
    seq_len: int,
    scale: float,
    fa_version: int,
) -> Any:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=output,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        fa_version=fa_version,
    )


def _vllm_flash_prefill_paged(
    q: Any,
    key_cache: Any,
    value_cache: Any,
    output: Any,
    cu_seqlens: Any,
    seq_lens: Any,
    block_table: Any,
    *,
    seq_len: int,
    scale: float,
    fa_version: int,
) -> Any:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens,
        seqused_k=seq_lens,
        block_table=block_table,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        fa_version=fa_version,
    )


def _make_paged_cache(
    k: Any,
    v: Any,
    *,
    batch_size: int,
    seq_len: int,
    block_size: int,
) -> tuple[Any, Any, Any]:
    import torch

    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    key_cache = torch.empty(
        num_blocks,
        block_size,
        k.shape[1],
        k.shape[2],
        dtype=k.dtype,
        device=k.device,
    )
    value_cache = torch.empty_like(key_cache)
    block_table = torch.empty(
        batch_size,
        blocks_per_seq,
        dtype=torch.int32,
        device=k.device,
    )
    for seq_idx in range(batch_size):
        block_start = seq_idx * blocks_per_seq
        block_end = block_start + blocks_per_seq
        block_table[seq_idx].copy_(
            torch.arange(block_start, block_end, dtype=torch.int32, device=k.device)
        )
        token_start = seq_idx * seq_len
        token_end = token_start + seq_len
        cache_start = block_start * block_size
        cache_end = cache_start + seq_len
        key_cache.view(-1, k.shape[1], k.shape[2])[cache_start:cache_end].copy_(
            k[token_start:token_end]
        )
        value_cache.view(-1, v.shape[1], v.shape[2])[cache_start:cache_end].copy_(
            v[token_start:token_end]
        )
    return key_cache, value_cache, block_table


def _stats_ms(samples: list[float]) -> dict[str, float]:
    sorted_samples = sorted(samples)
    p90_index = min(len(sorted_samples) - 1, int(0.9 * (len(sorted_samples) - 1)))
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": sorted_samples[p90_index],
    }


def _time_cuda_kernel(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], Any]:
    import torch

    result = None
    for _ in range(warmup):
        result = fn()
    torch.accelerator.synchronize()

    events = []
    for _ in range(iters):
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        events.append((start_event, end_event))
    torch.accelerator.synchronize()

    samples = [start.elapsed_time(end) for start, end in events]
    return _stats_ms(samples), result


def _max_abs_diff(lhs: Any, rhs: Any) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _benchmark_one(args: argparse.Namespace, seq_len: int) -> list[dict[str, Any]]:
    import torch

    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    dtype = _dtype(args.dtype)
    device = torch.device("cuda")
    total_tokens = args.batch_size * seq_len
    scale = args.head_size**-0.5
    torch.manual_seed(args.seed + seq_len)

    q = torch.randn(
        total_tokens,
        args.num_heads,
        args.head_size,
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        total_tokens,
        args.num_kv_heads,
        args.head_size,
        dtype=dtype,
        device=device,
    )
    v = torch.randn_like(k)
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full(
        (args.batch_size,),
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    key_cache, value_cache, block_table = _make_paged_cache(
        k,
        v,
        batch_size=args.batch_size,
        seq_len=seq_len,
        block_size=16,
    )

    fa_version = args.fa_version
    if fa_version is None:
        fa_version = get_flash_attn_version(head_size=args.head_size)
    if fa_version is None:
        raise RuntimeError("could not determine a vLLM FlashAttention version")

    outputs = {
        "byte_v2_sdpa": torch.empty_like(q),
        "torch_sdpa_flash": torch.empty_like(q),
        "torch_sdpa_flash_batched": torch.empty_like(q),
        "vllm_default_prefill": torch.empty_like(q),
    }
    if args.include_raw_flash:
        outputs["vllm_flash_attn_raw"] = torch.empty_like(q)

    funcs: dict[str, Callable[[], Any]] = {
        "byte_v2_sdpa": lambda: _sdpa_prefill_exact(
            q,
            k,
            v,
            outputs["byte_v2_sdpa"],
            cu_seqlens,
            scale=scale,
            force_flash=False,
        ),
        "torch_sdpa_flash": lambda: _sdpa_prefill_exact(
            q,
            k,
            v,
            outputs["torch_sdpa_flash"],
            cu_seqlens,
            scale=scale,
            force_flash=True,
        ),
        "torch_sdpa_flash_batched": lambda: _sdpa_prefill_batched_flash(
            q,
            k,
            v,
            outputs["torch_sdpa_flash_batched"],
            batch_size=args.batch_size,
            seq_len=seq_len,
            scale=scale,
        ),
        "vllm_default_prefill": lambda: _vllm_flash_prefill_paged(
            q,
            key_cache,
            value_cache,
            outputs["vllm_default_prefill"],
            cu_seqlens,
            seq_lens,
            block_table,
            seq_len=seq_len,
            scale=scale,
            fa_version=fa_version,
        ),
    }
    if args.include_raw_flash:
        funcs["vllm_flash_attn_raw"] = lambda: _vllm_flash_prefill_raw(
            q,
            k,
            v,
            outputs["vllm_flash_attn_raw"],
            cu_seqlens,
            seq_len=seq_len,
            scale=scale,
            fa_version=fa_version,
        )

    results = []
    reference = None
    for name, fn in funcs.items():
        try:
            stats, output = _time_cuda_kernel(
                fn,
                warmup=args.warmup,
                iters=args.iters,
            )
            normalized_output = output[0] if isinstance(output, tuple) else output
            if reference is None:
                reference = normalized_output.detach().clone()
                max_abs_diff = 0.0
            else:
                max_abs_diff = _max_abs_diff(normalized_output, reference)
            result = {
                "name": name,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "total_tokens": total_tokens,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_size": args.head_size,
                "dtype": args.dtype,
                "fa_version": fa_version,
                "warmup": args.warmup,
                "iters": args.iters,
                "max_abs_diff_vs_first": max_abs_diff,
                **stats,
            }
        except Exception as exc:
            result = {
                "name": name,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "total_tokens": total_tokens,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_size": args.head_size,
                "dtype": args.dtype,
                "fa_version": fa_version,
                "warmup": args.warmup,
                "iters": args.iters,
                "error": repr(exc),
            }
        results.append(result)
    return results


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "seq_len,name,median_ms,mean_ms,min_ms,p90_ms,"
        "speedup_vs_vllm_default,max_abs_diff_vs_first,error"
    )
    by_seq_default = {
        row["seq_len"]: row
        for row in rows
        if row["name"] == "vllm_default_prefill" and "error" not in row
    }
    for row in rows:
        error = row.get("error", "")
        default = by_seq_default.get(row["seq_len"])
        speedup = ""
        if default is not None and "error" not in row:
            speedup = f"{default['median_ms'] / row['median_ms']:.4f}"
        print(
            f"{row['seq_len']},{row['name']},"
            f"{row.get('median_ms', '')},{row.get('mean_ms', '')},"
            f"{row.get('min_ms', '')},{row.get('p90_ms', '')},"
            f"{speedup},{row.get('max_abs_diff_vs_first', '')},{error}"
        )


def main() -> None:
    _prepend_venv_bin_to_path()

    args = parse_args()

    import torch

    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA-compatible accelerator is required")

    rows = []
    for seq_len in args.seq_lens:
        print(f"prefill_bench.start seq_len={seq_len}", flush=True)
        rows.extend(_benchmark_one(args, seq_len))
        print(f"prefill_bench.done seq_len={seq_len}", flush=True)

    _write_jsonl(args.output_jsonl, rows)
    print(f"prefill_bench.output_jsonl={args.output_jsonl}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
