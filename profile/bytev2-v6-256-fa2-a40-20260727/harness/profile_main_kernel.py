# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile one 65K raw, ByteV2 V6, or SplitZip FA2 main-kernel launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splitzip_fixed_page import pack_splitzip_fixed_pages  # noqa: E402

from scripts import byte_v2_fa2_oracle as oracle  # noqa: E402
from vllm.v1.attention.backends.byte_v2_ops import (  # noqa: E402
    byte_v2_reshape_and_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("raw", "bytev2", "splitzip"),
        required=True,
    )
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--num-splits", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--capture",
        type=Path,
        default=(
            REPO_ROOT / "profile/bytev2-splitzip-codec-a40-20260726/raw/"
            "llama_layer0_cal2048_eval4096.pt"
        ),
    )
    args = parser.parse_args()
    if args.seq_len <= 0 or args.seq_len % 16:
        parser.error("--seq-len must be a positive multiple of 16")
    if not 1 <= args.num_splits <= 128:
        parser.error("--num-splits must be in [1, 128]")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def repeat_capture(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    repeats = (tokens + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats, 1, 1))[:tokens].contiguous()


def install_real_capture(
    tensors: dict[str, torch.Tensor | float | int],
    capture_path: Path,
    seq_len: int,
) -> None:
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    key_cpu = repeat_capture(capture["evaluation_key"], seq_len)
    value_cpu = repeat_capture(capture["evaluation_value"], seq_len)
    if (
        key_cpu.dtype != torch.bfloat16
        or value_cpu.dtype != torch.bfloat16
        or key_cpu.shape[1:] != (8, 128)
        or value_cpu.shape != key_cpu.shape
    ):
        raise ValueError("capture must contain BF16 K/V tensors shaped [N, 8, 128]")

    key = tensors["key"]
    value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    key.copy_(key_cpu.to(device=key.device).view_as(key))
    value.copy_(value_cpu.to(device=value.device).view_as(value))
    byte_cache.zero_()
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=key.device)
    byte_v2_reshape_and_cache(
        key.view(seq_len, 8, 128),
        value.view(seq_len, 8, 128),
        byte_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    torch.accelerator.synchronize()


def make_calls(
    tensors: dict[str, torch.Tensor | float | int],
    splitzip_cache: torch.Tensor,
    *,
    num_splits: int,
) -> tuple[Any, dict[str, Any]]:
    query = tensors["query"]
    max_seq_len = tensors["max_seq_len"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(max_seq_len, int)
    outputs = {name: torch.empty_like(query) for name in ("raw", "bytev2", "splitzip")}
    common: tuple[Any, ...] = (
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        1,
        max_seq_len,
        0.0,
        tensors["scale"],
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        num_splits,
        None,
    )

    def raw() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.varlen_fwd(
            query,
            tensors["key"],
            tensors["value"],
            outputs["raw"],
            *common,
        )

    def bytev2() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.byte_v2_varlen_fwd(
            query,
            tensors["byte_cache"],
            None,
            outputs["bytev2"],
            *common,
        )

    def splitzip() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._vllm_fa2_C.splitzip_varlen_fwd(
            query,
            splitzip_cache,
            outputs["splitzip"],
            *common,
        )

    return raw, {"bytev2": bytev2, "splitzip": splitzip}


def bit_mismatches(
    candidate: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> tuple[int, int]:
    output_mismatches = int(
        (candidate[0].view(torch.int16) != reference[0].view(torch.int16)).sum()
    )
    lse_mismatches = int(
        (candidate[1].view(torch.int32) != reference[1].view(torch.int32)).sum()
    )
    return output_mismatches, lse_mismatches


def main() -> None:
    args = parse_args()
    if not oracle.fa2_oracle_ops_are_available():
        raise RuntimeError("raw and ByteV2 FA2 ops are unavailable")
    if not hasattr(torch.ops._vllm_fa2_C, "splitzip_varlen_fwd"):
        raise RuntimeError("SplitZip FA2 reader op is unavailable")

    tensors = oracle.make_inputs((args.seq_len,), query_len=1, permute_pages=False)
    install_real_capture(tensors, args.capture, args.seq_len)
    key = tensors["key"]
    value = tensors["value"]
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    splitzip_cache, pack_stats = pack_splitzip_fixed_pages(key, value)
    raw, candidates = make_calls(
        tensors,
        splitzip_cache,
        num_splits=args.num_splits,
    )
    candidate = raw if args.backend == "raw" else candidates[args.backend]

    reference_result = raw()
    candidate_result = candidate()
    torch.accelerator.synchronize()
    output_mismatches, lse_mismatches = bit_mismatches(
        candidate_result,
        reference_result,
    )
    if output_mismatches or lse_mismatches:
        raise RuntimeError(
            "candidate failed bitwise gate: "
            f"output={output_mismatches}, lse={lse_mismatches}"
        )

    for _ in range(args.warmup):
        candidate()
    torch.accelerator.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    candidate()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    properties = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "seq_len": args.seq_len,
                "query_len": 1,
                "num_splits": args.num_splits,
                "warmup": args.warmup,
                "capture": str(args.capture),
                "gpu": properties.name,
                "output_bit_mismatches": output_mismatches,
                "lse_bit_mismatches": lse_mismatches,
                "splitzip_pack_stats": pack_stats.to_dict(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
