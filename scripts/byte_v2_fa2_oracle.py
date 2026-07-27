# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the ByteV2 FA2 loader against raw paged FA2."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from vllm.v1.attention.backends.byte_v2_layout import (
    ByteV2PageLayoutV6,
    ByteV2RawStagingLayout,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_fa2_decode_is_available,
    byte_v2_reshape_and_cache,
)


def _parse_seq_lens(value: str) -> tuple[int, ...]:
    try:
        seq_lens = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sequence lengths must be comma-separated integers"
        ) from error
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return seq_lens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--seq-lens",
        type=_parse_seq_lens,
        help="Comma-separated ragged sequence lengths; overrides --seq-len/--num-seqs.",
    )
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument(
        "--permute-pages",
        action="store_true",
        help="Map logical KV blocks to a deterministic non-monotonic physical order.",
    )
    parser.add_argument(
        "--shared-prefix-blocks",
        type=int,
        default=0,
        help="Share this many complete leading KV blocks across all sequences.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=0,
        help="Pass an explicit FA2 split count (0 keeps the FA2 heuristic).",
    )
    parser.add_argument("--byte-only", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--print-pointers", action="store_true")
    parser.add_argument("--force-outliers", action="store_true")
    fatal_group = parser.add_mutually_exclusive_group()
    fatal_group.add_argument(
        "--inject-overflow",
        action="store_true",
        help="Mark the first referenced V6 page overflowed before FA2 decode.",
    )
    fatal_group.add_argument(
        "--inject-fallback",
        action="store_true",
        help="Set a fallback bit on the first referenced V6 page before decode.",
    )
    fatal_group.add_argument(
        "--stress-pool-overflow",
        action="store_true",
        help="Construct one page whose outlier demand exceeds the V6 pool.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Run interleaved ByteV2/raw attention this many times.",
    )
    return parser.parse_args()


def fa2_oracle_ops_are_available() -> bool:
    """Return whether both ByteV2 and raw FA2 entry points are registered."""
    if not byte_v2_fa2_decode_is_available():
        return False
    op_namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    return (
        op_namespace is not None
        and getattr(op_namespace, "varlen_fwd", None) is not None
    )


def hybrid_fa2_oracle_ops_are_available() -> bool:
    """Return whether the hybrid ByteV2 and raw FA2 ops are registered."""
    if not fa2_oracle_ops_are_available():
        return False
    op_namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    return (
        op_namespace is not None
        and getattr(op_namespace, "byte_v2_hybrid_varlen_fwd", None) is not None
    )


def _physical_page_permutation(num_pages: int) -> list[int]:
    """Build a deterministic zig-zag permutation for address coverage."""
    physical_ids = []
    low = 0
    high = num_pages - 1
    while low <= high:
        physical_ids.append(high)
        high -= 1
        if low <= high:
            physical_ids.append(low)
            low += 1
    return physical_ids


def _make_block_table(
    seq_lens: Sequence[int],
    *,
    block_size: int,
    shared_prefix_blocks: int,
    permute_pages: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [(seq_len + block_size - 1) // block_size for seq_len in seq_lens]
    max_shared_blocks = min(seq_len // block_size for seq_len in seq_lens)
    if not 0 <= shared_prefix_blocks <= max_shared_blocks:
        raise ValueError(
            "--shared-prefix-blocks must be between 0 and the number of "
            "complete blocks in the shortest sequence"
        )

    logical_rows: list[list[int]] = []
    next_logical_page = shared_prefix_blocks
    for num_blocks in blocks_per_seq:
        private_blocks = num_blocks - shared_prefix_blocks
        row = list(range(shared_prefix_blocks))
        row.extend(range(next_logical_page, next_logical_page + private_blocks))
        logical_rows.append(row)
        next_logical_page += private_blocks

    num_pages = next_logical_page
    logical_to_physical = (
        _physical_page_permutation(num_pages)
        if permute_pages
        else list(range(num_pages))
    )
    max_blocks = max(blocks_per_seq)
    block_table_cpu = torch.zeros(
        (len(seq_lens), max_blocks),
        dtype=torch.int32,
    )
    for seq_idx, logical_row in enumerate(logical_rows):
        block_table_cpu[seq_idx, : len(logical_row)] = torch.tensor(
            [logical_to_physical[page] for page in logical_row],
            dtype=torch.int32,
        )
    return block_table_cpu.to(device=device), num_pages


def make_inputs(
    seq_lens: Sequence[int],
    *,
    query_len: int = 1,
    force_outliers: bool = False,
    force_pool_overflow: bool = False,
    permute_pages: bool = False,
    shared_prefix_blocks: int = 0,
) -> dict[str, torch.Tensor | float | int]:
    """Create matching compressed and raw paged-KV inputs for FA2."""
    seq_lens = tuple(seq_lens)
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must contain positive lengths")
    if query_len <= 0:
        raise ValueError("query_len must be positive")

    device = torch.device("cuda")
    layout = ByteV2PageLayoutV6()
    assert layout.page_size_bytes == 50560

    block_size = 16
    num_kv_heads = 8
    num_heads = 32
    head_dim = 128
    block_table, num_pages = _make_block_table(
        seq_lens,
        block_size=block_size,
        shared_prefix_blocks=shared_prefix_blocks,
        permute_pages=permute_pages,
        device=device,
    )

    used_slots = set()
    block_table_cpu = block_table.cpu()
    for seq_idx, seq_len in enumerate(seq_lens):
        for token_idx in range(seq_len):
            physical_page = int(block_table_cpu[seq_idx, token_idx // block_size])
            used_slots.add(physical_page * block_size + token_idx % block_size)
    slot_mapping = torch.tensor(
        sorted(used_slots),
        dtype=torch.int64,
        device=device,
    )
    total_k = slot_mapping.numel()

    num_kv_elements = total_k * num_kv_heads * head_dim
    if force_pool_overflow:
        bit_indices = torch.arange(
            num_kv_elements,
            dtype=torch.int32,
            device=device,
        )
        key_high7 = (bit_indices % 16) * 8
        value_high7 = ((bit_indices + 5) % 16) * 8
        lows = bit_indices % 251
        key = ((key_high7 << 8) | lows).to(torch.int16).view(torch.bfloat16)
        value = (
            ((value_high7 << 8) | ((lows + 17) % 251))
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        key = key.reshape(total_k, num_kv_heads, head_dim)
        value = value.reshape(total_k, num_kv_heads, head_dim)
    else:
        base = torch.arange(
            num_kv_elements,
            dtype=torch.float32,
            device=device,
        )
        if force_outliers:
            key_values = (base % 257) / 1024
            value_values = ((base + 17) % 263) / 1024
        else:
            key_values = ((base % 255) + 1) / 1024
            value_values = (((base + 17) % 255) + 1) / 1024
        key = key_values.reshape(total_k, num_kv_heads, head_dim).to(torch.bfloat16)
        value = value_values.reshape(total_k, num_kv_heads, head_dim).to(torch.bfloat16)

    byte_cache = torch.zeros(
        (num_pages, layout.page_size_bytes),
        dtype=torch.uint8,
        device=device,
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        byte_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    raw_key = torch.zeros(
        (num_pages, block_size, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    raw_value = torch.zeros_like(raw_key)
    raw_key.view(-1, num_kv_heads, head_dim)[slot_mapping] = key
    raw_value.view(-1, num_kv_heads, head_dim)[slot_mapping] = value

    total_q = len(seq_lens) * query_len
    query_base = torch.arange(
        total_q * num_heads * head_dim,
        dtype=torch.float32,
        device=device,
    )
    query = (
        ((query_base % 127) / 31)
        .reshape(total_q, num_heads, head_dim)
        .to(torch.bfloat16)
    )
    cu_seqlens_q = torch.arange(
        0,
        total_q + 1,
        query_len,
        dtype=torch.int32,
        device=device,
    )
    return {
        "query": query,
        "key": raw_key,
        "value": raw_value,
        "byte_cache": byte_cache,
        "block_table": block_table,
        "seq_lens": torch.tensor(
            seq_lens,
            dtype=torch.int32,
            device=device,
        ),
        "cu_seqlens_q": cu_seqlens_q,
        "dummy_cu_seqlens_k": torch.zeros_like(cu_seqlens_q),
        "scale": head_dim**-0.5,
        "max_seq_len": max(seq_lens),
    }


def add_hybrid_raw_pages(
    tensors: dict[str, torch.Tensor | float | int],
    raw_physical_pages: Sequence[int],
    *,
    reverse_raw_slots: bool = True,
    poison_compact_pages: bool = True,
) -> None:
    """Attach an authoritative raw sidecar for selected physical pages.

    The sidecar deliberately uses ByteV2's existing head-major staging
    layout, not raw FA2's row-major paged-cache layout. Reversing raw slots
    proves that the loader follows the side table rather than assuming that a
    physical page id is also a raw slot id. Selected compact pages are
    poisoned by default so an implementation that ignores the raw map cannot
    pass the oracle accidentally.
    """
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)

    pages = tuple(dict.fromkeys(int(page) for page in raw_physical_pages))
    num_pages = byte_cache.size(0)
    if any(page < 0 or page >= num_pages for page in pages):
        raise ValueError("raw physical page is outside the compact cache")

    staging_layout = ByteV2RawStagingLayout()
    assert staging_layout.slot_size_bytes == 65536
    assert staging_layout.value_base_bytes == 32768
    num_raw_slots = max(len(pages), 1)
    raw_staging_bf16 = torch.zeros(
        (num_raw_slots, 2, 8, 16, 128),
        dtype=torch.bfloat16,
        device=byte_cache.device,
    )
    page_to_raw_slot = torch.full(
        (num_pages,),
        -1,
        dtype=torch.int32,
        device=byte_cache.device,
    )
    slot_order = list(range(len(pages)))
    if reverse_raw_slots:
        slot_order.reverse()
    for physical_page, raw_slot in zip(pages, slot_order):
        raw_staging_bf16[raw_slot, 0].copy_(raw_key[physical_page].permute(1, 0, 2))
        raw_staging_bf16[raw_slot, 1].copy_(raw_value[physical_page].permute(1, 0, 2))
        page_to_raw_slot[physical_page] = raw_slot
    if poison_compact_pages:
        for physical_page in pages:
            byte_cache[physical_page].zero_()

    raw_staging = raw_staging_bf16.view(torch.uint8).reshape(
        num_raw_slots, staging_layout.slot_size_bytes
    )
    assert raw_staging.stride() == (staging_layout.slot_size_bytes, 1)
    assert raw_staging.data_ptr() % 16 == 0
    tensors["raw_staging"] = raw_staging
    tensors["page_to_raw_slot"] = page_to_raw_slot


def _run_attention_pair(
    tensors: dict[str, torch.Tensor | float | int],
    *,
    query_len: int,
    iterations: int,
    num_splits: int,
    run_byte: bool,
    run_raw: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    query = tensors["query"]
    max_seq_len = tensors["max_seq_len"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(max_seq_len, int)
    common = (
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        query_len,
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
    byte_out = byte_lse = None
    raw_out = raw_lse = None
    for _ in range(iterations):
        if run_byte:
            byte_out, byte_lse = torch.ops._vllm_fa2_C.byte_v2_varlen_fwd(
                query,
                tensors["byte_cache"],
                None,
                torch.empty_like(query),
                *common,
            )
        if run_raw:
            raw_out, raw_lse = torch.ops._vllm_fa2_C.varlen_fwd(
                query,
                tensors["key"],
                tensors["value"],
                torch.empty_like(query),
                *common,
            )
    torch.accelerator.synchronize()
    return byte_out, byte_lse, raw_out, raw_lse


def compare_byte_v2_and_raw(
    tensors: dict[str, torch.Tensor | float | int],
    *,
    query_len: int = 1,
    iterations: int = 1,
    num_splits: int = 0,
) -> dict[str, int | float | list[int]]:
    """Run interleaved ByteV2/raw FA2 and return bitwise mismatch counts."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 <= num_splits <= 128:
        raise ValueError("num_splits must be between 0 and 128")
    byte_out, byte_lse, raw_out, raw_lse = _run_attention_pair(
        tensors,
        query_len=query_len,
        iterations=iterations,
        num_splits=num_splits,
        run_byte=True,
        run_raw=True,
    )
    assert byte_out is not None and byte_lse is not None
    assert raw_out is not None and raw_lse is not None
    return _comparison_result(
        tensors,
        query_len=query_len,
        iterations=iterations,
        num_splits=num_splits,
        byte_out=byte_out,
        byte_lse=byte_lse,
        raw_out=raw_out,
        raw_lse=raw_lse,
    )


def compare_hybrid_byte_v2_and_raw(
    tensors: dict[str, torch.Tensor | float | int],
    *,
    query_len: int = 1,
    iterations: int = 1,
    num_splits: int = 0,
) -> dict[str, int | float | list[int]]:
    """Run hybrid ByteV2 and raw FA2 and return bitwise mismatch counts."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 <= num_splits <= 128:
        raise ValueError("num_splits must be between 0 and 128")
    query = tensors["query"]
    raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    max_seq_len = tensors["max_seq_len"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    assert isinstance(max_seq_len, int)
    common = (
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        query_len,
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
    hybrid_out = hybrid_lse = raw_out = raw_lse = None
    for _ in range(iterations):
        hybrid_out, hybrid_lse = torch.ops._vllm_fa2_C.byte_v2_hybrid_varlen_fwd(
            query,
            tensors["byte_cache"],
            raw_staging,
            page_to_raw_slot,
            None,
            torch.empty_like(query),
            *common,
        )
        raw_out, raw_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            query,
            tensors["key"],
            tensors["value"],
            torch.empty_like(query),
            *common,
        )
    torch.accelerator.synchronize()
    assert hybrid_out is not None and hybrid_lse is not None
    assert raw_out is not None and raw_lse is not None
    return _comparison_result(
        tensors,
        query_len=query_len,
        iterations=iterations,
        num_splits=num_splits,
        byte_out=hybrid_out,
        byte_lse=hybrid_lse,
        raw_out=raw_out,
        raw_lse=raw_lse,
    )


def _comparison_result(
    tensors: dict[str, torch.Tensor | float | int],
    *,
    query_len: int,
    iterations: int,
    num_splits: int,
    byte_out: torch.Tensor,
    byte_lse: torch.Tensor,
    raw_out: torch.Tensor,
    raw_lse: torch.Tensor,
) -> dict[str, int | float | list[int]]:
    seq_lens = tensors["seq_lens"]
    assert isinstance(seq_lens, torch.Tensor)
    return {
        "iterations": iterations,
        "query_len": query_len,
        "seq_lens": seq_lens.cpu().tolist(),
        "num_splits": num_splits,
        "out_mismatch": int(
            (byte_out.view(torch.int16) != raw_out.view(torch.int16)).sum()
        ),
        "lse_mismatch": int(
            (byte_lse.view(torch.int32) != raw_lse.view(torch.int32)).sum()
        ),
        "out_max_abs": float((byte_out.float() - raw_out.float()).abs().max()),
        "lse_max_abs": float((byte_lse - raw_lse).abs().max()),
    }


def main() -> None:
    args = parse_args()
    if args.byte_only and args.raw_only:
        raise ValueError("--byte-only and --raw-only are mutually exclusive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.seq_lens is None:
        if args.seq_len <= 0 or args.num_seqs <= 0:
            raise ValueError("--seq-len and --num-seqs must be positive")
        seq_lens = (args.seq_len,) * args.num_seqs
    else:
        seq_lens = args.seq_lens
    if not 0 <= args.num_splits <= 128:
        raise ValueError("--num-splits must be between 0 and 128")
    fatal_mode = None
    if args.inject_overflow:
        fatal_mode = "overflow-marker"
    elif args.inject_fallback:
        fatal_mode = "fallback-bit"
    elif args.stress_pool_overflow:
        fatal_mode = "writer-pool-overflow"
    if fatal_mode is not None and not args.byte_only:
        raise ValueError("fatal-page probes require --byte-only")
    if not fa2_oracle_ops_are_available():
        raise RuntimeError("ByteV2 and raw FA2 extension ops must be registered")
    if fatal_mode is not None:
        print({"fatal_test": fatal_mode}, flush=True)
    tensors = make_inputs(
        seq_lens,
        query_len=args.query_len,
        force_outliers=args.force_outliers,
        force_pool_overflow=args.stress_pool_overflow,
        permute_pages=args.permute_pages,
        shared_prefix_blocks=args.shared_prefix_blocks,
    )
    if args.inject_overflow or args.inject_fallback:
        byte_cache = tensors["byte_cache"]
        block_table = tensors["block_table"]
        assert isinstance(byte_cache, torch.Tensor)
        assert isinstance(block_table, torch.Tensor)
        first_page = block_table[0, 0].to(torch.int64)
        layout = ByteV2PageLayoutV6()
        if args.inject_overflow:
            byte_cache[first_page, layout.outlier_pool_overflow_offset] = 1
        else:
            byte_cache[first_page, layout.k_fallback_mask_offset(kv_head=0)] = 1
    query = tensors["query"]
    assert isinstance(query, torch.Tensor)
    if args.print_pointers:
        pointers = {
            name: hex(tensor.data_ptr())
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor)
        }
        print(pointers, flush=True)

    byte_out, byte_lse, raw_out, raw_lse = _run_attention_pair(
        tensors,
        query_len=args.query_len,
        iterations=args.iterations,
        num_splits=args.num_splits,
        run_byte=not args.raw_only,
        run_raw=not args.byte_only,
    )
    if args.print_pointers and byte_out is not None:
        print({"byte_out": hex(byte_out.data_ptr())}, flush=True)
    if args.byte_only:
        print("ByteV2 FA2 completed")
        return
    if args.raw_only:
        print("Raw FA2 completed")
        return
    assert byte_out is not None and byte_lse is not None
    assert raw_out is not None and raw_lse is not None
    result = _comparison_result(
        tensors,
        query_len=args.query_len,
        iterations=args.iterations,
        num_splits=args.num_splits,
        byte_out=byte_out,
        byte_lse=byte_lse,
        raw_out=raw_out,
        raw_lse=raw_lse,
    )
    result["permuted_pages"] = int(args.permute_pages)
    result["shared_prefix_blocks"] = args.shared_prefix_blocks
    print(result)


if __name__ == "__main__":
    main()
