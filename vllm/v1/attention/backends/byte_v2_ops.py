# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrappers for experimental ByteV2 custom ops."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_MISSING_KERNEL_MSG = (
    "ByteV2 custom kernels are not registered in this checkout. "
    "Build and register the ByteV2 cache and decode kernels before enabling "
    "the ByteV2 backend."
)
_REQUIRED_BYTE_V2_CUSTOM_OPS = (
    ("_C_cache_ops", "byte_v2_reshape_and_cache"),
    ("_C_cache_ops", "byte_v2_reshape_and_cache_high_byte"),
    ("_C_cache_ops", "byte_v2_reshape_and_cache_sideband_high"),
    ("_C_cache_ops", "byte_v2_update_cache_single_token_fused"),
    ("_C_cache_ops", "byte_v2_append_raw_staging"),
    ("_C_cache_ops", "byte_v2_prepare_raw_staging"),
    ("_C_cache_ops", "byte_v2_hydrate_raw_staging_from_cache"),
    ("_C_cache_ops", "byte_v2_release_raw_staging"),
    ("_C_cache_ops", "byte_v2_release_raw_staging_and_update_flags"),
    ("_C_cache_ops", "byte_v2_commit_raw_staging_to_cache"),
    ("_C_cache_ops", "byte_v2_collect_cache_stats"),
    ("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"),
    ("_C", "byte_v2_paged_decode_attention"),
    ("_C", "byte_v2_paged_decode_attention_split_k"),
    ("_C", "byte_v2_paged_decode_attention_split_k_guarded"),
    ("_C", "byte_v2_speculative_verify_q4"),
    ("_C", "byte_v2_speculative_verify_gqa"),
    ("_C", "byte_v2_prefill_attention"),
)
_CUSTOM_OPS_LOAD_ATTEMPTED = False


def _find_op(namespace: str, op_name: str):
    _ensure_custom_ops_loaded()
    op_namespace = getattr(torch.ops, namespace, None)
    return getattr(op_namespace, op_name, None) if op_namespace is not None else None


def _ensure_custom_ops_loaded() -> None:
    global _CUSTOM_OPS_LOAD_ATTEMPTED
    if _CUSTOM_OPS_LOAD_ATTEMPTED:
        return
    _CUSTOM_OPS_LOAD_ATTEMPTED = True
    try:
        import vllm._C  # noqa: F401
        import vllm._C_stable_libtorch  # noqa: F401
    except ImportError:
        pass


def _require_op(namespace: str, op_name: str):
    op = _find_op(namespace, op_name)
    if op is None:
        raise NotImplementedError(_MISSING_KERNEL_MSG)
    return op


def byte_v2_custom_ops_are_available() -> bool:
    """Return whether all ByteV2 custom ops needed by the backend exist."""
    return all(
        _find_op(namespace, op_name) is not None
        for namespace, op_name in _REQUIRED_BYTE_V2_CUSTOM_OPS
    )


def missing_byte_v2_custom_ops() -> tuple[str, ...]:
    """Return the missing ByteV2 custom op qualified names."""
    return tuple(
        f"torch.ops.{namespace}.{op_name}"
        for namespace, op_name in _REQUIRED_BYTE_V2_CUSTOM_OPS
        if _find_op(namespace, op_name) is None
    )


def byte_v2_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Encode K/V tensors into the ByteV2 compressed cache."""
    _require_op("_C_cache_ops", "byte_v2_reshape_and_cache")(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_reshape_and_cache_high_byte(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Encode K/V tensors into the ByteV2 high-byte upper-bound cache."""
    _require_op("_C_cache_ops", "byte_v2_reshape_and_cache_high_byte")(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_reshape_and_cache_sideband_high(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Encode K/V tensors with tile-local high-byte sideband for outlier tiles."""
    _require_op("_C_cache_ops", "byte_v2_reshape_and_cache_sideband_high")(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_update_cache_single_token(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    page_unsafe_flags: torch.Tensor | None = None,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Fast-path ByteV2 cache update for one decode token."""
    if page_unsafe_flags is None:
        page_unsafe_flags = torch.empty(
            (0,),
            dtype=torch.int32,
            device=kv_cache.device,
        )
    _require_op("_C_cache_ops", "byte_v2_update_cache_single_token_fused")(
        key,
        value,
        kv_cache,
        slot_mapping,
        page_unsafe_flags,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_append_raw_staging(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Append raw bf16 K/V bits into ByteV2 partial-block staging slots."""
    _require_op("_C_cache_ops", "byte_v2_append_raw_staging")(
        key,
        value,
        raw_staging,
        slot_mapping,
        block_to_staging_slot,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_prepare_raw_staging(
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    *,
    alloc_block_tokens: int,
) -> None:
    """Build ByteV2 raw staging slot maps from a device slot mapping."""
    _require_op("_C_cache_ops", "byte_v2_prepare_raw_staging")(
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        alloc_block_tokens,
    )


def byte_v2_hydrate_raw_staging_from_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Hydrate staging slots with existing ByteV2 cache rows before append."""
    _require_op("_C_cache_ops", "byte_v2_hydrate_raw_staging_from_cache")(
        raw_staging,
        kv_cache,
        staging_to_physical_block,
        valid_rows,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_release_raw_staging(
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
) -> None:
    """Release ByteV2 raw staging slot maps for reuse."""
    _require_op("_C_cache_ops", "byte_v2_release_raw_staging")(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
    )


def byte_v2_release_raw_staging_and_update_flags(
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    tile_policy: Sequence[int],
) -> None:
    """Refresh touched page flags while releasing raw staging slots."""
    _require_op("_C_cache_ops", "byte_v2_release_raw_staging_and_update_flags")(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_unsafe_flags,
        kv_cache,
        list(tile_policy),
    )


def byte_v2_commit_raw_staging_to_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Finalize raw staging slots into the ByteV2 compressed V4 cache."""
    _require_op("_C_cache_ops", "byte_v2_commit_raw_staging_to_cache")(
        raw_staging,
        kv_cache,
        staging_to_physical_block,
        valid_rows,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_update_cache_raw_staging(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    *,
    tile_policy: Sequence[int],
    fuse_metadata_clear: bool = False,
    bypass_serial_metadata: bool = False,
    warp_parallel_histogram: bool = False,
) -> None:
    """Run the raw-staging cache update through one native dispatcher call."""
    _require_op("_C_cache_ops", "byte_v2_update_cache_raw_staging")(
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
        list(tile_policy),
        fuse_metadata_clear,
        bypass_serial_metadata,
        warp_parallel_histogram,
    )


def byte_v2_collect_cache_stats(
    stats: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int,
    tile_policy: Sequence[int],
) -> None:
    """Collect ByteV2 cache metadata stats for the referenced pages."""
    _require_op("_C_cache_ops", "byte_v2_collect_cache_stats")(
        stats,
        kv_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        list(tile_policy),
    )


def byte_v2_update_cache_unsafe_flags(
    page_unsafe_flags: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    tile_policy: Sequence[int],
) -> None:
    """Refresh unsafe-page flags for the ByteV2 cache blocks touched by slots."""
    _require_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags")(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        list(tile_policy),
    )


def byte_v2_paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    tile_policy: Sequence[int],
) -> None:
    """Run ByteV2 compressed-cache paged decode attention."""
    _require_op("_C", "byte_v2_paged_decode_attention")(
        output,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
        max_seq_len,
        list(tile_policy),
    )


def byte_v2_paged_decode_attention_split_k(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    partition_size: int,
    tile_policy: Sequence[int],
) -> None:
    """Run ByteV2 split-k paged decode attention with reusable workspace."""
    _require_op("_C", "byte_v2_paged_decode_attention_split_k")(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
        max_seq_len,
        partition_size,
        list(tile_policy),
    )


def byte_v2_paged_decode_attention_split_k_guarded(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    partition_size: int,
    tile_policy: Sequence[int],
) -> None:
    """Run split-k decode using fast loads guarded by per-page unsafe flags."""
    _require_op("_C", "byte_v2_paged_decode_attention_split_k_guarded")(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
        max_seq_len,
        partition_size,
        list(tile_policy),
    )


def byte_v2_speculative_verify_q4(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    partition_size: int,
    tile_policy: Sequence[int],
) -> None:
    """Verify four shared-prefix queries with one FA2-direct M16 launch."""
    _require_op("_C", "byte_v2_speculative_verify_q4")(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
        max_seq_len,
        partition_size,
        list(tile_policy),
    )


def byte_v2_speculative_verify_gqa(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    speculative_query_len: int,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    partition_size: int,
    tile_policy: Sequence[int],
) -> None:
    """Run a uniform Q2/Q4/Q8/Q16 batch with shared GQA KV decode."""
    _require_op("_C", "byte_v2_speculative_verify_gqa")(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        speculative_query_len,
        scale,
        num_kv_heads,
        block_size,
        max_seq_len,
        partition_size,
        list(tile_policy),
    )


def byte_v2_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    max_query_len: int,
    scale: float,
    num_kv_heads: int,
    causal: bool,
    tile_policy: Sequence[int],
) -> None:
    """Run ByteV2 native raw-QKV prefill attention."""
    _require_op("_C", "byte_v2_prefill_attention")(
        output,
        query,
        key,
        value,
        query_start_loc,
        max_query_len,
        scale,
        num_kv_heads,
        causal,
        list(tile_policy),
    )
