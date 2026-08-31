# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrappers for experimental ByteV2 custom ops."""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Sequence
from contextlib import suppress

import torch

_MISSING_KERNEL_MSG = (
    "ByteV2 custom kernels are not registered in this checkout. "
    "Build and register the ByteV2 cache and decode kernels before enabling "
    "the ByteV2 backend."
)
_MISSING_FA2_KERNEL_MSG = (
    "ByteV2 FA2 decode is not registered in this checkout. Build the "
    "vLLM FlashAttention-2 extension with ByteV2 support before selecting "
    "BYTE_V2_DECODE_KERNEL=fa2."
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
    ("_C", "byte_v2_speculative_verify_ragged_q4"),
    ("_C", "byte_v2_prefill_attention"),
)
_CUSTOM_OPS_LOAD_ATTEMPTED = False
_FA2_OPS_LOAD_ATTEMPTED = False
_STATIC_W16_FA2_PROFILE_SEQ_RANGE_ENV = "BYTE_V2_STATIC_W16_FA2_PROFILE_SEQ_RANGE"
_STATIC_W16_FA2_BASELINE_SPLITS = 20
_STATIC_W16_FA2_BLOCK_N = 128


def _as_float32(value: float) -> float:
    """Round a Python float exactly as a C++ ``float`` assignment does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _fa2_num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Mirror FA2's host-side split-K occupancy heuristic."""
    almost_full = _as_float32(_as_float32(0.8) * num_sms)
    if batch_nheads_mblocks >= almost_full:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    efficiencies = [0.0] * max_splits
    max_efficiency = 0.0
    for num_splits in range(1, max_splits + 1):
        blocks_per_split = (num_n_blocks + num_splits - 1) // num_splits
        previous_blocks_per_split = (
            (num_n_blocks + num_splits - 2) // (num_splits - 1)
            if num_splits > 1
            else None
        )
        if num_splits > 1 and blocks_per_split == previous_blocks_per_split:
            continue
        n_waves = _as_float32(_as_float32(batch_nheads_mblocks * num_splits) / num_sms)
        efficiency = _as_float32(n_waves / math.ceil(n_waves))
        efficiencies[num_splits - 1] = efficiency
        max_efficiency = max(max_efficiency, efficiency)

    for num_splits, efficiency in enumerate(efficiencies, start=1):
        blocks_per_split = (num_n_blocks + num_splits - 1) // num_splits
        previous_blocks_per_split = (
            (num_n_blocks + num_splits - 2) // (num_splits - 1)
            if num_splits > 1
            else None
        )
        eligible = num_splits == 1 or blocks_per_split != previous_blocks_per_split
        if eligible and efficiency >= 0.85 * max_efficiency:
            return num_splits
    return 1


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


def _ensure_fa2_ops_loaded() -> None:
    global _FA2_OPS_LOAD_ATTEMPTED
    if _FA2_OPS_LOAD_ATTEMPTED:
        return
    _FA2_OPS_LOAD_ATTEMPTED = True
    with suppress(ImportError, OSError):
        import vllm.vllm_flash_attn  # noqa: F401


def _require_op(namespace: str, op_name: str):
    op = _find_op(namespace, op_name)
    if op is None:
        raise NotImplementedError(_MISSING_KERNEL_MSG)
    return op


def _op_has_schema_argument(op, argument_name: str) -> bool:
    try:
        arguments = op.default._schema.arguments
    except (AttributeError, RuntimeError):
        return False
    return any(argument.name == argument_name for argument in arguments)


def _static_w8_bases_enabled(k_base: int, v_base: int) -> bool:
    if k_base == -1 and v_base == -1:
        return False
    if 0 <= k_base <= 120 and 0 <= v_base <= 120:
        return True
    raise ValueError(
        "ByteV2 Static-W8 bases must both be -1 (Dynamic-W8) or integers "
        f"in [0, 120], got K={k_base}, V={v_base}"
    )


def _find_fa2_op(op_name: str):
    _ensure_fa2_ops_loaded()
    op_namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    return getattr(op_namespace, op_name, None) if op_namespace is not None else None


def _require_fa2_op(op_name: str):
    op = _find_fa2_op(op_name)
    if op is None:
        raise NotImplementedError(_MISSING_FA2_KERNEL_MSG)
    return op


def _static_w16_fa2_profile_seq_range() -> tuple[int, int] | None:
    value = os.environ.get(_STATIC_W16_FA2_PROFILE_SEQ_RANGE_ENV)
    if value is None:
        return None
    pieces = value.split(":")
    try:
        seq_range = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise RuntimeError(
            f"{_STATIC_W16_FA2_PROFILE_SEQ_RANGE_ENV} must be MIN:MAX"
        ) from error
    if len(seq_range) != 2:
        raise RuntimeError(f"{_STATIC_W16_FA2_PROFILE_SEQ_RANGE_ENV} must be MIN:MAX")
    min_seq_len, max_seq_len = seq_range
    if min_seq_len <= 0 or max_seq_len < min_seq_len:
        raise RuntimeError(
            f"{_STATIC_W16_FA2_PROFILE_SEQ_RANGE_ENV} must contain a positive "
            "inclusive range"
        )
    return min_seq_len, max_seq_len


def _static_w16_fa2_profile_split_for_shape(
    profile_seq_range: tuple[int, int] | None,
    *,
    device_name: str,
    num_sms: int,
    query_shape: tuple[int, ...],
    query_is_bf16: bool,
    query_is_cuda: bool,
    block_table_shape: tuple[int, ...],
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool,
) -> int:
    """Trim only empty Q1 split-K CTAs, or keep FA2 auto.

    The profile range is an explicit CUDA Graph runtime assertion. A split
    override is returned only when every N=128 tile count in that range keeps
    the same blocks-per-split value as Raw-FA2's automatic launch. Therefore,
    all nonempty split boundaries and partial reductions remain unchanged.
    """
    if profile_seq_range is None:
        return 0
    if len(query_shape) != 3:
        return 0
    batch_size, num_query_heads, head_size = query_shape
    if (
        device_name != "NVIDIA A40"
        or num_sms != 84
        or batch_size <= 0
        or num_query_heads != 32
        or head_size != 128
        or not query_is_bf16
        or not query_is_cuda
        or len(block_table_shape) != 2
        or block_table_shape[0] != batch_size
        or max_query_len != 1
        or not causal
        or preserve_mixed_dispatch
        or max_seq_len <= 0
    ):
        return 0

    min_seq_len, profile_max_seq_len = profile_seq_range
    block_table_capacity = block_table_shape[1] * 16
    if profile_max_seq_len > max_seq_len or profile_max_seq_len > block_table_capacity:
        return 0

    min_tiles = (min_seq_len + _STATIC_W16_FA2_BLOCK_N - 1) // (_STATIC_W16_FA2_BLOCK_N)
    max_tiles = (profile_max_seq_len + _STATIC_W16_FA2_BLOCK_N - 1) // (
        _STATIC_W16_FA2_BLOCK_N
    )

    # Q1 GQA4 is transformed by FA2 from 32 query heads to 8 heads with a
    # logical query length of four. That leaves one M=64 block per KV head.
    if batch_size == 1 and 32768 <= max_seq_len <= 131072:
        baseline_splits = _STATIC_W16_FA2_BASELINE_SPLITS
    else:
        max_n_blocks = (max_seq_len + _STATIC_W16_FA2_BLOCK_N - 1) // (
            _STATIC_W16_FA2_BLOCK_N
        )
        baseline_splits = _fa2_num_splits_heuristic(
            batch_size * 8,
            num_sms * 2,
            max_n_blocks,
            128,
        )
    if baseline_splits <= 1:
        return 0

    min_blocks_per_split = (min_tiles + baseline_splits - 1) // baseline_splits
    max_blocks_per_split = (max_tiles + baseline_splits - 1) // baseline_splits
    if min_blocks_per_split != max_blocks_per_split:
        return 0

    trimmed_splits = (max_tiles + max_blocks_per_split - 1) // max_blocks_per_split
    # Keep the split-K combine topology and only remove trailing empty slots.
    trimmed_splits = max(2, trimmed_splits)
    return trimmed_splits if trimmed_splits < baseline_splits else 0


def _static_w16_fa2_profile_num_splits(
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool,
) -> int:
    """Resolve the opt-in Static-W16 split override without a device sync."""
    profile_seq_range = _static_w16_fa2_profile_seq_range()
    if profile_seq_range is None:
        return 0
    query_is_cuda = query.is_cuda
    device_name = ""
    num_sms = 0
    if query_is_cuda:
        properties = torch.cuda.get_device_properties(query.device)
        device_name = properties.name
        num_sms = properties.multi_processor_count
    return _static_w16_fa2_profile_split_for_shape(
        profile_seq_range,
        device_name=device_name,
        num_sms=num_sms,
        query_shape=tuple(query.shape),
        query_is_bf16=query.dtype == torch.bfloat16,
        query_is_cuda=query_is_cuda,
        block_table_shape=tuple(block_tables.shape),
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        causal=causal,
        preserve_mixed_dispatch=preserve_mixed_dispatch,
    )


def byte_v2_custom_ops_are_available() -> bool:
    """Return whether all ByteV2 custom ops needed by the backend exist."""
    return all(
        _find_op(namespace, op_name) is not None
        for namespace, op_name in _REQUIRED_BYTE_V2_CUSTOM_OPS
    )


def byte_v2_fa2_decode_is_available() -> bool:
    """Return whether the optional ByteV2 FA2 decode entry point exists."""
    return _find_fa2_op("byte_v2_varlen_fwd") is not None


def byte_v2_fa2_hybrid_decode_is_available() -> bool:
    """Return whether the experimental compact/raw FA2 entry point exists."""
    return _find_fa2_op("byte_v2_hybrid_varlen_fwd") is not None


def byte_v2_static_w16_runtime_is_available() -> bool:
    """Return whether the experimental W16 reader and online writer exist."""
    cache_ops = (
        "byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache",
        "byte_v2_static_w16_commit_raw_staging_to_hybrid_cache",
        "byte_v2_static_w16_update_hybrid_cache_raw_tail_q1",
    )
    return _find_fa2_op("static_w16_canonical_varlen_fwd") is not None and all(
        _find_op("_C_cache_ops", op_name) is not None for op_name in cache_ops
    )


def byte_v2_static_w16_safe_full_page_retention_is_available() -> bool:
    """Return whether the W16 commit op can retain safe full raw pages."""
    op = _find_op(
        "_C_cache_ops",
        "byte_v2_static_w16_commit_raw_staging_to_hybrid_cache",
    )
    return op is not None and _op_has_schema_argument(
        op,
        "retain_safe_full_pages",
    )


def byte_v2_hybrid_cache_update_is_available() -> bool:
    """Return whether persistent raw fallback update ops are registered."""
    return all(
        _find_op("_C_cache_ops", op_name) is not None
        for op_name in (
            "byte_v2_hydrate_raw_staging_from_hybrid_cache",
            "byte_v2_commit_raw_staging_to_hybrid_cache",
            "byte_v2_update_hybrid_cache_raw_staging_q1",
            "byte_v2_reset_raw_fallback_pages",
        )
    )


def byte_v2_batched_raw_fallback_reset_is_available() -> bool:
    """Return whether the optional cross-layer reset entry point exists."""
    return (
        _find_op(
            "_C_cache_ops",
            "byte_v2_reset_raw_fallback_pages_batched",
        )
        is not None
    )


def byte_v2_hybrid_raw_tail_q1_is_available() -> bool:
    """Return whether raw-tail Q1 and safe Q>1 demotion are registered."""
    q1_op = _find_op(
        "_C_cache_ops",
        "byte_v2_update_hybrid_cache_raw_tail_q1",
    )
    multi_token_op = _find_op(
        "_C_cache_ops",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
    )
    if q1_op is None or multi_token_op is None:
        return False

    return _op_has_schema_argument(
        q1_op,
        "fuse_commit_finalize",
    ) and _op_has_schema_argument(
        multi_token_op,
        "demote_safe_raw_pages",
    )


def byte_v2_static_w8_writer_is_available() -> bool:
    """Return whether every production compact writer accepts frozen bases."""
    op_names = (
        "byte_v2_update_cache_raw_staging",
        "byte_v2_update_hybrid_cache_raw_staging_q1",
        "byte_v2_update_hybrid_cache_raw_tail_q1",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token_retained",
    )
    for op_name in op_names:
        op = _find_op("_C_cache_ops", op_name)
        if op is None:
            return False
        if not _op_has_schema_argument(op, "static_k_high7_base"):
            return False
        if not _op_has_schema_argument(op, "static_v_high7_base"):
            return False
    return True


def byte_v2_test_forced_raw_promotion_is_available() -> bool:
    """Return whether the test-only forced raw-promotion op is registered."""
    return (
        _find_op(
            "_C_cache_ops",
            "byte_v2_test_force_promote_raw_staging_q1",
        )
        is not None
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


def byte_v2_hydrate_raw_staging_from_hybrid_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Hydrate transient pages from compact or authoritative raw storage."""
    _require_op("_C_cache_ops", "byte_v2_hydrate_raw_staging_from_hybrid_cache")(
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        page_to_raw_slot,
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


def byte_v2_commit_raw_staging_to_hybrid_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    codec_token_block: int,
    codec_dim_block: int,
    alloc_block_tokens: int,
) -> None:
    """Commit compact pages and publish exact raw fallback pages atomically."""
    _require_op("_C_cache_ops", "byte_v2_commit_raw_staging_to_hybrid_cache")(
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        staging_to_physical_block,
        valid_rows,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    )


def byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    staging_to_physical: torch.Tensor,
    valid_rows: torch.Tensor,
    fatal: torch.Tensor,
) -> None:
    """Materialize exact W16 compact/raw pages into transient raw staging."""
    _require_op(
        "_C_cache_ops",
        "byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache",
    )(
        raw_staging,
        kv_cache,
        raw_pages,
        page_to_raw_slot,
        staging_to_physical,
        valid_rows,
        fatal,
    )


def byte_v2_static_w16_commit_raw_staging_to_hybrid_cache(
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_slots: torch.Tensor,
    free_count: torch.Tensor,
    fatal: torch.Tensor,
    staging_to_physical: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    k_base: int,
    v_base: int,
    retain_safe_full_pages: bool = False,
) -> None:
    """Seal full W16 pages and persist partial/overflow pages exactly."""
    op = _require_op(
        "_C_cache_ops",
        "byte_v2_static_w16_commit_raw_staging_to_hybrid_cache",
    )
    args = (
        raw_staging,
        kv_cache,
        raw_pages,
        page_to_raw_slot,
        free_slots,
        free_count,
        fatal,
        staging_to_physical,
        valid_rows,
        k_base,
        v_base,
    )
    if retain_safe_full_pages:
        if not _op_has_schema_argument(op, "retain_safe_full_pages"):
            raise NotImplementedError(
                "ByteV2 Static-W16 safe full-page retention is unavailable"
            )
        op(*args, True)
    else:
        # Preserve compatibility with the pre-retention schema.
        op(*args)


def byte_v2_static_w16_update_hybrid_cache_raw_tail_q1(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_slots: torch.Tensor,
    free_count: torch.Tensor,
    fatal: torch.Tensor,
    *,
    k_base: int,
    v_base: int,
) -> None:
    """Append one Q1 token per request and seal tails that reach row 15."""
    _require_op(
        "_C_cache_ops",
        "byte_v2_static_w16_update_hybrid_cache_raw_tail_q1",
    )(
        key,
        value,
        kv_cache,
        raw_pages,
        slot_mapping,
        page_to_raw_slot,
        free_slots,
        free_count,
        fatal,
        k_base,
        v_base,
    )


def byte_v2_reset_raw_fallback_pages(
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    physical_block_ids: torch.Tensor,
) -> None:
    """Unpublish raw pages and return their slots to the free stack."""
    _require_op("_C_cache_ops", "byte_v2_reset_raw_fallback_pages")(
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        physical_block_ids,
    )


def byte_v2_reset_raw_fallback_pages_batched(
    page_to_raw_slots: list[torch.Tensor],
    free_raw_slots: list[torch.Tensor],
    free_raw_slot_counts: list[torch.Tensor],
    raw_pool_overflows: list[torch.Tensor],
    physical_block_ids: torch.Tensor,
) -> None:
    """Release selected physical pages across multiple layer sidecars."""
    _require_op(
        "_C_cache_ops",
        "byte_v2_reset_raw_fallback_pages_batched",
    )(
        page_to_raw_slots,
        free_raw_slots,
        free_raw_slot_counts,
        raw_pool_overflows,
        physical_block_ids,
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
    fuse_single_token_staging: bool = False,
    fuse_single_token_commit_release: bool = False,
    fuse_single_token_stage_metadata_clear: bool = False,
    static_k_high7_base: int = -1,
    static_v_high7_base: int = -1,
) -> None:
    """Run the raw-staging cache update through one native dispatcher call."""
    op = _require_op("_C_cache_ops", "byte_v2_update_cache_raw_staging")
    args = (
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
        fuse_single_token_staging,
        fuse_single_token_commit_release,
        fuse_single_token_stage_metadata_clear,
    )
    if _static_w8_bases_enabled(static_k_high7_base, static_v_high7_base):
        op(*args, static_k_high7_base, static_v_high7_base)
    else:
        op(*args)


def byte_v2_update_hybrid_cache_raw_staging_q1(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    *,
    tile_policy: Sequence[int],
    page_unsafe_flags: torch.Tensor | None = None,
    static_k_high7_base: int = -1,
    static_v_high7_base: int = -1,
) -> None:
    """Run the exact compact/raw hybrid Q1 update in three CUDA kernels."""
    op = _require_op("_C_cache_ops", "byte_v2_update_hybrid_cache_raw_staging_q1")
    args = (
        key,
        value,
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        list(tile_policy),
        page_unsafe_flags,
    )
    if _static_w8_bases_enabled(static_k_high7_base, static_v_high7_base):
        op(*args, static_k_high7_base, static_v_high7_base)
    else:
        op(*args)


def byte_v2_update_hybrid_cache_raw_tail_q1(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    *,
    tile_policy: Sequence[int],
    page_unsafe_flags: torch.Tensor | None = None,
    fuse_commit_finalize: bool = False,
    static_k_high7_base: int = -1,
    static_v_high7_base: int = -1,
) -> None:
    """Append Q1 to a raw tail and seal a full page on the current CUDA stream.

    Update and attention must stay ordered on that stream; this experimental
    path does not synchronize concurrent cache readers on other streams.
    """
    op = _require_op("_C_cache_ops", "byte_v2_update_hybrid_cache_raw_tail_q1")
    args = (
        key,
        value,
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        list(tile_policy),
        page_unsafe_flags,
    )
    if _static_w8_bases_enabled(static_k_high7_base, static_v_high7_base):
        op(
            *args,
            fuse_commit_finalize,
            static_k_high7_base,
            static_v_high7_base,
        )
    elif fuse_commit_finalize:
        op(*args, True)
    else:
        # Preserve compatibility with the pre-fused-finalize schema. The
        # capability gate requires the trailing bool before enabling raw-tail
        # Q1, but direct wrapper callers may still use the legacy operation.
        op(*args)


def byte_v2_update_hybrid_cache_raw_staging_multi_token(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    *,
    tile_policy: Sequence[int],
    page_unsafe_flags: torch.Tensor | None = None,
    demote_safe_raw_pages: bool = False,
    static_k_high7_base: int = -1,
    static_v_high7_base: int = -1,
) -> None:
    """Run an exact compact/raw hybrid multi-token cache update."""
    op = _require_op(
        "_C_cache_ops",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
    )
    args = (
        key,
        value,
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        list(tile_policy),
        page_unsafe_flags,
    )
    if _static_w8_bases_enabled(static_k_high7_base, static_v_high7_base):
        op(
            *args,
            demote_safe_raw_pages,
            static_k_high7_base,
            static_v_high7_base,
        )
    elif demote_safe_raw_pages:
        op(*args, True)
    else:
        # Preserve compatibility with the pre-demotion schema. The capability
        # gate keeps raw-tail Q1 disabled unless the trailing bool is present.
        op(*args)


def byte_v2_update_hybrid_cache_raw_staging_multi_token_retained(
    key: torch.Tensor,
    value: torch.Tensor,
    raw_staging: torch.Tensor,
    kv_cache: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    staging_to_physical_block: torch.Tensor,
    valid_rows: torch.Tensor,
    next_staging_slot: torch.Tensor,
    overflow: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    *,
    tile_policy: Sequence[int],
    page_unsafe_flags: torch.Tensor | None = None,
    static_k_high7_base: int = -1,
    static_v_high7_base: int = -1,
) -> None:
    """Retain transient pages after the hybrid multi-token cache commit."""
    op = _require_op(
        "_C_cache_ops",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token_retained",
    )
    args = (
        key,
        value,
        raw_staging,
        kv_cache,
        persistent_raw_staging,
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        list(tile_policy),
        page_unsafe_flags,
    )
    if _static_w8_bases_enabled(static_k_high7_base, static_v_high7_base):
        op(*args, static_k_high7_base, static_v_high7_base)
    else:
        op(*args)


def byte_v2_test_force_promote_raw_staging_q1(
    raw_staging: torch.Tensor,
    persistent_raw_staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    free_raw_slots: torch.Tensor,
    free_raw_slot_count: torch.Tensor,
    raw_pool_overflow: torch.Tensor,
    diagnostic: torch.Tensor,
) -> None:
    """Force one diagnostic Q1 page into the persistent raw sidecar."""
    _require_op(
        "_C_cache_ops",
        "byte_v2_test_force_promote_raw_staging_q1",
    )(
        raw_staging,
        persistent_raw_staging,
        slot_mapping,
        page_to_raw_slot,
        free_raw_slots,
        free_raw_slot_count,
        raw_pool_overflow,
        diagnostic,
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


def byte_v2_fa2_paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_seq_len: int,
    causal: bool,
) -> None:
    """Run Q1 ByteV2 decode through the original FA2 split-KV template.

    The extension only replaces FA2's global-to-shared K/V copy policy with
    ByteV2 decode. QK, softmax, PV, split selection, and combine remain FA2.
    """
    _require_fa2_op("byte_v2_varlen_fwd")(
        query,
        kv_cache,
        None,
        output,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        block_tables,
        None,
        1,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )


def byte_v2_fa2_raw_staging_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    raw_staging: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    block_tables_are_staging_slots: bool = False,
) -> None:
    """Run original raw paged FA2 directly over exact staging pages."""
    expected_slot_bytes = 2 * 2 * num_kv_heads * block_size * head_dim
    if (
        raw_staging.dtype != torch.uint8
        or raw_staging.ndim != 2
        or raw_staging.shape[1] != expected_slot_bytes
        or raw_staging.stride(1) != 1
    ):
        raise ValueError(
            "ByteV2 raw staging must be a contiguous uint8 page matrix with "
            f"{expected_slot_bytes} bytes per slot"
        )
    raw_kv = raw_staging.view(torch.bfloat16).view(
        raw_staging.shape[0],
        2,
        num_kv_heads,
        block_size,
        head_dim,
    )
    key_cache = raw_kv[:, 0].permute(0, 2, 1, 3)
    value_cache = raw_kv[:, 1].permute(0, 2, 1, 3)
    staging_block_tables = (
        block_tables
        if block_tables_are_staging_slots
        else block_to_staging_slot[block_tables]
    )
    _require_fa2_op("varlen_fwd")(
        query,
        key_cache,
        value_cache,
        output,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        staging_block_tables,
        None,
        max_query_len,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )


def byte_v2_fa2_raw_staging_attention_with_lse(
    query: torch.Tensor,
    raw_staging: torch.Tensor,
    block_to_staging_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    block_tables_are_staging_slots: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run raw paged FA2 over exact staging pages and return output/LSE."""
    expected_slot_bytes = 2 * 2 * num_kv_heads * block_size * head_dim
    if (
        raw_staging.dtype != torch.uint8
        or raw_staging.ndim != 2
        or raw_staging.shape[1] != expected_slot_bytes
        or raw_staging.stride(1) != 1
    ):
        raise ValueError(
            "ByteV2 raw staging must be a contiguous uint8 page matrix with "
            f"{expected_slot_bytes} bytes per slot"
        )
    raw_kv = raw_staging.view(torch.bfloat16).view(
        raw_staging.shape[0],
        2,
        num_kv_heads,
        block_size,
        head_dim,
    )
    key_cache = raw_kv[:, 0].permute(0, 2, 1, 3)
    value_cache = raw_kv[:, 1].permute(0, 2, 1, 3)
    staging_block_tables = (
        block_tables
        if block_tables_are_staging_slots
        else block_to_staging_slot[block_tables]
    )
    result = _require_fa2_op("varlen_fwd")(
        query,
        key_cache,
        value_cache,
        None,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        staging_block_tables,
        None,
        max_query_len,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError("Raw staging FA2 did not return output and LSE tensors")
    return result[0], result[1]


def byte_v2_fa2_direct_paged_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool = False,
) -> None:
    """Run raw paged FA2 over zero-copy views of current BF16 K/V."""

    def is_supported_page_view(pages: torch.Tensor) -> bool:
        if (
            pages.dtype != torch.bfloat16
            or pages.ndim != 4
            or pages.shape[0] <= 0
            or pages.shape[1:] != (16, 8, 128)
        ):
            return False
        page_stride, row_stride, head_stride, dim_stride = pages.stride()
        return (
            dim_stride == 1
            and head_stride == 128
            and row_stride >= 8 * 128
            and page_stride == 16 * row_stride
        )

    if (
        value_pages.shape != key_pages.shape
        or not is_supported_page_view(key_pages)
        or not is_supported_page_view(value_pages)
    ):
        raise ValueError(
            "ByteV2 direct prefill K/V must be row-strided BF16 pages with "
            "shape (num_pages, 16, 8, 128) and contiguous head dimensions"
        )
    if preserve_mixed_dispatch and (not causal or max_query_len <= 0):
        raise ValueError(
            "Preserving mixed raw FA2 dispatch requires causal attention and "
            "a positive maximum query length"
        )
    dispatch_max_query_len = (
        max(max_query_len, 2) if preserve_mixed_dispatch else max_query_len
    )
    _require_fa2_op("varlen_fwd")(
        query,
        key_pages,
        value_pages,
        output,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        block_tables,
        None,
        dispatch_max_query_len,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )


def byte_v2_fa2_hybrid_paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_staging: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool = False,
) -> None:
    """Run FA2 over mixed compact and authoritative raw pages."""
    if preserve_mixed_dispatch and (not causal or max_query_len <= 0):
        raise ValueError(
            "Preserving mixed ByteV2 dispatch requires causal attention and "
            "a positive maximum query length"
        )
    dispatch_max_query_len = (
        max(max_query_len, 2) if preserve_mixed_dispatch else max_query_len
    )
    _require_fa2_op("byte_v2_hybrid_varlen_fwd")(
        query,
        kv_cache,
        raw_staging,
        page_to_raw_slot,
        None,
        output,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        block_tables,
        None,
        dispatch_max_query_len,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )


def byte_v2_static_w16_fa2_paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool = False,
) -> None:
    """Run unchanged FA2 attention over W16 compact/raw KV pages."""
    _byte_v2_static_w16_fa2_paged_attention(
        output,
        query,
        kv_cache,
        raw_pages,
        page_to_raw_slot,
        query_start_locs,
        block_tables,
        seq_lens,
        scale=scale,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        causal=causal,
        preserve_mixed_dispatch=preserve_mixed_dispatch,
    )


def byte_v2_static_w16_fa2_paged_attention_with_lse(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run W16 FA2 and return its output and log-sum-exp state."""
    result = _byte_v2_static_w16_fa2_paged_attention(
        None,
        query,
        kv_cache,
        raw_pages,
        page_to_raw_slot,
        query_start_locs,
        block_tables,
        seq_lens,
        scale=scale,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        causal=causal,
        preserve_mixed_dispatch=preserve_mixed_dispatch,
    )
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError("Static-W16 FA2 did not return output and LSE tensors")
    return result[0], result[1]


def _byte_v2_static_w16_fa2_paged_attention(
    output: torch.Tensor | None,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    raw_pages: torch.Tensor,
    page_to_raw_slot: torch.Tensor,
    query_start_locs: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    preserve_mixed_dispatch: bool,
) -> object:
    if preserve_mixed_dispatch and (not causal or max_query_len <= 0):
        raise ValueError(
            "Preserving mixed Static-W16 dispatch requires causal attention "
            "and a positive maximum query length"
        )
    dispatch_max_query_len = (
        max(max_query_len, 2) if preserve_mixed_dispatch else max_query_len
    )
    num_splits = _static_w16_fa2_profile_num_splits(
        query,
        block_tables,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        causal=causal,
        preserve_mixed_dispatch=preserve_mixed_dispatch,
    )
    result = _require_fa2_op("static_w16_canonical_varlen_fwd")(
        query,
        kv_cache,
        raw_pages,
        page_to_raw_slot,
        output,
        query_start_locs,
        query_start_locs,
        seq_lens,
        None,
        block_tables,
        None,
        dispatch_max_query_len,
        max_seq_len,
        0.0,
        scale,
        False,
        causal,
        -1,
        -1,
        0.0,
        False,
        num_splits,
        None,
    )
    return result


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


def byte_v2_speculative_verify_ragged_q4(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_unsafe_flags: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_locs: torch.Tensor,
    *,
    num_actual_tokens: int,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    partition_size: int,
    tile_policy: Sequence[int],
) -> None:
    """Run a batched mixed-Q request set through the Q4 shared-KV kernel."""
    _require_op("_C", "byte_v2_speculative_verify_ragged_q4")(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        query_start_locs,
        num_actual_tokens,
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
