# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python-registered Byte-v2 custom ops.

These ops provide stable call sites for the Byte-v2 backend. The first
implementation uses CPU reference helpers; later integration steps can replace
the internals with C++/CUDA kernels while keeping the Python call signature.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.byte_v2_decode import (
    byte_v2_paged_decode_attention_ref,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    ByteV2PageLayout,
    ByteV2PageLayoutV3,
    byte_v2_reshape_and_cache_ref,
)
from vllm.v1.attention.backends.byte_v2_torch import (
    byte_v2_paged_decode_attention_torch,
    byte_v2_reshape_and_cache_torch,
)


def _make_layout(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
) -> ByteV2PageLayout | ByteV2PageLayoutV3:
    layout = ByteV2PageLayout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
    )
    if layout.page_size_bytes == page_size_bytes:
        return layout

    compressed_layout = ByteV2PageLayout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        raw_tail_bytes=0,
    )
    if compressed_layout.page_size_bytes == page_size_bytes:
        return compressed_layout

    v3_layout = ByteV2PageLayoutV3(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
    )
    if v3_layout.page_size_bytes == page_size_bytes:
        return v3_layout

    raise ValueError(
        "Byte-v2 op page size mismatch: "
        f"layout expects {layout.page_size_bytes} or "
        f"{compressed_layout.page_size_bytes} or "
        f"{v3_layout.page_size_bytes}, got {page_size_bytes}"
    )


def _byte_v2_reshape_and_cache_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_next_slot: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    fallback_tile_next_slot: torch.Tensor | None = None,
    deferred_error: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
    outlier_next_entry: torch.Tensor | None = None,
    decode_append_fast_path_safe: bool = False,
) -> torch.Tensor:
    del decode_append_fast_path_safe
    if (
        fallback_pool is not None
        or fallback_block_ids is not None
        or fallback_next_slot is not None
        or fallback_tile_ids is not None
        or fallback_tile_next_slot is not None
        or deferred_error is not None
        or outlier_arena is not None
        or outlier_block_flags is not None
        or outlier_tile_bitmap is not None
        or outlier_tile_meta is not None
        or outlier_next_entry is not None
    ):
        raise NotImplementedError(
            "Byte-v2 sparse fallback pool requires the native CUDA ops"
        )
    layout = _make_layout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        page_size_bytes=page_size_bytes,
    )
    if key.device.type == "cuda" or kv_cache.device.type == "cuda":
        return byte_v2_reshape_and_cache_torch(
            key, value, kv_cache, slot_mapping, layout
        )

    packed_block_ids = byte_v2_reshape_and_cache_ref(
        key, value, kv_cache, slot_mapping, layout
    )
    return torch.tensor(packed_block_ids, device=slot_mapping.device, dtype=torch.int64)


def _byte_v2_reshape_and_cache_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_next_slot: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    fallback_tile_next_slot: torch.Tensor | None = None,
    deferred_error: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
    outlier_next_entry: torch.Tensor | None = None,
    decode_append_fast_path_safe: bool = False,
) -> torch.Tensor:
    del decode_append_fast_path_safe
    return torch.empty(0, device=slot_mapping.device, dtype=torch.int64)


direct_register_custom_op(
    op_name="byte_v2_reshape_and_cache",
    op_func=_byte_v2_reshape_and_cache_impl,
    fake_impl=_byte_v2_reshape_and_cache_fake,
    mutates_args=["kv_cache"],
    dispatch_key="CompositeExplicitAutograd",
)


def _byte_v2_paged_decode_attention_impl(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    partial_workspace: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    del partial_workspace
    if (
        fallback_pool is not None
        or fallback_block_ids is not None
        or fallback_tile_ids is not None
        or outlier_arena is not None
        or outlier_block_flags is not None
        or outlier_tile_bitmap is not None
        or outlier_tile_meta is not None
    ):
        raise NotImplementedError(
            "Byte-v2 sparse fallback pool requires the native CUDA ops"
        )
    layout = _make_layout(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        page_size_bytes=page_size_bytes,
    )
    if query.device.type == "cuda" or kv_cache.device.type == "cuda":
        return byte_v2_paged_decode_attention_torch(
            query, kv_cache, block_table, seq_lens, layout, scale
        )

    return byte_v2_paged_decode_attention_ref(
        query, kv_cache, block_table, seq_lens, layout, scale
    )


def _byte_v2_paged_decode_attention_fake(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    partial_workspace: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        partial_workspace,
        outlier_arena,
        outlier_block_flags,
        outlier_tile_bitmap,
        outlier_tile_meta,
    )
    return torch.empty(
        query.shape[0],
        query.shape[1],
        head_size_v,
        device=query.device,
        dtype=query.dtype,
    )


direct_register_custom_op(
    op_name="byte_v2_paged_decode_attention",
    op_func=_byte_v2_paged_decode_attention_impl,
    fake_impl=_byte_v2_paged_decode_attention_fake,
    dispatch_key="CompositeExplicitAutograd",
)


def byte_v2_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_next_slot: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    fallback_tile_next_slot: torch.Tensor | None = None,
    deferred_error: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
    outlier_next_entry: torch.Tensor | None = None,
    decode_append_fast_path_safe: bool = False,
) -> torch.Tensor:
    return torch.ops.vllm.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        page_size_bytes,
        fallback_pool,
        fallback_block_ids,
        fallback_next_slot,
        fallback_tile_ids,
        fallback_tile_next_slot,
        deferred_error,
        outlier_arena,
        outlier_block_flags,
        outlier_tile_bitmap,
        outlier_tile_meta,
        outlier_next_entry,
        decode_append_fast_path_safe,
    )


def byte_v2_paged_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    page_size_bytes: int,
    fallback_pool: torch.Tensor | None = None,
    fallback_block_ids: torch.Tensor | None = None,
    fallback_tile_ids: torch.Tensor | None = None,
    partial_workspace: torch.Tensor | None = None,
    outlier_arena: torch.Tensor | None = None,
    outlier_block_flags: torch.Tensor | None = None,
    outlier_tile_bitmap: torch.Tensor | None = None,
    outlier_tile_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.byte_v2_paged_decode_attention(
        query,
        kv_cache,
        block_table,
        seq_lens,
        scale,
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        page_size_bytes,
        fallback_pool,
        fallback_block_ids,
        fallback_tile_ids,
        partial_workspace,
        outlier_arena,
        outlier_block_flags,
        outlier_tile_bitmap,
        outlier_tile_meta,
    )
