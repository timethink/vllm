# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-v2 compressed KV cache attention backend.

This module wires Byte-v2 into vLLM's attention backend interface. CUDA uses
native Byte-v2 cache-update and paged-decode kernels by default, with a PyTorch
eager correctness fallback available through ``VLLM_BYTE_V2_USE_NATIVE_KERNELS=0``.
"""

import math
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.byte_v2_decode import (
    byte_v2_paged_prefill_attention_ref,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    BYTE_V2_TILE_SIZE,
    ByteV2PageLayout,
)
from vllm.v1.attention.backends.byte_v2_torch import (
    byte_v2_paged_prefill_attention_torch,
    byte_v2_raw_prefill_attention_torch,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


def _byte_v2_bad_tile_miss_summary(
    exp_tiles: torch.Tensor,
) -> dict[str, int | float]:
    """Return lossless Byte-v2 fallback stats for exponent tiles.

    Args:
        exp_tiles: Tensor shaped `[num_tiles, valid_rows * 16]`, containing
            BF16 exponent bytes for one logical Byte-v2 tile per row.

    Returns:
        Aggregate stats for tiles whose exponents do not fit in one
        16-exponent window.
    """
    num_tiles = int(exp_tiles.shape[0])
    if num_tiles == 0:
        return {
            "bad_tiles": 0,
            "sum_bad_tile_misses": 0,
            "max_misses_per_bad_tile": 0,
            "mean_misses_per_bad_tile": 0.0,
            "bad_tiles_misses_le_1": 0,
            "bad_tiles_misses_le_2": 0,
            "bad_tiles_misses_le_4": 0,
            "bad_tiles_misses_le_8": 0,
            "bad_tiles_misses_gt_8": 0,
        }

    min_exp = exp_tiles.min(dim=1).values
    max_exp = exp_tiles.max(dim=1).values
    bad_exp_tiles = exp_tiles[(max_exp - min_exp) > 15]
    if bad_exp_tiles.numel() == 0:
        return {
            "bad_tiles": 0,
            "sum_bad_tile_misses": 0,
            "max_misses_per_bad_tile": 0,
            "mean_misses_per_bad_tile": 0.0,
            "bad_tiles_misses_le_1": 0,
            "bad_tiles_misses_le_2": 0,
            "bad_tiles_misses_le_4": 0,
            "bad_tiles_misses_le_8": 0,
            "bad_tiles_misses_gt_8": 0,
        }

    tile_elems = int(bad_exp_tiles.shape[1])
    misses_parts: list[torch.Tensor] = []
    # Keep the temporary searchsorted tensors modest; this is a diagnostic path.
    for offset in range(0, int(bad_exp_tiles.shape[0]), 2048):
        chunk = bad_exp_tiles[offset : offset + 2048].to(torch.int16)
        sorted_exp = torch.sort(chunk, dim=1).values
        best_covered = torch.zeros(
            sorted_exp.shape[0], device=sorted_exp.device, dtype=torch.int64
        )
        for start in range(tile_elems):
            right = torch.searchsorted(
                sorted_exp,
                (sorted_exp[:, start] + 15).unsqueeze(1),
                right=True,
            ).squeeze(1)
            best_covered = torch.maximum(best_covered, right - start)
        misses_parts.append((tile_elems - best_covered).detach().cpu())

    misses = torch.cat(misses_parts)
    bad_tiles = int(misses.numel())
    sum_misses = int(misses.sum().item())
    return {
        "bad_tiles": bad_tiles,
        "sum_bad_tile_misses": sum_misses,
        "max_misses_per_bad_tile": int(misses.max().item()),
        "mean_misses_per_bad_tile": float(sum_misses / bad_tiles),
        "bad_tiles_misses_le_1": int((misses <= 1).sum().item()),
        "bad_tiles_misses_le_2": int((misses <= 2).sum().item()),
        "bad_tiles_misses_le_4": int((misses <= 4).sum().item()),
        "bad_tiles_misses_le_8": int((misses <= 8).sum().item()),
        "bad_tiles_misses_gt_8": int((misses > 8).sum().item()),
    }


class ByteV2AttentionBackend(AttentionBackend):
    """Attention backend for Byte-v2 compressed KV cache."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["byte_v2"]

    @staticmethod
    def get_name() -> str:
        return "BYTE_V2"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == 16

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and head_size % 16 == 0 and head_size <= 128

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @staticmethod
    def get_impl_cls() -> type["ByteV2AttentionImpl"]:
        return ByteV2AttentionImpl

    @staticmethod
    def get_builder_cls() -> type["ByteV2MetadataBuilder"]:
        return ByteV2MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "byte_v2",
    ) -> tuple[int, ...]:
        spec = ByteV2FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=torch.uint8,
            raw_tail_bytes=(
                0 if envs.VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE else None
            ),
        )
        return (num_blocks, spec.page_size_bytes)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if block_size is not None and block_size != 16:
            return "Byte-v2 currently requires block_size=16"
        if has_sink:
            return "Byte-v2 does not support attention sinks yet"
        if use_sparse:
            return "Byte-v2 does not support sparse attention yet"
        if use_mla:
            return "Byte-v2 does not support MLA yet"
        return None


@dataclass
class ByteV2Metadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    block_size: int
    page_size_bytes: int
    causal: bool = True
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


@dataclass
class ByteV2SparseFallbackPool:
    kv_cache_ptr: int
    fallback_pool: torch.Tensor
    fallback_block_ids: torch.Tensor
    fallback_next_slot: torch.Tensor
    fallback_tile_ids: torch.Tensor | None = None
    fallback_tile_next_slot: torch.Tensor | None = None
    deferred_error: torch.Tensor | None = None


class ByteV2MetadataBuilder(AttentionMetadataBuilder[ByteV2Metadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: ByteV2FullAttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.page_size_bytes = kv_cache_spec.page_size_bytes
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> ByteV2Metadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ByteV2Metadata:
        # Byte-v2 does not use vLLM's cascade-attention split yet.  Prefix-cache
        # block reuse is still correct through the ordinary paged block table.
        del common_prefix_len

        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )

        return ByteV2Metadata(
            seq_lens=common_attn_metadata.seq_lens,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            query_start_loc=common_attn_metadata.query_start_loc,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            block_size=self.block_size,
            page_size_bytes=self.page_size_bytes,
            causal=common_attn_metadata.causal,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu_upper_bound,
        )

    def update_block_table(
        self,
        metadata: ByteV2Metadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> ByteV2Metadata:
        return replace(
            metadata,
            block_table=blk_table,
            slot_mapping=slot_mapping,
        )


class ByteV2AttentionImpl(AttentionImpl[ByteV2Metadata]):
    _DEFERRED_ERROR_NUMEL: ClassVar[int] = 4
    _deferred_cache_update_errors: ClassVar[
        dict[tuple[str, int], torch.Tensor]
    ] = {}
    _CACHE_UPDATE_ERROR_MESSAGES: ClassVar[dict[int, str]] = {
        1: "invalid page state",
        2: "invalid slot mapping",
        3: "duplicate token slot in one cache update",
        4: "attempted to update a finalized KV block",
        5: "sparse fallback pool is required but missing",
        6: "sparse fallback block id is invalid",
        7: "sparse fallback pool exhausted",
        8: "invalid Byte-v2 valid row count",
        9: "no touched token found for cache block",
    }

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "byte_v2",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size_v = head_size
        self.kv_cache_dtype = kv_cache_dtype
        self._sparse_fallback_pool: ByteV2SparseFallbackPool | None = None
        self._decode_partial_workspace: torch.Tensor | None = None
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=head_size)

    @classmethod
    def _deferred_error_key(cls, device: torch.device) -> tuple[str, int]:
        if device.type == "cuda":
            index = device.index
            if index is None:
                index = torch.cuda.current_device()
            return (device.type, int(index))
        return (device.type, -1 if device.index is None else int(device.index))

    @classmethod
    def _deferred_error_device(cls, device: torch.device) -> torch.device:
        if device.type != "cuda":
            return device
        _, index = cls._deferred_error_key(device)
        return torch.device("cuda", index)

    @classmethod
    def _get_deferred_cache_update_error(
        cls,
        device: torch.device,
    ) -> torch.Tensor:
        key = cls._deferred_error_key(device)
        error = cls._deferred_cache_update_errors.get(key)
        if error is None:
            error = torch.zeros(
                cls._DEFERRED_ERROR_NUMEL,
                dtype=torch.int32,
                device=cls._deferred_error_device(device),
            )
            cls._deferred_cache_update_errors[key] = error
        return error

    @classmethod
    def _maybe_get_deferred_cache_update_error(
        cls,
        device: torch.device,
    ) -> torch.Tensor | None:
        if (
            device.type != "cuda"
            or not envs.VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK
        ):
            return None
        return cls._get_deferred_cache_update_error(device)

    @classmethod
    def reset_deferred_cache_update_error(cls, device: torch.device) -> None:
        if (
            device.type != "cuda"
            or not envs.VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK
        ):
            return
        cls._get_deferred_cache_update_error(device).zero_()

    @classmethod
    def check_deferred_cache_update_error(cls, device: torch.device) -> None:
        if (
            device.type != "cuda"
            or not envs.VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK
        ):
            return
        error = cls._deferred_cache_update_errors.get(
            cls._deferred_error_key(device)
        )
        if error is None:
            return

        code, block_id, fallback_used, fallback_capacity = (
            int(value) for value in error.cpu().tolist()
        )
        if code == 0:
            return

        error.zero_()
        detail = cls._CACHE_UPDATE_ERROR_MESSAGES.get(code, "unknown error")
        raise RuntimeError(
            "Byte-v2 deferred decode cache append failed: "
            f"{detail} (error_code={code}, block_id={block_id}, "
            f"fallback_pool_used={fallback_used}, "
            f"fallback_pool_capacity={fallback_capacity})"
        )

    def _make_layout(self, page_size_bytes: int) -> ByteV2PageLayout:
        raw_overlay_layout = ByteV2PageLayout(
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
        )
        if raw_overlay_layout.page_size_bytes == page_size_bytes:
            return raw_overlay_layout

        compressed_layout = ByteV2PageLayout(
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            raw_tail_bytes=0,
        )
        if compressed_layout.page_size_bytes == page_size_bytes:
            return compressed_layout

        raise ValueError(
            "Byte-v2 page size mismatch: expected "
            f"{raw_overlay_layout.page_size_bytes} or "
            f"{compressed_layout.page_size_bytes}, got {page_size_bytes}"
        )

    def _get_sparse_fallback_pool(
        self,
        kv_cache: torch.Tensor,
        layout: ByteV2PageLayout,
    ) -> ByteV2SparseFallbackPool | None:
        if layout.raw_tail_bytes != 0:
            return None
        if not envs.VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL:
            return None
        ratio = envs.VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO
        if ratio <= 0:
            return None

        kv_cache_ptr = kv_cache.data_ptr()
        num_blocks = kv_cache.shape[0]
        if num_blocks <= 0:
            return None
        deferred_error = self._maybe_get_deferred_cache_update_error(
            kv_cache.device
        )
        min_blocks = envs.VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS
        pool_blocks = min(
            num_blocks,
            max(1, min_blocks, math.ceil(num_blocks * ratio)),
        )
        total_tiles = layout.num_kv_heads * (
            layout.k_dim_tiles + layout.v_dim_tiles
        )
        pool = self._sparse_fallback_pool
        if (
            pool is not None
            and pool.kv_cache_ptr == kv_cache_ptr
            and pool.fallback_block_ids.numel() == num_blocks
            and pool.fallback_pool.shape == (pool_blocks, layout.raw_block_bytes)
            and pool.fallback_tile_ids is not None
            and pool.fallback_tile_ids.shape == (num_blocks, total_tiles)
            and pool.fallback_tile_next_slot is not None
        ):
            pool.deferred_error = deferred_error
            return pool

        fallback_pool = torch.empty(
            pool_blocks,
            layout.raw_block_bytes,
            dtype=torch.uint8,
            device=kv_cache.device,
        )
        fallback_block_ids = torch.full(
            (num_blocks,),
            -1,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        fallback_next_slot = torch.zeros(
            1,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        fallback_tile_ids = torch.full(
            (num_blocks, total_tiles),
            -1,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        fallback_tile_next_slot = torch.zeros(
            1,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        pool = ByteV2SparseFallbackPool(
            kv_cache_ptr=kv_cache_ptr,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
            fallback_tile_ids=fallback_tile_ids,
            fallback_tile_next_slot=fallback_tile_next_slot,
            deferred_error=deferred_error,
        )
        self._sparse_fallback_pool = pool
        return pool

    def _decode_num_kv_splits(
        self,
        *,
        num_decode_tokens: int,
        num_logical_pages: int,
    ) -> int:
        q_per_kv = self.num_heads // self.num_kv_heads
        if q_per_kv <= 1 or q_per_kv > 8:
            return 1

        max_split_k = 1
        if num_logical_pages >= 256:
            max_split_k = 128
        elif num_logical_pages >= 128:
            max_split_k = 64
        elif num_logical_pages >= 96:
            max_split_k = 32
        elif num_logical_pages >= 16:
            max_split_k = 16

        active_head_groups = num_decode_tokens * self.num_kv_heads
        if active_head_groups >= 64:
            max_split_k = min(max_split_k, 8)
        elif active_head_groups >= 32:
            max_split_k = min(max_split_k, 16)

        requested_split_k = envs.VLLM_BYTE_V2_DECODE_SPLIT_K
        if requested_split_k > 0:
            max_split_k = max(1, requested_split_k)

        if max_split_k <= 1 or num_logical_pages < 16:
            return 1
        return min(max_split_k, num_logical_pages)

    def _get_decode_partial_workspace(
        self,
        query: torch.Tensor,
        *,
        num_decode_tokens: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not envs.VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE
            or query.device.type != "cuda"
            or num_decode_tokens <= 0
        ):
            return None

        num_kv_splits = self._decode_num_kv_splits(
            num_decode_tokens=num_decode_tokens,
            num_logical_pages=int(block_table.shape[1]),
        )
        if num_kv_splits <= 1:
            return None

        partial_numel = (
            num_decode_tokens
            * self.num_heads
            * num_kv_splits
            * (self.head_size_v + 1)
        )
        if is_workspace_manager_initialized():
            (workspace,) = current_workspace_manager().get_simultaneous(
                ((partial_numel,), torch.float32)
            )
            return workspace

        workspace = self._decode_partial_workspace
        if (
            workspace is None
            or workspace.device != query.device
            or workspace.numel() < partial_numel
        ):
            workspace = torch.empty(
                partial_numel,
                dtype=torch.float32,
                device=query.device,
            )
            self._decode_partial_workspace = workspace
        return workspace[:partial_numel]

    def register_sparse_fallback_pool(
        self,
        kv_cache: torch.Tensor,
        fallback_pool: torch.Tensor,
        fallback_block_ids: torch.Tensor,
        fallback_next_slot: torch.Tensor,
        fallback_tile_ids: torch.Tensor | None = None,
        fallback_tile_next_slot: torch.Tensor | None = None,
        deferred_error: torch.Tensor | None = None,
    ) -> None:
        if deferred_error is None:
            deferred_error = self._maybe_get_deferred_cache_update_error(
                kv_cache.device
            )
        self._sparse_fallback_pool = ByteV2SparseFallbackPool(
            kv_cache_ptr=kv_cache.data_ptr(),
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
            fallback_tile_ids=fallback_tile_ids,
            fallback_tile_next_slot=fallback_tile_next_slot,
            deferred_error=deferred_error,
        )

    def get_sparse_fallback_pool_stats(self) -> dict[str, int | bool]:
        pool = self._sparse_fallback_pool
        if pool is None:
            return {
                "enabled": False,
                "capacity": 0,
                "next_slot": 0,
                "assigned_blocks": 0,
                "exhausted": False,
            }

        capacity = int(pool.fallback_pool.shape[0])
        next_slot = int(pool.fallback_next_slot.item())
        assigned_blocks = int((pool.fallback_block_ids >= 0).sum().item())
        tile_capacity = int(pool.fallback_pool.numel() // 512)
        tile_next_slot = (
            0
            if pool.fallback_tile_next_slot is None
            else int(pool.fallback_tile_next_slot.item())
        )
        assigned_tiles = (
            0
            if pool.fallback_tile_ids is None
            else int((pool.fallback_tile_ids >= 0).sum().item())
        )
        return {
            "enabled": True,
            "capacity": capacity,
            "next_slot": next_slot,
            "assigned_blocks": assigned_blocks,
            "tile_capacity": tile_capacity,
            "tile_next_slot": tile_next_slot,
            "assigned_tiles": assigned_tiles,
            "exhausted": next_slot > capacity,
        }

    def get_tile_fallback_stats(
        self,
        kv_cache: torch.Tensor,
    ) -> dict[str, int | float | bool]:
        """Estimate tile-level fallback needs from current Byte-v2 cache.

        The current lossless sparse fallback pool stores whole raw BF16 blocks.
        This diagnostic re-checks those raw blocks at `16x16` tile granularity
        to estimate how much raw storage tile-level fallback would require.
        """
        layout = self._make_layout(kv_cache.shape[1])
        tiles_per_block = layout.num_kv_heads * (
            layout.k_dim_tiles + layout.v_dim_tiles
        )
        statuses = kv_cache[:, BYTE_V2_PAGE_STATUS_OFFSET]
        valid_rows = kv_cache[:, BYTE_V2_PAGE_VALID_ROWS_OFFSET].to(torch.int64)
        full_active_mask = (
            (
                (statuses == BYTE_V2_PAGE_STATUS_COMPRESSED)
                | (statuses == BYTE_V2_PAGE_STATUS_RAW_FALLBACK)
            )
            & (valid_rows == BYTE_V2_TILE_SIZE)
        )
        full_raw_mask = (
            (statuses == BYTE_V2_PAGE_STATUS_RAW_FALLBACK)
            & (valid_rows == BYTE_V2_TILE_SIZE)
        )
        partial_raw_blocks = int(
            (
                (statuses == BYTE_V2_PAGE_STATUS_RAW_FALLBACK)
                & (valid_rows > 0)
                & (valid_rows < BYTE_V2_TILE_SIZE)
            )
            .sum()
            .item()
        )

        full_active_blocks = int(full_active_mask.sum().item())
        full_raw_blocks = int(full_raw_mask.sum().item())
        full_total_tiles = full_active_blocks * tiles_per_block
        raw_full_total_tiles = full_raw_blocks * tiles_per_block

        empty_summary = {
            "enabled": self._sparse_fallback_pool is not None
            or layout.raw_tail_bytes != 0,
            "full_active_blocks": full_active_blocks,
            "full_raw_fallback_blocks": full_raw_blocks,
            "partial_raw_fallback_blocks": partial_raw_blocks,
            "tiles_per_block": tiles_per_block,
            "full_total_tiles": full_total_tiles,
            "raw_full_total_tiles": raw_full_total_tiles,
            "full_bad_tiles": 0,
            "full_good_tiles_inside_raw_fallback_blocks": raw_full_total_tiles,
            "full_tile_fallback_ratio": 0.0,
            "bad_tile_ratio_within_raw_fallback_blocks": 0.0,
            "invalid_raw_fallback_slots": 0,
            "sum_bad_tile_misses": 0,
            "max_misses_per_bad_tile": 0,
            "mean_misses_per_bad_tile": 0.0,
            "bad_tiles_misses_le_1": 0,
            "bad_tiles_misses_le_2": 0,
            "bad_tiles_misses_le_4": 0,
            "bad_tiles_misses_le_8": 0,
            "bad_tiles_misses_gt_8": 0,
        }
        if full_raw_blocks == 0:
            return empty_summary

        raw_block_ids = torch.nonzero(full_raw_mask, as_tuple=False).flatten()
        pool = self._sparse_fallback_pool
        invalid_slots = 0
        if pool is not None:
            fallback_slots = pool.fallback_block_ids[raw_block_ids].to(torch.int64)
            valid_slot_mask = (
                (fallback_slots >= 0)
                & (fallback_slots < pool.fallback_pool.shape[0])
            )
            invalid_slots = int((~valid_slot_mask).sum().item())
            fallback_slots = fallback_slots[valid_slot_mask]
            if fallback_slots.numel() == 0:
                empty_summary["invalid_raw_fallback_slots"] = invalid_slots
                return empty_summary
            raw_bytes = pool.fallback_pool[fallback_slots]
        elif layout.raw_tail_bytes != 0:
            raw_bytes = kv_cache[
                raw_block_ids,
                layout.page_header_bytes : layout.page_header_bytes
                + layout.raw_block_bytes,
            ]
        else:
            empty_summary["invalid_raw_fallback_slots"] = full_raw_blocks
            return empty_summary

        raw_bits = raw_bytes.contiguous().view(torch.int16).to(torch.int32)
        raw_bits = raw_bits & 0xFFFF
        key_elems = layout.raw_key_bytes // 2
        value_elems = layout.raw_value_bytes // 2
        num_raw_blocks = int(raw_bits.shape[0])
        key_bits = raw_bits[:, :key_elems].reshape(
            num_raw_blocks,
            BYTE_V2_TILE_SIZE,
            layout.num_kv_heads,
            layout.head_size,
        )
        value_bits = raw_bits[:, key_elems : key_elems + value_elems].reshape(
            num_raw_blocks,
            BYTE_V2_TILE_SIZE,
            layout.num_kv_heads,
            layout.head_size_v,
        )

        key_exp_tiles = ((key_bits >> 7) & 0xFF).reshape(
            num_raw_blocks,
            BYTE_V2_TILE_SIZE,
            layout.num_kv_heads,
            layout.k_dim_tiles,
            BYTE_V2_TILE_SIZE,
        )
        key_exp_tiles = key_exp_tiles.permute(0, 2, 3, 1, 4).reshape(
            -1, BYTE_V2_TILE_SIZE * BYTE_V2_TILE_SIZE
        )
        value_exp_tiles = ((value_bits >> 7) & 0xFF).reshape(
            num_raw_blocks,
            BYTE_V2_TILE_SIZE,
            layout.num_kv_heads,
            layout.v_dim_tiles,
            BYTE_V2_TILE_SIZE,
        )
        value_exp_tiles = value_exp_tiles.permute(0, 2, 3, 1, 4).reshape(
            -1, BYTE_V2_TILE_SIZE * BYTE_V2_TILE_SIZE
        )
        exp_tiles = torch.cat((key_exp_tiles, value_exp_tiles), dim=0)

        miss_summary = _byte_v2_bad_tile_miss_summary(exp_tiles)
        full_bad_tiles = int(miss_summary["bad_tiles"])
        empty_summary.update(miss_summary)
        empty_summary["full_bad_tiles"] = full_bad_tiles
        empty_summary["full_good_tiles_inside_raw_fallback_blocks"] = (
            raw_full_total_tiles - full_bad_tiles
        )
        empty_summary["full_tile_fallback_ratio"] = (
            float(full_bad_tiles / full_total_tiles)
            if full_total_tiles > 0
            else 0.0
        )
        empty_summary["bad_tile_ratio_within_raw_fallback_blocks"] = (
            float(full_bad_tiles / raw_full_total_tiles)
            if raw_full_total_tiles > 0
            else 0.0
        )
        empty_summary["invalid_raw_fallback_slots"] = invalid_slots
        return empty_summary

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        layout = self._make_layout(kv_cache.shape[1])
        fallback_pool = self._get_sparse_fallback_pool(kv_cache, layout)
        ops.byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            page_size_bytes=kv_cache.shape[1],
            fallback_pool=(
                None if fallback_pool is None else fallback_pool.fallback_pool
            ),
            fallback_block_ids=(
                None if fallback_pool is None else fallback_pool.fallback_block_ids
            ),
            fallback_next_slot=(
                None if fallback_pool is None else fallback_pool.fallback_next_slot
            ),
            fallback_tile_ids=(
                None if fallback_pool is None else fallback_pool.fallback_tile_ids
            ),
            fallback_tile_next_slot=(
                None
                if fallback_pool is None
                else fallback_pool.fallback_tile_next_slot
            ),
            deferred_error=(
                None if fallback_pool is None else fallback_pool.deferred_error
            ),
        )

    def _prefill_has_raw_only_context(
        self,
        attn_metadata: ByteV2Metadata,
    ) -> bool:
        start_req_idx = attn_metadata.num_decodes
        if attn_metadata.num_prefills == 0:
            return False

        query_start_loc = (
            attn_metadata.query_start_loc_cpu
            if attn_metadata.query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        query_lens = (
            query_start_loc[start_req_idx + 1 :]
            - query_start_loc[start_req_idx:-1]
        )
        seq_lens = (
            attn_metadata.seq_lens_cpu
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens
        )
        prefill_seq_lens = seq_lens[start_req_idx : start_req_idx + query_lens.numel()]
        return bool(torch.equal(prefill_seq_lens.cpu(), query_lens.cpu()))

    def _raw_prefill_attention_flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: ByteV2Metadata,
    ) -> torch.Tensor | None:
        if (
            key is None
            or value is None
            or query.device.type != "cuda"
            or not is_flash_attn_varlen_func_available()
            or self.vllm_flash_attn_version is None
            or not self._prefill_has_raw_only_context(attn_metadata)
        ):
            return None

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        prefill_end = num_decode_tokens + num_prefill_tokens
        q_prefill = query[num_decode_tokens:prefill_end].contiguous()
        k_prefill = key[num_decode_tokens:prefill_end].contiguous()
        v_prefill = value[num_decode_tokens:prefill_end].contiguous()

        query_start_loc = attn_metadata.query_start_loc
        start_req_idx = attn_metadata.num_decodes
        end_req_idx = start_req_idx + attn_metadata.num_prefills + 1
        cu_seqlens = (
            query_start_loc[start_req_idx:end_req_idx]
            - query_start_loc[start_req_idx]
        )
        prefill_out = torch.empty(
            num_prefill_tokens,
            self.num_heads,
            self.head_size_v,
            dtype=query.dtype,
            device=query.device,
        )
        flash_attn_varlen_func(
            q=q_prefill,
            k=k_prefill,
            v=v_prefill,
            out=prefill_out,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=attn_metadata.max_query_len,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=None,
            softcap=0,
            fa_version=self.vllm_flash_attn_version,
        )
        return prefill_out

    def _paged_prefill_attention_decode_like(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ByteV2Metadata,
        layout: ByteV2PageLayout,
        fallback_pool: ByteV2SparseFallbackPool | None,
    ) -> torch.Tensor | None:
        if query.device.type != "cuda":
            return None

        query_start_loc = (
            attn_metadata.query_start_loc_cpu
            if attn_metadata.query_start_loc_cpu is not None
            else attn_metadata.query_start_loc.detach().cpu()
        )
        seq_lens = (
            attn_metadata.seq_lens_cpu
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens.detach().cpu()
        )
        start_req_idx = attn_metadata.num_decodes
        num_reqs = query_start_loc.numel() - 1
        query_parts: list[torch.Tensor] = []
        block_table_parts: list[torch.Tensor] = []
        seq_len_parts: list[torch.Tensor] = []

        for req_idx in range(start_req_idx, num_reqs):
            q_start = int(query_start_loc[req_idx].item())
            q_end = int(query_start_loc[req_idx + 1].item())
            query_len = q_end - q_start
            if query_len <= 0:
                continue

            seq_len = int(seq_lens[req_idx].item())
            context_len = seq_len - query_len
            if context_len < 0:
                raise ValueError("prefill query_len cannot exceed seq_len")

            query_parts.append(query[q_start:q_end])
            block_table_parts.append(
                attn_metadata.block_table[req_idx : req_idx + 1].expand(
                    query_len, -1
                )
            )
            if attn_metadata.causal:
                seq_len_parts.append(
                    torch.arange(
                        context_len + 1,
                        seq_len + 1,
                        dtype=attn_metadata.seq_lens.dtype,
                        device=attn_metadata.seq_lens.device,
                    )
                )
            else:
                seq_len_parts.append(
                    torch.full(
                        (query_len,),
                        seq_len,
                        dtype=attn_metadata.seq_lens.dtype,
                        device=attn_metadata.seq_lens.device,
                    )
                )

        if not query_parts:
            return torch.empty(
                0,
                self.num_heads,
                self.head_size_v,
                dtype=query.dtype,
                device=query.device,
            )

        paged_query = torch.cat(query_parts, dim=0).contiguous()
        paged_block_table = torch.cat(block_table_parts, dim=0).contiguous()
        paged_seq_lens = torch.cat(seq_len_parts, dim=0).contiguous()
        partial_workspace = self._get_decode_partial_workspace(
            paged_query,
            num_decode_tokens=int(paged_query.shape[0]),
            block_table=paged_block_table,
        )

        return ops.byte_v2_paged_decode_attention(
            paged_query,
            kv_cache,
            paged_block_table,
            paged_seq_lens,
            self.scale,
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            page_size_bytes=kv_cache.shape[1],
            fallback_pool=(
                None if fallback_pool is None else fallback_pool.fallback_pool
            ),
            fallback_block_ids=(
                None if fallback_pool is None else fallback_pool.fallback_block_ids
            ),
            fallback_tile_ids=(
                None if fallback_pool is None else fallback_pool.fallback_tile_ids
            ),
            partial_workspace=partial_workspace,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ByteV2Metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Byte-v2 attention does not support fused output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.num_decodes != attn_metadata.num_decode_tokens:
            raise NotImplementedError(
                "Byte-v2 reference forward does not support speculative decode"
            )
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens].reshape(
            num_actual_tokens, self.num_heads, self.head_size
        ).contiguous()
        layout = self._make_layout(attn_metadata.page_size_bytes)
        fallback_pool = self._get_sparse_fallback_pool(kv_cache, layout)
        output_rows = 0
        if attn_metadata.num_decode_tokens:
            num_decode_tokens = attn_metadata.num_decode_tokens
            partial_workspace = self._get_decode_partial_workspace(
                q[:num_decode_tokens],
                num_decode_tokens=num_decode_tokens,
                block_table=attn_metadata.block_table[:num_decode_tokens],
            )
            decoded = ops.byte_v2_paged_decode_attention(
                q[:num_decode_tokens],
                kv_cache,
                attn_metadata.block_table[:num_decode_tokens],
                attn_metadata.seq_lens[:num_decode_tokens],
                self.scale,
                block_size=16,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                page_size_bytes=kv_cache.shape[1],
                fallback_pool=(
                    None if fallback_pool is None else fallback_pool.fallback_pool
                ),
                fallback_block_ids=(
                    None if fallback_pool is None else fallback_pool.fallback_block_ids
                ),
                fallback_tile_ids=(
                    None if fallback_pool is None else fallback_pool.fallback_tile_ids
                ),
                partial_workspace=partial_workspace,
            )
            if output.ndim == 3:
                output[:num_decode_tokens].copy_(decoded)
            else:
                output[:num_decode_tokens].copy_(decoded.reshape(num_decode_tokens, -1))
            output_rows = num_decode_tokens

        if attn_metadata.num_prefill_tokens:
            query_start_loc = (
                attn_metadata.query_start_loc_cpu
                if attn_metadata.query_start_loc_cpu is not None
                else attn_metadata.query_start_loc
            )
            raw_prefill_out = self._raw_prefill_attention_flash(
                q,
                None if key is None else key[:num_actual_tokens],
                None if value is None else value[:num_actual_tokens],
                attn_metadata,
            )
            prefill_out = raw_prefill_out
            if (
                prefill_out is None
                and key is not None
                and value is not None
                and self._prefill_has_raw_only_context(attn_metadata)
            ):
                prefill_out = byte_v2_raw_prefill_attention_torch(
                    q,
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    attn_metadata.seq_lens,
                    query_start_loc,
                    layout,
                    self.scale,
                    start_req_idx=attn_metadata.num_decodes,
                    causal=attn_metadata.causal,
                )
            if prefill_out is None:
                prefill_out = self._paged_prefill_attention_decode_like(
                    q,
                    kv_cache,
                    attn_metadata,
                    layout,
                    fallback_pool,
                )
            if prefill_out is None and q.device.type == "cpu":
                prefill_out = byte_v2_paged_prefill_attention_ref(
                    q,
                    kv_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    query_start_loc,
                    layout,
                    self.scale,
                    start_req_idx=attn_metadata.num_decodes,
                    causal=attn_metadata.causal,
                )
            elif prefill_out is None:
                prefill_out = byte_v2_paged_prefill_attention_torch(
                    q,
                    kv_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    query_start_loc,
                    layout,
                    self.scale,
                    start_req_idx=attn_metadata.num_decodes,
                    causal=attn_metadata.causal,
                    fallback_pool=(
                        None if fallback_pool is None else fallback_pool.fallback_pool
                    ),
                    fallback_block_ids=(
                        None
                        if fallback_pool is None
                        else fallback_pool.fallback_block_ids
                    ),
                    fallback_tile_ids=(
                        None
                        if fallback_pool is None
                        else fallback_pool.fallback_tile_ids
                    ),
                )
            prefill_end = output_rows + attn_metadata.num_prefill_tokens
            if output.ndim == 3:
                output[output_rows:prefill_end].copy_(prefill_out)
            else:
                output[output_rows:prefill_end].copy_(
                    prefill_out.reshape(attn_metadata.num_prefill_tokens, -1)
                )
            output_rows = prefill_end

        if output.shape[0] > output_rows:
            output[output_rows:].zero_()
        return output
