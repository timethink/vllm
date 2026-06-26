# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental ByteV2 attention backend scaffold.

The implementation is intentionally fail-closed until native ByteV2 CUDA
kernels are registered. This file wires the policy/configuration layer that the
kernels will consume, without making an incomplete runtime path selectable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    DEFAULT_BYTE_V2_TILE_POLICY,
    ByteV2PageLayoutV4,
    ByteV2RawStagingLayout,
    ByteV2TilePolicy,
    byte_v2_tile_policy_from_env,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_append_raw_staging,
    byte_v2_collect_cache_stats,
    byte_v2_commit_raw_staging_to_cache,
    byte_v2_custom_ops_are_available,
    byte_v2_hydrate_raw_staging_from_cache,
    byte_v2_paged_decode_attention,
    byte_v2_paged_decode_attention_split_k,
    byte_v2_paged_decode_attention_split_k_guarded,
    byte_v2_prefill_attention,
    byte_v2_prepare_raw_staging,
    byte_v2_release_raw_staging,
    byte_v2_reshape_and_cache,
    byte_v2_update_cache_single_token,
    byte_v2_update_cache_unsafe_flags,
)

_BYTE_V2_KERNELS_NOT_READY = "ByteV2 native CUDA kernels are not registered yet"
_BYTE_V2_MAX_RAW_STAGING_TOKENS = 1024
logger = init_logger(__name__)


def _debug_warmup_enabled() -> bool:
    return bool(
        os.environ.get("BYTE_V2_DEBUG_WARMUP")
        or os.environ.get("VLLM_BYTE_V2_DEBUG_WARMUP")
    )


def _debug_warmup(message: str, *args: object) -> None:
    if _debug_warmup_enabled():
        logger.warning("[ByteV2] %s", message % args if args else message)


def _decode_raw_fallback_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_RAW_FALLBACK")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_assume_no_outlier_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_ASSUME_NO_OUTLIER")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_gqa_packed_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_GQA_PACKED")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_gqa_fa2_like_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_GQA_FA2_LIKE")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_gqa_fa2_direct_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_GQA_FA2_DIRECT")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_validate_no_outlier_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_VALIDATE_NO_OUTLIER")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _decode_page_unsafe_flags_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_PAGE_UNSAFE_FLAGS")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _decode_split_k_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_SPLIT_K")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _prefill_backend() -> str:
    value = os.environ.get("BYTE_V2_PREFILL_BACKEND")
    if value is None:
        return "sdpa"
    backend = value.lower()
    if backend not in ("sdpa", "native", "fallback"):
        logger.warning("[ByteV2] ignoring invalid BYTE_V2_PREFILL_BACKEND=%r", value)
        return "sdpa"
    return backend


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("[ByteV2] ignoring invalid %s=%r", name, value)
        return default
    if parsed <= 0:
        logger.warning("[ByteV2] ignoring non-positive %s=%r", name, value)
        return default
    return parsed


def _debug_sync(label: str) -> None:
    if not _debug_warmup_enabled():
        return
    logger.warning("[ByteV2] sync start %s", label)
    torch.accelerator.synchronize()
    logger.warning("[ByteV2] sync done %s", label)


@dataclass
class ByteV2AttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool
    common_prefix_len: int = 0
    tile_policy: ByteV2TilePolicy = DEFAULT_BYTE_V2_TILE_POLICY


class ByteV2AttentionMetadataBuilder(AttentionMetadataBuilder[ByteV2AttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.tile_policy = getattr(
            kv_cache_spec,
            "tile_policy",
            ByteV2TilePolicy(
                alloc_block_tokens=kv_cache_spec.block_size,
                head_dim=kv_cache_spec.head_size,
                head_dim_v=getattr(
                    kv_cache_spec, "head_size_v", kv_cache_spec.head_size
                ),
            ),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ByteV2AttentionMetadata:
        del fast_build
        return ByteV2AttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            common_prefix_len=int(common_prefix_len),
            tile_policy=self.tile_policy,
        )

    def update_block_table(
        self,
        metadata: ByteV2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> ByteV2AttentionMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)


class ByteV2AttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "BYTE_V2"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return (
            block_size is None
            or block_size == DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens
        )

    @staticmethod
    def get_impl_cls() -> type[ByteV2AttentionImpl]:
        return ByteV2AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[ByteV2AttentionMetadataBuilder]:
        return ByteV2AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        tile_policy = byte_v2_tile_policy_from_env(
            block_size=block_size,
            head_dim=head_size,
            head_dim_v=head_size,
        )
        layout = ByteV2PageLayoutV4(
            tile_policy=tile_policy,
            num_kv_heads=num_kv_heads,
        )
        return (num_blocks, layout.page_size_bytes)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [DEFAULT_BYTE_V2_TILE_POLICY.head_dim]

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

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
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del head_size, dtype, kv_cache_dtype
        del use_mla, has_sink, use_sparse, use_mm_prefix
        if device_capability < DeviceCapability(8, 0):
            return "ByteV2 requires CUDA compute capability >= 8.0"
        if block_size not in (None, DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens):
            return "ByteV2 currently requires block_size=16"
        if not byte_v2_custom_ops_are_available():
            return _BYTE_V2_KERNELS_NOT_READY
        return None


class ByteV2RawStagingManager:
    """Reusable raw staging buffers for small ByteV2 cache updates."""

    def __init__(
        self,
        *,
        tile_policy: ByteV2TilePolicy,
        num_kv_heads: int,
        max_tokens_per_update: int = _BYTE_V2_MAX_RAW_STAGING_TOKENS,
    ) -> None:
        self.tile_policy = tile_policy
        self.raw_layout = ByteV2RawStagingLayout(
            tile_policy=tile_policy,
            num_kv_heads=num_kv_heads,
        )
        self.max_tokens_per_update = max_tokens_per_update

        self.raw_staging: torch.Tensor | None = None
        self.block_to_staging_slot: torch.Tensor | None = None
        self.staging_to_physical_block: torch.Tensor | None = None
        self.valid_rows: torch.Tensor | None = None
        self.next_staging_slot: torch.Tensor | None = None
        self.overflow: torch.Tensor | None = None

    def _should_stage(self, slot_mapping: torch.Tensor) -> bool:
        num_tokens = slot_mapping.shape[0]
        return 0 < num_tokens <= self.max_tokens_per_update

    def _needs_reallocation(
        self,
        *,
        kv_cache: torch.Tensor,
        num_staging_slots: int,
    ) -> bool:
        num_blocks = kv_cache.shape[0]
        slot_size_bytes = self.raw_layout.slot_size_bytes
        return (
            self.raw_staging is None
            or self.raw_staging.device != kv_cache.device
            or self.raw_staging.shape[0] < num_staging_slots
            or self.raw_staging.shape[1] != slot_size_bytes
            or self.block_to_staging_slot is None
            or self.block_to_staging_slot.device != kv_cache.device
            or self.block_to_staging_slot.shape[0] != num_blocks
            or self.staging_to_physical_block is None
            or self.staging_to_physical_block.device != kv_cache.device
            or self.staging_to_physical_block.shape[0] < num_staging_slots
            or self.valid_rows is None
            or self.valid_rows.device != kv_cache.device
            or self.valid_rows.shape[0] < num_staging_slots
            or self.next_staging_slot is None
            or self.next_staging_slot.device != kv_cache.device
            or self.next_staging_slot.shape[0] != 1
            or self.overflow is None
            or self.overflow.device != kv_cache.device
            or self.overflow.shape[0] != 1
        )

    def _ensure_capacity(
        self,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> bool:
        num_blocks = kv_cache.shape[0]
        num_staging_slots = min(num_blocks, slot_mapping.shape[0])
        if num_staging_slots <= 0:
            return False

        if not self._needs_reallocation(
            kv_cache=kv_cache,
            num_staging_slots=num_staging_slots,
        ):
            return True

        device = kv_cache.device
        self.raw_staging = torch.empty(
            (num_staging_slots, self.raw_layout.slot_size_bytes),
            dtype=torch.uint8,
            device=device,
        )
        self.block_to_staging_slot = torch.empty(
            (num_blocks,),
            dtype=torch.int32,
            device=device,
        )
        self.staging_to_physical_block = torch.empty(
            (num_staging_slots,),
            dtype=torch.int32,
            device=device,
        )
        self.valid_rows = torch.empty(
            (num_staging_slots,),
            dtype=torch.int32,
            device=device,
        )
        self.next_staging_slot = torch.empty((1,), dtype=torch.int32, device=device)
        self.overflow = torch.empty((1,), dtype=torch.int32, device=device)
        self._initialize_allocator_state()
        return True

    def _initialize_allocator_state(self) -> None:
        assert self.block_to_staging_slot is not None
        assert self.staging_to_physical_block is not None
        assert self.valid_rows is not None
        assert self.next_staging_slot is not None
        assert self.overflow is not None
        self.block_to_staging_slot.fill_(-1)
        self.staging_to_physical_block.fill_(-1)
        self.valid_rows.zero_()
        self.next_staging_slot.zero_()
        self.overflow.zero_()

    def _release_allocator_state(self) -> None:
        assert self.block_to_staging_slot is not None
        assert self.staging_to_physical_block is not None
        assert self.valid_rows is not None
        assert self.next_staging_slot is not None
        assert self.overflow is not None
        byte_v2_release_raw_staging(
            self.block_to_staging_slot,
            self.staging_to_physical_block,
            self.valid_rows,
            self.next_staging_slot,
            self.overflow,
        )

    def update(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> bool:
        if slot_mapping.shape[0] == 0:
            return True
        if slot_mapping.shape[0] == 1 and slot_mapping.is_cuda:
            try:
                byte_v2_update_cache_single_token(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                    codec_token_block=self.tile_policy.codec_token_block,
                    codec_dim_block=self.tile_policy.codec_dim_block,
                    alloc_block_tokens=self.tile_policy.alloc_block_tokens,
                )
                return True
            except NotImplementedError:
                pass
        if not self._should_stage(slot_mapping):
            return False
        if not self._ensure_capacity(kv_cache, slot_mapping):
            return False

        assert self.raw_staging is not None
        assert self.block_to_staging_slot is not None
        assert self.staging_to_physical_block is not None
        assert self.valid_rows is not None
        assert self.next_staging_slot is not None
        assert self.overflow is not None

        try:
            _debug_warmup(
                "raw staging prepare start slot_mapping=%s raw_staging=%s",
                tuple(slot_mapping.shape),
                tuple(self.raw_staging.shape),
            )
            byte_v2_prepare_raw_staging(
                slot_mapping,
                self.block_to_staging_slot,
                self.staging_to_physical_block,
                self.valid_rows,
                self.next_staging_slot,
                self.overflow,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            _debug_sync("raw staging prepare")
            _debug_warmup("raw staging prepare done")
            _debug_warmup("raw staging hydrate start")
            byte_v2_hydrate_raw_staging_from_cache(
                self.raw_staging,
                kv_cache,
                self.staging_to_physical_block,
                self.valid_rows,
                codec_token_block=self.tile_policy.codec_token_block,
                codec_dim_block=self.tile_policy.codec_dim_block,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            _debug_sync("raw staging hydrate")
            _debug_warmup("raw staging hydrate done")
            _debug_warmup("raw staging append start")
            byte_v2_append_raw_staging(
                key,
                value,
                self.raw_staging,
                slot_mapping,
                self.block_to_staging_slot,
                codec_token_block=self.tile_policy.codec_token_block,
                codec_dim_block=self.tile_policy.codec_dim_block,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            _debug_sync("raw staging append")
            _debug_warmup("raw staging append done")
            _debug_warmup("raw staging commit start")
            byte_v2_commit_raw_staging_to_cache(
                self.raw_staging,
                kv_cache,
                self.staging_to_physical_block,
                self.valid_rows,
                codec_token_block=self.tile_policy.codec_token_block,
                codec_dim_block=self.tile_policy.codec_dim_block,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            _debug_sync("raw staging commit")
            _debug_warmup("raw staging commit done")
            _debug_warmup("raw staging release start")
            self._release_allocator_state()
            _debug_sync("raw staging release")
            _debug_warmup("raw staging release done")
        except NotImplementedError:
            self._initialize_allocator_state()
            return False
        return True


class ByteV2AttentionImpl(AttentionImpl[ByteV2AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.tile_policy = byte_v2_tile_policy_from_env(
            head_dim=head_size,
            head_dim_v=head_size,
        )
        self.raw_staging_manager = ByteV2RawStagingManager(
            tile_policy=self.tile_policy,
            num_kv_heads=self.num_kv_heads,
        )
        self.prefill_backend = _prefill_backend()
        self.decode_raw_fallback = _decode_raw_fallback_enabled()
        self.decode_assume_no_outlier = _decode_assume_no_outlier_enabled()
        self.decode_gqa_packed = _decode_gqa_packed_enabled()
        self.decode_gqa_fa2_like = _decode_gqa_fa2_like_enabled()
        self.decode_gqa_fa2_direct = _decode_gqa_fa2_direct_enabled()
        self.decode_validate_no_outlier = _decode_validate_no_outlier_enabled()
        self.decode_page_unsafe_flags = _decode_page_unsafe_flags_enabled()
        if self.decode_gqa_packed and not self.decode_assume_no_outlier:
            logger.warning(
                "[ByteV2] disabling BYTE_V2_DECODE_GQA_PACKED because it "
                "requires BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1"
            )
            self.decode_gqa_packed = False
        if self.decode_gqa_fa2_like and not self.decode_gqa_packed:
            logger.warning(
                "[ByteV2] disabling BYTE_V2_DECODE_GQA_FA2_LIKE because it "
                "requires BYTE_V2_DECODE_GQA_PACKED=1"
            )
            self.decode_gqa_fa2_like = False
        if self.decode_gqa_fa2_direct and not self.decode_gqa_fa2_like:
            logger.warning(
                "[ByteV2] disabling BYTE_V2_DECODE_GQA_FA2_DIRECT because it "
                "requires BYTE_V2_DECODE_GQA_FA2_LIKE=1"
            )
            self.decode_gqa_fa2_direct = False
        if self.decode_gqa_fa2_like and self.tile_policy.compute_block_n != 64:
            logger.warning(
                "[ByteV2] disabling BYTE_V2_DECODE_GQA_FA2_LIKE because it "
                "currently requires compute_block_n=64"
            )
            self.decode_gqa_fa2_like = False
            self.decode_gqa_fa2_direct = False
        self.decode_gqa_packed_min_seq_len = _positive_int_env(
            "BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN",
            2048,
        )
        self.decode_gqa_packed_partition_size_explicit = (
            os.environ.get("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE") is not None
        )
        self.decode_gqa_packed_partition_size = _positive_int_env(
            "BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE",
            64,
        )
        self.decode_split_k = _decode_split_k_enabled()
        self.decode_split_k_partition_size_explicit = (
            os.environ.get("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE") is not None
        )
        self.decode_split_k_partition_size = _positive_int_env(
            "BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE",
            16,
        )
        long_partition_default = (
            self.decode_split_k_partition_size
            if os.environ.get("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE") is not None
            else 32
        )
        self.decode_split_k_long_partition_size_explicit = (
            os.environ.get("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE") is not None
        )
        self.decode_split_k_long_partition_size = _positive_int_env(
            "BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE",
            long_partition_default,
        )
        self.decode_split_k_long_min_seq_len = _positive_int_env(
            "BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN",
            4096,
        )
        self.decode_split_k_min_seq_len = _positive_int_env(
            "BYTE_V2_DECODE_SPLIT_K_MIN_SEQ_LEN",
            24,
        )
        self._decode_exp_sums: torch.Tensor | None = None
        self._decode_max_logits: torch.Tensor | None = None
        self._decode_tmp_out: torch.Tensor | None = None
        self._decode_cache_stats: torch.Tensor | None = None
        self._decode_page_unsafe_flags: torch.Tensor | None = None
        self._decode_page_unsafe_flags_cache_ptr: int | None = None

    def _use_gqa_packed_decode(self, max_seq_len: int) -> bool:
        if not self.decode_gqa_packed:
            return False
        if max_seq_len < self.decode_gqa_packed_min_seq_len:
            return False
        if self.num_kv_heads != 8:
            return False
        if self.num_heads % self.num_kv_heads != 0:
            return False
        q_heads_per_kv = self.num_heads // self.num_kv_heads
        if self.decode_gqa_fa2_direct:
            if q_heads_per_kv not in (1, 2, 4, 8, 16, 32):
                return False
        elif q_heads_per_kv != 4:
            return False
        if self.head_size != 128 or self.tile_policy.head_dim_v != 128:
            return False
        if self.tile_policy.alloc_block_tokens != 16:
            return False
        return self.tile_policy.compute_block_n in (64, 128)

    def _decode_tile_policy_tuple(
        self,
        max_seq_len: int,
        *,
        assume_no_outlier: bool | None = None,
        use_gqa_packed: bool | None = None,
    ) -> tuple[int, ...]:
        if assume_no_outlier is None:
            assume_no_outlier = self.decode_assume_no_outlier
        if use_gqa_packed is None:
            use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        policy = (
            self.tile_policy.codec_token_block,
            self.tile_policy.codec_dim_block,
            self.tile_policy.alloc_block_tokens,
            self.tile_policy.compute_block_n,
            self.tile_policy.head_dim,
            self.tile_policy.head_dim_v,
            int(self.decode_raw_fallback),
            int(assume_no_outlier),
            int(use_gqa_packed),
        )
        if self.decode_gqa_fa2_like and use_gqa_packed:
            if self.decode_gqa_fa2_direct:
                return policy + (1, 1, 0, 0, 1)
            return policy + (1,)
        return policy

    def _decode_partition_size(
        self,
        max_seq_len: int,
        *,
        num_decode_tokens: int = 1,
        use_gqa_packed: bool | None = None,
    ) -> int:
        if use_gqa_packed is None:
            use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        if use_gqa_packed:
            if self.decode_gqa_fa2_like:
                if self.decode_gqa_fa2_direct:
                    if not self.decode_gqa_packed_partition_size_explicit:
                        return self._auto_gqa_fa2_direct_partition_size(
                            max_seq_len,
                            num_decode_tokens,
                        )
                    return self.decode_gqa_packed_partition_size
                return self.tile_policy.compute_block_n
            if not self.decode_gqa_packed_partition_size_explicit:
                return self._auto_gqa_packed_partition_size(
                    max_seq_len,
                    num_decode_tokens,
                )
            return self.decode_gqa_packed_partition_size
        if not (
            self.decode_split_k_partition_size_explicit
            or self.decode_split_k_long_partition_size_explicit
        ):
            return self._auto_split_k_partition_size(
                max_seq_len,
                num_decode_tokens,
            )
        if max_seq_len >= self.decode_split_k_long_min_seq_len:
            return self.decode_split_k_long_partition_size
        return self.decode_split_k_partition_size

    def _auto_split_k_partition_size(
        self,
        max_seq_len: int,
        num_decode_tokens: int,
    ) -> int:
        # Page-loop split-k uses small partitions at short context, then
        # increases enough to reduce reduction/workspace overhead at long context.
        if num_decode_tokens >= 4:
            if max_seq_len >= 16384:
                return 128
            if max_seq_len >= 4096:
                return 64
            return 16
        if num_decode_tokens >= 2:
            if max_seq_len >= 4096:
                return 64
            return 16
        if max_seq_len >= 16384:
            return 128
        if max_seq_len >= 8192:
            return 64
        if max_seq_len >= 4096:
            return 32
        return 16

    def _auto_gqa_packed_partition_size(
        self,
        max_seq_len: int,
        num_decode_tokens: int,
    ) -> int:
        # GQA4 packed shares KV work across four query heads, so it benefits
        # from larger partitions at batch=1. Batch>=4 already provides more CTAs,
        # so the best measured partition is one notch smaller.
        if num_decode_tokens >= 4:
            if max_seq_len >= 8192:
                return 64
            if max_seq_len >= 2048:
                return 32
            return 16
        if num_decode_tokens >= 2:
            if max_seq_len >= 16384:
                return 64
            if max_seq_len >= 2048:
                return 32
            return 16
        if max_seq_len >= 16384:
            return 64
        if max_seq_len >= 8192:
            return 128
        if max_seq_len >= 4096:
            return 64
        if max_seq_len >= 2048:
            return 32
        return 16

    def _auto_gqa_fa2_direct_partition_size(
        self,
        max_seq_len: int,
        num_decode_tokens: int,
    ) -> int:
        # FA2-direct now runs multiple 64-token tiles inside a CTA. Keep small
        # partitions when decode batch already supplies CTAs; increase at
        # single-token long context to reduce split/reduce overhead.
        if num_decode_tokens >= 4:
            if max_seq_len >= 4096:
                return 128
            return 64
        if num_decode_tokens >= 2:
            if max_seq_len >= 4096:
                return 128
            return 64
        if max_seq_len >= 4096:
            return 256
        if max_seq_len >= 2048:
            return 128
        return 64

    def _use_split_k_decode(
        self,
        max_seq_len: int,
        *,
        num_decode_tokens: int = 1,
        use_gqa_packed: bool | None = None,
    ) -> bool:
        if not self.decode_split_k:
            return False
        if max_seq_len < self.decode_split_k_min_seq_len:
            return False
        partition_size = self._decode_partition_size(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        )
        if partition_size % self.tile_policy.alloc_block_tokens != 0:
            return False
        return (max_seq_len + partition_size - 1) // partition_size > 1

    def _get_decode_cache_stats(self, kv_cache: torch.Tensor) -> torch.Tensor:
        if (
            self._decode_cache_stats is None
            or self._decode_cache_stats.device != kv_cache.device
        ):
            self._decode_cache_stats = torch.empty(
                (4,),
                dtype=torch.int32,
                device=kv_cache.device,
            )
        return self._decode_cache_stats

    def _base_tile_policy_tuple(self) -> tuple[int, ...]:
        return (
            self.tile_policy.codec_token_block,
            self.tile_policy.codec_dim_block,
            self.tile_policy.alloc_block_tokens,
            self.tile_policy.compute_block_n,
            self.tile_policy.head_dim,
            self.tile_policy.head_dim_v,
        )

    def _get_decode_page_unsafe_flags(self, kv_cache: torch.Tensor) -> torch.Tensor:
        cache_ptr = kv_cache.data_ptr()
        if (
            self._decode_page_unsafe_flags is None
            or self._decode_page_unsafe_flags.device != kv_cache.device
            or self._decode_page_unsafe_flags.shape[0] < kv_cache.shape[0]
            or self._decode_page_unsafe_flags_cache_ptr != cache_ptr
        ):
            self._decode_page_unsafe_flags = torch.zeros(
                (kv_cache.shape[0],),
                dtype=torch.int32,
                device=kv_cache.device,
            )
            self._decode_page_unsafe_flags_cache_ptr = cache_ptr
        return self._decode_page_unsafe_flags

    def _update_decode_page_unsafe_flags(
        self,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not (self.decode_page_unsafe_flags and self.decode_assume_no_outlier):
            return
        if not (kv_cache.is_cuda and slot_mapping.is_cuda):
            return
        flags = self._get_decode_page_unsafe_flags(kv_cache)
        byte_v2_update_cache_unsafe_flags(
            flags,
            kv_cache,
            slot_mapping,
            tile_policy=self._base_tile_policy_tuple(),
        )

    def _usable_decode_page_unsafe_flags(
        self,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        flags = self._decode_page_unsafe_flags
        if flags is None:
            return None
        if flags.device != kv_cache.device:
            return None
        if flags.shape[0] < kv_cache.shape[0]:
            return None
        if self._decode_page_unsafe_flags_cache_ptr != kv_cache.data_ptr():
            return None
        return flags

    def _decode_cache_has_fallback_or_outlier(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> bool:
        stats = self._get_decode_cache_stats(kv_cache)
        byte_v2_collect_cache_stats(
            stats,
            kv_cache,
            block_table,
            seq_lens,
            max_seq_len=max_seq_len,
            tile_policy=self._base_tile_policy_tuple(),
        )
        return bool(stats[0].item())

    def _resolve_decode_fast_path(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> tuple[bool, bool]:
        assume_no_outlier = self.decode_assume_no_outlier
        use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        if not assume_no_outlier:
            return False, False
        if not self.decode_validate_no_outlier:
            return assume_no_outlier, use_gqa_packed
        if not (kv_cache.is_cuda and block_table.is_cuda and seq_lens.is_cuda):
            return False, False
        if self._decode_cache_has_fallback_or_outlier(
            kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
        ):
            return False, False
        return assume_no_outlier, use_gqa_packed

    def _get_split_k_workspace(
        self,
        output: torch.Tensor,
        max_seq_len: int,
        *,
        num_decode_tokens: int,
        use_gqa_packed: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        partition_size = self._decode_partition_size(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        )
        num_partitions = max(1, (max_seq_len + partition_size - 1) // partition_size)
        stats_shape = (output.shape[0], self.num_heads, num_partitions)
        tmp_shape = (*stats_shape, self.head_size)

        needs_alloc = (
            self._decode_exp_sums is None
            or tuple(self._decode_exp_sums.shape) != stats_shape
            or self._decode_exp_sums.device != output.device
        )
        if needs_alloc:
            self._decode_exp_sums = torch.empty(
                stats_shape,
                dtype=torch.float32,
                device=output.device,
            )
            self._decode_tmp_out = torch.empty(
                tmp_shape,
                dtype=torch.float32,
                device=output.device,
            )
        if (
            self._decode_max_logits is None
            or self._decode_max_logits.device != output.device
        ):
            self._decode_max_logits = torch.empty(
                (0,),
                dtype=torch.float32,
                device=output.device,
            )
        assert self._decode_exp_sums is not None
        assert self._decode_max_logits is not None
        assert self._decode_tmp_out is not None
        return self._decode_exp_sums, self._decode_max_logits, self._decode_tmp_out

    def _run_paged_decode(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> None:
        assume_no_outlier, use_gqa_packed = self._resolve_decode_fast_path(
            kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
        )
        page_unsafe_flags = self._usable_decode_page_unsafe_flags(kv_cache)
        if (
            assume_no_outlier
            and self.decode_page_unsafe_flags
            and not self.decode_validate_no_outlier
            and page_unsafe_flags is None
        ):
            assume_no_outlier = False
            use_gqa_packed = False
        tile_policy = self._decode_tile_policy_tuple(
            max_seq_len,
            assume_no_outlier=assume_no_outlier,
            use_gqa_packed=use_gqa_packed,
        )
        num_decode_tokens = int(output.shape[0])
        partition_size = self._decode_partition_size(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        )
        if output.is_cuda and self._use_split_k_decode(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        ):
            exp_sums, max_logits, tmp_out = self._get_split_k_workspace(
                output,
                max_seq_len,
                num_decode_tokens=num_decode_tokens,
                use_gqa_packed=use_gqa_packed,
            )
            if (
                assume_no_outlier
                and self.decode_page_unsafe_flags
                and page_unsafe_flags is not None
            ):
                byte_v2_paged_decode_attention_split_k_guarded(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_out,
                    query,
                    kv_cache,
                    page_unsafe_flags,
                    block_table,
                    seq_lens,
                    scale=self.scale,
                    num_kv_heads=self.num_kv_heads,
                    block_size=self.tile_policy.alloc_block_tokens,
                    max_seq_len=max_seq_len,
                    partition_size=partition_size,
                    tile_policy=tile_policy,
                )
                return

            byte_v2_paged_decode_attention_split_k(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                query,
                kv_cache,
                block_table,
                seq_lens,
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                block_size=self.tile_policy.alloc_block_tokens,
                max_seq_len=max_seq_len,
                partition_size=partition_size,
                tile_policy=tile_policy,
            )
            return

        byte_v2_paged_decode_attention(
            output,
            query,
            kv_cache,
            block_table,
            seq_lens,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            block_size=self.tile_policy.alloc_block_tokens,
            max_seq_len=max_seq_len,
            tile_policy=tile_policy,
        )

    def _forward_prefill_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        _debug_warmup(
            "prefill fallback start query=%s key=%s output=%s max_query_len=%s "
            "num_actual_tokens=%s",
            tuple(query.shape),
            None if key is None else tuple(key.shape),
            tuple(output.shape),
            attn_metadata.max_query_len,
            attn_metadata.num_actual_tokens,
        )
        if key is None or value is None:
            output.fill_(0)
            _debug_warmup("prefill fallback done without key/value")
            return output

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
        q_per_kv = self.num_heads // self.num_kv_heads

        for start, end in zip(query_start_locs[:-1], query_start_locs[1:]):
            start = min(int(start), num_actual_tokens)
            end = min(int(end), num_actual_tokens)
            if end <= start:
                continue

            q = query[start:end].transpose(0, 1).float()
            k = key[start:end]
            v = value[start:end]
            if q_per_kv != 1:
                k = k.repeat_interleave(q_per_kv, dim=1)
                v = v.repeat_interleave(q_per_kv, dim=1)
            k = k.transpose(0, 1).float()
            v = v.transpose(0, 1).float()

            scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
            if attn_metadata.causal:
                q_len = end - start
                causal_mask = torch.triu(
                    torch.ones(
                        (q_len, q_len),
                        dtype=torch.bool,
                        device=query.device,
                    ),
                    diagonal=1,
                )
                scores.masked_fill_(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            output[start:end].copy_(torch.matmul(probs, v).transpose(0, 1))

        _debug_warmup("prefill fallback done")
        return output

    def _forward_prefill_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
        q_per_kv = self.num_heads // self.num_kv_heads

        _debug_warmup(
            "prefill sdpa start query=%s key=%s output=%s max_query_len=%s "
            "num_actual_tokens=%s",
            tuple(query.shape),
            tuple(key.shape),
            tuple(output.shape),
            attn_metadata.max_query_len,
            num_actual_tokens,
        )
        for start, end in zip(query_start_locs[:-1], query_start_locs[1:]):
            start = min(int(start), num_actual_tokens)
            end = min(int(end), num_actual_tokens)
            if end <= start:
                continue

            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=attn_metadata.causal,
                scale=self.scale,
                enable_gqa=q_per_kv != 1,
            )
            output[start:end].copy_(attn_output.squeeze(0).transpose(0, 1))
        _debug_warmup("prefill sdpa done")
        return output

    def _forward_prefill_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        _debug_warmup(
            "prefill native start query=%s key=%s output=%s "
            "max_query_len=%s num_actual_tokens=%s",
            tuple(query.shape),
            tuple(key.shape),
            tuple(output.shape),
            attn_metadata.max_query_len,
            attn_metadata.num_actual_tokens,
        )
        byte_v2_prefill_attention(
            output,
            query,
            key,
            value,
            attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            causal=attn_metadata.causal,
            tile_policy=(
                self.tile_policy.codec_token_block,
                self.tile_policy.codec_dim_block,
                self.tile_policy.alloc_block_tokens,
                self.tile_policy.compute_block_n,
                self.tile_policy.head_dim,
                self.tile_policy.head_dim_v,
            ),
        )
        _debug_warmup("prefill native done")
        return output

    def _forward_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        if key is None or value is None:
            return output.fill_(0)
        if self._prefill_has_cached_context(attn_metadata):
            return self._forward_prefill_from_cache(
                query,
                kv_cache,
                output,
                attn_metadata,
            )
        if self.prefill_backend == "fallback":
            return self._forward_prefill_fallback(
                query,
                key,
                value,
                output,
                attn_metadata,
            )
        if self.prefill_backend == "native":
            try:
                return self._forward_prefill_native(
                    query,
                    key,
                    value,
                    output,
                    attn_metadata,
                )
            except NotImplementedError:
                return self._forward_prefill_fallback(
                    query,
                    key,
                    value,
                    output,
                    attn_metadata,
                )
        try:
            return self._forward_prefill_sdpa(
                query,
                key,
                value,
                output,
                attn_metadata,
            )
        except RuntimeError as err:
            logger.warning_once("[ByteV2] SDPA prefill failed; using fallback: %s", err)
            return self._forward_prefill_fallback(
                query,
                key,
                value,
                output,
                attn_metadata,
            )

    @staticmethod
    def _prefill_has_cached_context(
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        num_actual_tokens = attn_metadata.num_actual_tokens

        for seq_idx, (start, end) in enumerate(
            zip(query_start_locs[:-1], query_start_locs[1:])
        ):
            start = min(int(start), num_actual_tokens)
            end = min(int(end), num_actual_tokens)
            query_len = end - start
            if query_len <= 0 or seq_idx >= attn_metadata.seq_lens.shape[0]:
                continue
            if int(attn_metadata.seq_lens[seq_idx].item()) > query_len:
                return True
        return False

    def _forward_prefill_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        common_prefix_len = int(getattr(attn_metadata, "common_prefix_len", 0))

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])

        _debug_warmup(
            "prefill prefix-cache decode start query=%s output=%s "
            "common_prefix_len=%s num_actual_tokens=%s",
            tuple(query.shape),
            tuple(output.shape),
            common_prefix_len,
            num_actual_tokens,
        )
        for seq_idx, (start, end) in enumerate(
            zip(query_start_locs[:-1], query_start_locs[1:])
        ):
            start = min(int(start), num_actual_tokens)
            end = min(int(end), num_actual_tokens)
            if end <= start:
                continue

            query_len = end - start
            if seq_idx < attn_metadata.seq_lens.shape[0]:
                context_len = max(
                    int(attn_metadata.seq_lens[seq_idx].item()) - query_len,
                    0,
                )
            else:
                context_len = common_prefix_len
            block_table = attn_metadata.block_table[seq_idx : seq_idx + 1]
            seq_lens = torch.arange(
                context_len + 1,
                context_len + query_len + 1,
                dtype=torch.int32,
                device=query.device,
            )
            self._run_paged_decode(
                output[start:end],
                query[start:end],
                kv_cache,
                block_table.expand(query_len, -1),
                seq_lens,
                max_seq_len=context_len + query_len,
            )
        _debug_warmup("prefill prefix-cache decode done")
        return output

    @staticmethod
    def _is_decode_compatible(
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        return (
            attn_metadata.max_query_len == 1
            and attn_metadata.block_table.dim() == 2
            and attn_metadata.block_table.shape[0] >= output.shape[0]
            and attn_metadata.seq_lens.shape[0] >= output.shape[0]
        )

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, output_scale, output_block_scale
        if attn_metadata is None:
            _debug_warmup("forward profile run with empty metadata")
            return output.fill_(0)
        if not self._is_decode_compatible(output, attn_metadata):
            return self._forward_prefill(
                query,
                key,
                value,
                kv_cache,
                output,
                attn_metadata,
            )
        _debug_warmup(
            "decode forward start query=%s output=%s block_table=%s seq_lens=%s "
            "max_seq_len=%s",
            tuple(query.shape),
            tuple(output.shape),
            tuple(attn_metadata.block_table.shape),
            tuple(attn_metadata.seq_lens.shape),
            attn_metadata.max_seq_len,
        )
        self._run_paged_decode(
            output,
            query,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
        )
        _debug_warmup("decode forward done")
        return output

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        del layer
        _debug_warmup(
            "kv update start key=%s kv_cache=%s slot_mapping=%s",
            tuple(key.shape),
            tuple(kv_cache.shape),
            tuple(slot_mapping.shape),
        )
        if self.raw_staging_manager.update(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
        ):
            self._update_decode_page_unsafe_flags(kv_cache, slot_mapping)
            _debug_warmup("kv update done via cache update manager")
            return
        byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            codec_token_block=self.tile_policy.codec_token_block,
            codec_dim_block=self.tile_policy.codec_dim_block,
            alloc_block_tokens=self.tile_policy.alloc_block_tokens,
        )
        self._update_decode_page_unsafe_flags(kv_cache, slot_mapping)
        _debug_warmup("kv update done via direct cache writer")
