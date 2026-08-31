# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import os
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.byte_v2_layout import ByteV2TilePolicy

logger = init_logger(__name__)


BYTE_V2_DEFAULT_RAW_STAGING_SLOTS = 128
_BYTE_V2_RAW_STAGING_SLOTS_ENV = "BYTE_V2_FA2_RAW_STAGING_SLOTS"
_BYTE_V2_DECODED_PREFIX_CACHE_SLOTS_ENV = (
    "BYTE_V2_STATIC_W16_DECODED_PREFIX_CACHE_SLOTS"
)


def byte_v2_hybrid_raw_fallback_enabled() -> bool:
    """Return whether the experimental persistent ByteV2 sidecar is enabled."""
    value = os.environ.get("BYTE_V2_FA2_HYBRID_RAW_FALLBACK")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def byte_v2_hybrid_raw_mutable_tail_q1_enabled(
    *,
    hybrid_raw_fallback: bool | None = None,
) -> bool:
    """Return whether hybrid Q1 writes should retain a mutable raw tail."""
    value = os.environ.get("BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1")
    if value is None:
        if hybrid_raw_fallback is None:
            hybrid_raw_fallback = byte_v2_hybrid_raw_fallback_enabled()
        return hybrid_raw_fallback and not byte_v2_test_forced_raw_promotion_enabled()
    return value.lower() not in ("0", "false", "no", "off")


def resolve_byte_v2_hybrid_raw_mutable_tail_q1(
    *,
    native_available: bool,
    hybrid_raw_fallback: bool | None = None,
) -> bool:
    """Resolve raw-tail Q1 against the installed native operator ABI.

    An implicit default may fall back to the legacy writer when an older
    extension is installed. An explicit opt-in instead fails closed so cache
    planning and request admission cannot disagree with the attention path.
    """
    if not byte_v2_hybrid_raw_mutable_tail_q1_enabled(
        hybrid_raw_fallback=hybrid_raw_fallback
    ):
        return False
    if native_available:
        return True
    if os.environ.get("BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1") is not None:
        raise RuntimeError(
            "BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1=1 requires native raw-tail "
            "Q1 and safe Q>1 demotion operator schemas"
        )
    return False


def byte_v2_test_forced_raw_promotion_enabled() -> bool:
    """Return whether the test-only forced raw lifecycle hook is enabled."""
    value = os.environ.get("BYTE_V2_TEST_FORCE_RAW_PROMOTION")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def byte_v2_raw_staging_slots() -> int:
    """Return the strictly validated shared ByteV2 staging-slot count."""
    value = os.environ.get(_BYTE_V2_RAW_STAGING_SLOTS_ENV)
    if value is None:
        return BYTE_V2_DEFAULT_RAW_STAGING_SLOTS
    try:
        slots = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_BYTE_V2_RAW_STAGING_SLOTS_ENV} must be a positive integer, "
            f"got {value!r}"
        ) from error
    if slots <= 0:
        raise ValueError(
            f"{_BYTE_V2_RAW_STAGING_SLOTS_ENV} must be positive, got {value!r}"
        )
    return slots


def byte_v2_decoded_prefix_cache_slots() -> int:
    """Return the per-layer decoded-prefix cache capacity in raw pages."""
    value = os.environ.get(_BYTE_V2_DECODED_PREFIX_CACHE_SLOTS_ENV)
    if value is None:
        return 0
    try:
        slots = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_BYTE_V2_DECODED_PREFIX_CACHE_SLOTS_ENV} must be a "
            f"non-negative integer, got {value!r}"
        ) from error
    if slots < 0:
        raise ValueError(
            f"{_BYTE_V2_DECODED_PREFIX_CACHE_SLOTS_ENV} must be non-negative, "
            f"got {value!r}"
        )
    return slots


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    NVFP4 = 4  # packed fp4 data + fp8 block scales

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


class KVCacheSpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    MAMBA = "mamba"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:
        return self.block_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        if self.kv_quant_mode.is_per_token_head:
            real_page_size += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout: fp4 data + fp8 block scales per head.
            full_dim = nvfp4_kv_cache_full_dim(self.head_size)
            return (
                2
                * self.block_size
                * self.num_kv_heads
                * full_dim
                * get_dtype_size(self.dtype)
            )
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout per head: fp4 data + fp8 block scales.
            # fp4 data: head_size//2 bytes (2 fp4 values per byte)
            # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with TQ-aware page size.

    Python equivalent of the C++ TQ4FullAttentionSpec. Overrides
    real_page_size_bytes to use TQ slot bytes instead of the raw
    head_size * dtype formula.
    """

    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs), (
            "All TQ layers in the same KV cache group must use the same tq_slot_size."
        )
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)


def _default_byte_v2_tile_policy() -> ByteV2TilePolicy:
    from vllm.v1.attention.backends.byte_v2_layout import (
        byte_v2_tile_policy_from_env,
    )

    return byte_v2_tile_policy_from_env()


@dataclass(frozen=True, kw_only=True)
class ByteV2FullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with ByteV2 byte-page layout sizing.

    ByteV2 stores compressed K/V payloads plus per-page metadata in a custom
    byte layout. The raw KV allocation therefore uses ``torch.uint8`` and the
    page size comes from ``ByteV2PageLayoutV6`` instead of the normal
    ``head_size * dtype`` formula.
    """

    tile_policy: ByteV2TilePolicy = field(default_factory=_default_byte_v2_tile_policy)
    codec_low_bytes_per_elem: int = 1
    codec_exponent_code_bits: int = 4
    page_header_bytes: int = 128
    kv_head_meta_bytes: int = 96
    alignment_bytes: int = 128
    outlier_value_bits: int = 8
    outlier_entries_per_tile: int = 256
    outlier_pool_entries: int = 256
    include_raw_payload: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.dtype != torch.uint8:
            raise ValueError("ByteV2FullAttentionSpec requires dtype=torch.uint8")
        if self.kv_quant_mode != KVQuantMode.NONE:
            raise ValueError("ByteV2FullAttentionSpec does not use vLLM KVQuantMode")
        if self.head_size_v != self.head_size:
            raise ValueError("ByteV2 currently requires head_size_v == head_size")
        object.__setattr__(
            self,
            "tile_policy",
            self.tile_policy.with_updates(
                alloc_block_tokens=self.block_size,
                head_dim=self.head_size,
                head_dim_v=self.head_size_v,
            ),
        )
        compiled_layout = {
            "codec_low_bytes_per_elem": 1,
            "codec_exponent_code_bits": 4,
            "page_header_bytes": 128,
            "kv_head_meta_bytes": 96,
            "alignment_bytes": 128,
            "outlier_value_bits": 8,
            "outlier_entries_per_tile": 256,
            "outlier_pool_entries": 256,
            "include_raw_payload": False,
        }
        mismatches = [
            name
            for name, compiled_value in compiled_layout.items()
            if getattr(self, name) != compiled_value
        ]
        if mismatches:
            raise ValueError(
                "ByteV2FullAttentionSpec does not match the compiled V6 CUDA "
                f"layout; unsupported fields: {', '.join(mismatches)}"
            )

    def _page_layout(self):
        from vllm.v1.attention.backends.byte_v2_layout import (
            ByteV2CodecPayloadPolicy,
            ByteV2PageLayoutV6,
        )

        return ByteV2PageLayoutV6(
            tile_policy=self.tile_policy,
            codec_payload_policy=ByteV2CodecPayloadPolicy(
                low_bytes_per_elem=self.codec_low_bytes_per_elem,
                exponent_code_bits=self.codec_exponent_code_bits,
            ),
            num_kv_heads=self.num_kv_heads,
            page_header_bytes=self.page_header_bytes,
            kv_head_meta_bytes=self.kv_head_meta_bytes,
            alignment_bytes=self.alignment_bytes,
            outlier_value_bits=self.outlier_value_bits,
            outlier_entries_per_tile=self.outlier_entries_per_tile,
            outlier_pool_entries=self.outlier_pool_entries,
            include_raw_payload=self.include_raw_payload,
        )

    @property
    def real_page_size_bytes(self) -> int:
        from vllm.v1.attention.backends.byte_v2_static_w16 import (
            STATIC_W16_PAGE_BYTES,
            byte_v2_static_w16_requested,
        )

        if byte_v2_static_w16_requested():
            return STATIC_W16_PAGE_BYTES
        return self._page_layout().page_size_bytes

    @property
    def page_metadata_size_bytes(self) -> int:
        """Return the metadata range size reset before block reuse."""
        from vllm.v1.attention.backends.byte_v2_static_w16 import (
            byte_v2_static_w16_requested,
        )

        if byte_v2_static_w16_requested():
            return 128
        return self._page_layout().aligned_metadata_bytes

    @property
    def page_metadata_offset_bytes(self) -> int:
        """Return the byte offset of metadata reset on block reuse."""
        from vllm.v1.attention.backends.byte_v2_static_w16 import (
            byte_v2_static_w16_requested,
        )

        if byte_v2_static_w16_requested():
            return 49_152
        return 0

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(
            self,
            block_size=block_size,
            tile_policy=self.tile_policy.with_updates(alloc_block_tokens=block_size),
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        first = specs[0]
        assert all(isinstance(s, ByteV2FullAttentionSpec) for s in specs), (
            "All ByteV2 layers in the same KV cache group must use "
            "ByteV2FullAttentionSpec."
        )
        for spec in specs[1:]:
            assert spec.tile_policy == first.tile_policy, (
                "All ByteV2 layers in the same KV cache group must use the "
                "same tile policy."
            )
            assert (
                spec.codec_low_bytes_per_elem == first.codec_low_bytes_per_elem
                and spec.codec_exponent_code_bits == first.codec_exponent_code_bits
                and spec.page_header_bytes == first.page_header_bytes
                and spec.kv_head_meta_bytes == first.kv_head_meta_bytes
                and spec.alignment_bytes == first.alignment_bytes
                and spec.outlier_value_bits == first.outlier_value_bits
                and spec.outlier_entries_per_tile == first.outlier_entries_per_tile
                and spec.outlier_pool_entries == first.outlier_pool_entries
                and spec.include_raw_payload == first.include_raw_payload
            ), (
                "All ByteV2 layers in the same KV cache group must use the "
                "same page layout policy."
            )
        return replace(
            merged,
            tile_policy=first.tile_policy,
            codec_low_bytes_per_elem=first.codec_low_bytes_per_elem,
            codec_exponent_code_bits=first.codec_exponent_code_bits,
            page_header_bytes=first.page_header_bytes,
            kv_head_meta_bytes=first.kv_head_meta_bytes,
            alignment_bytes=first.alignment_bytes,
            outlier_value_bits=first.outlier_value_bits,
            outlier_entries_per_tile=first.outlier_entries_per_tile,
            outlier_pool_entries=first.outlier_pool_entries,
            include_raw_payload=first.include_raw_payload,
        )


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1  # Default to 1 for no compression.
    model_version: str | None = None

    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, and model version."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.
        """
        # During chunked prefill, we hold KV for at most one chunk window.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def real_page_size_bytes(self) -> int:
        # Mirror ``FullAttentionSpec.real_page_size_bytes`` for NVFP4 KV cache.
        if self.kv_quant_mode.is_nvfp4:
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the newly scheduled tokens, and never more
        # than `max_model_len`.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowMLASpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == self.num_speculative_blocks
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # NOTE: below util functions are only used by DeepseekV4 for now.
    def get_page_sizes(self) -> list[int]:
        return list(set(spec.page_size_bytes for spec in self.kv_cache_specs.values()))

    def get_num_layer_tuples(self) -> int:
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, SinkFullAttentionSpec):
        return KVCacheSpecKind.SINK_FULL_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """
    byte_v2_raw_fallback_slots: int = 0
    """Persistent raw fallback slots allocated per actual ByteV2 layer."""
    byte_v2_raw_fallback_sidecar_bytes: int = 0
    """Total persistent ByteV2 sidecar bytes for this worker."""
    byte_v2_raw_staging_slots: int = 0
    """Raw staging slots in the runner-owned cross-layer workspace."""
    byte_v2_raw_staging_workspace_bytes: int = 0
    """Total bytes in the runner-owned ByteV2 staging workspace."""
    byte_v2_decoded_prefix_cache_slots: int = 0
    """Persistent decoded-prefix raw pages allocated per ByteV2 layer."""
    byte_v2_decoded_prefix_cache_bytes: int = 0
    """Total persistent decoded-prefix cache bytes for this worker."""
    byte_v2_raw_mutable_tail_q1: bool = False
    """Whether this cache plan reserves one mutable raw tail per request."""
    byte_v2_static_w16_retain_cascade_q16: bool = False
    """Whether this plan reserves one retained cascade-Q16 page per request."""

    @property
    def has_mamba_layers(self) -> bool:
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    @property
    def has_byte_v2_layers(self) -> bool:
        return self.num_byte_v2_layers > 0

    @property
    def num_byte_v2_layers(self) -> int:
        """Return the number of actual ByteV2 layers on this worker."""
        num_layers = 0
        for group in self.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, ByteV2FullAttentionSpec):
                num_layers += len(group.layer_names)
            elif isinstance(spec, UniformTypeKVCacheSpecs):
                num_layers += sum(
                    isinstance(spec.kv_cache_specs[layer_name], ByteV2FullAttentionSpec)
                    for layer_name in group.layer_names
                )
        return num_layers

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        return self.has_mamba_layers or self.byte_v2_raw_fallback_slots > 0
