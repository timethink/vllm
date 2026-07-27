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
    ByteV2PageLayoutV6,
    ByteV2RawStagingLayout,
    ByteV2TilePolicy,
    byte_v2_tile_policy_from_env,
)
from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_append_raw_staging,
    byte_v2_collect_cache_stats,
    byte_v2_commit_raw_staging_to_cache,
    byte_v2_commit_raw_staging_to_hybrid_cache,
    byte_v2_custom_ops_are_available,
    byte_v2_fa2_decode_is_available,
    byte_v2_fa2_direct_paged_prefill_attention,
    byte_v2_fa2_hybrid_decode_is_available,
    byte_v2_fa2_hybrid_paged_decode_attention,
    byte_v2_fa2_paged_decode_attention,
    byte_v2_fa2_raw_staging_prefill_attention,
    byte_v2_hybrid_cache_update_is_available,
    byte_v2_hybrid_raw_tail_q1_is_available,
    byte_v2_hydrate_raw_staging_from_cache,
    byte_v2_hydrate_raw_staging_from_hybrid_cache,
    byte_v2_paged_decode_attention,
    byte_v2_paged_decode_attention_split_k,
    byte_v2_paged_decode_attention_split_k_guarded,
    byte_v2_prefill_attention,
    byte_v2_prepare_raw_staging,
    byte_v2_release_raw_staging,
    byte_v2_release_raw_staging_and_update_flags,
    byte_v2_reset_raw_fallback_pages,
    byte_v2_reshape_and_cache,
    byte_v2_speculative_verify_gqa,
    byte_v2_speculative_verify_ragged_q4,
    byte_v2_test_force_promote_raw_staging_q1,
    byte_v2_test_forced_raw_promotion_is_available,
    byte_v2_update_cache_raw_staging,
    byte_v2_update_cache_single_token,
    byte_v2_update_cache_unsafe_flags,
    byte_v2_update_hybrid_cache_raw_staging_multi_token,
    byte_v2_update_hybrid_cache_raw_staging_multi_token_retained,
    byte_v2_update_hybrid_cache_raw_staging_q1,
    byte_v2_update_hybrid_cache_raw_tail_q1,
)
from vllm.v1.kv_cache_interface import (
    byte_v2_hybrid_raw_fallback_enabled,
    byte_v2_hybrid_raw_mutable_tail_q1_enabled,
    byte_v2_test_forced_raw_promotion_enabled,
    resolve_byte_v2_hybrid_raw_mutable_tail_q1,
)

_BYTE_V2_KERNELS_NOT_READY = "ByteV2 native CUDA kernels are not registered yet"
_BYTE_V2_MAX_RAW_STAGING_TOKENS = 1024
_BYTE_V2_UNNORMALIZED_PARTITION_OUTPUT_MODE = 27
_BYTE_V2_FA2_PAGE_SIZE_BYTES = ByteV2PageLayoutV6().page_size_bytes
_BYTE_V2_CACHED_PREFILL_AUTO_MIN_QUERY_LEN = 16_384
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


def _decode_kernel_mode() -> str:
    value = os.environ.get("BYTE_V2_DECODE_KERNEL")
    if value is None:
        return "legacy"
    mode = value.lower()
    if mode not in ("legacy", "fa2", "auto"):
        logger.warning("[ByteV2] ignoring invalid BYTE_V2_DECODE_KERNEL=%r", value)
        return "legacy"
    return mode


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


def _decode_unnormalized_partition_output_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT")
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


def _fused_staging_release_flags_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FUSED_STAGING_RELEASE_FLAGS")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _native_raw_staging_update_enabled() -> bool:
    value = os.environ.get("BYTE_V2_NATIVE_RAW_STAGING_UPDATE")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _native_single_token_update_enabled() -> bool:
    value = os.environ.get("BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _fused_single_token_staging_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FUSED_SINGLE_TOKEN_STAGING")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _fused_single_token_commit_release_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _fused_single_token_stage_metadata_clear_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _fused_commit_metadata_clear_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FUSED_COMMIT_METADATA_CLEAR")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _warp_parallel_commit_histogram_enabled() -> bool:
    value = os.environ.get("BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _decode_split_k_enabled() -> bool:
    value = os.environ.get("BYTE_V2_DECODE_SPLIT_K")
    if value is None:
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _speculative_verify_q4_enabled() -> bool:
    value = os.environ.get("BYTE_V2_SPECULATIVE_VERIFY_Q4")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _speculative_verify_gqa_enabled() -> bool:
    value = os.environ.get("BYTE_V2_SPECULATIVE_VERIFY_GQA")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _speculative_verify_ragged_q4_enabled() -> bool:
    value = os.environ.get("BYTE_V2_SPECULATIVE_VERIFY_RAGGED_Q4")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _cached_prefix_q16_enabled() -> bool:
    value = os.environ.get("BYTE_V2_CACHED_PREFIX_Q16")
    if value is None:
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _direct_paged_prefill_enabled() -> bool:
    value = os.environ.get("BYTE_V2_FA2_DIRECT_PREFILL")
    if value is None:
        return True
    if value not in ("0", "1"):
        raise ValueError(
            f"BYTE_V2_FA2_DIRECT_PREFILL must be unset, 0, or 1; got {value!r}"
        )
    return value == "1"


def _cached_prefill_hydrate_to_raw_mode() -> str:
    value = os.environ.get("BYTE_V2_FA2_CACHED_PREFILL_HYDRATE_TO_RAW")
    if value is None:
        return "auto"
    if value not in ("0", "1"):
        raise ValueError(
            "BYTE_V2_FA2_CACHED_PREFILL_HYDRATE_TO_RAW must be unset, 0, or "
            f"1; got {value!r}"
        )
    return "enabled" if value == "1" else "disabled"


def _cached_prefill_hydrate_to_raw_enabled() -> bool:
    return _cached_prefill_hydrate_to_raw_mode() != "disabled"


def _hybrid_raw_mutable_tail_q1_enabled(
    *,
    hybrid_raw_fallback: bool,
) -> bool:
    return byte_v2_hybrid_raw_mutable_tail_q1_enabled(
        hybrid_raw_fallback=hybrid_raw_fallback
    )


def _hybrid_raw_tail_fused_finalize_enabled() -> bool:
    value = os.environ.get("BYTE_V2_HYBRID_RAW_TAIL_FUSED_FINALIZE")
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


@dataclass(frozen=True)
class ByteV2DirectPrefillGroup:
    """One zero-copy raw-paged group with page-aligned internal boundaries."""

    first_request: int
    num_requests: int
    first_token: int
    num_tokens: int
    rounded_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    block_table: torch.Tensor


@dataclass(frozen=True)
class ByteV2DirectPrefillPlan:
    """Cached-prefix and direct-initial-suffix routing for one scheduler step."""

    cached_request_count: int
    cached_token_count: int
    cached_max_query_len: int
    cached_max_seq_len: int
    groups: tuple[ByteV2DirectPrefillGroup, ...]


@dataclass
class ByteV2AttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    seq_lens_cpu_upper_bound: torch.Tensor | None
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool
    is_prefilling: torch.Tensor | None = None
    seq_len_sum: int | None = None
    common_prefix_len: int = 0
    tile_policy: ByteV2TilePolicy = DEFAULT_BYTE_V2_TILE_POLICY
    direct_prefill_plan: ByteV2DirectPrefillPlan | None = None


def _build_byte_v2_direct_prefill_plan(
    *,
    query_start_loc_cpu: torch.Tensor,
    query_start_loc: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None,
    num_reqs: int,
    num_actual_tokens: int,
    device: torch.device,
    block_size: int,
) -> ByteV2DirectPrefillPlan | None:
    """Build a fail-closed cached-prefix/initial-suffix direct-paged plan."""
    query_start_loc_device = (
        query_start_loc_cpu if query_start_loc is None else query_start_loc
    )
    if (
        seq_lens_cpu is None
        or num_reqs <= 0
        or block_size <= 0
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.shape[0] < num_reqs + 1
        or query_start_loc_device.device != device
        or query_start_loc_device.dtype != torch.int32
        or query_start_loc_device.ndim != 1
        or query_start_loc_device.shape[0] < num_reqs + 1
        or not query_start_loc_device.is_contiguous()
        or seq_lens_cpu.device.type != "cpu"
        or seq_lens_cpu.ndim != 1
        or seq_lens_cpu.shape[0] < num_reqs
    ):
        return None

    starts = [int(value) for value in query_start_loc_cpu[: num_reqs + 1]]
    if (
        starts[0] != 0
        or starts[-1] != num_actual_tokens
        or any(end <= start for start, end in zip(starts, starts[1:]))
    ):
        return None
    query_lens = [end - start for start, end in zip(starts, starts[1:])]
    seq_lens = [int(value) for value in seq_lens_cpu[:num_reqs]]
    context_lens = [
        seq_len - query_len for seq_len, query_len in zip(seq_lens, query_lens)
    ]
    if any(context_len < 0 for context_len in context_lens):
        return None

    first_initial = next(
        (index for index, context_len in enumerate(context_lens) if context_len == 0),
        num_reqs,
    )
    if first_initial == num_reqs:
        return None
    if any(context_len <= 0 for context_len in context_lens[:first_initial]) or any(
        context_len != 0 for context_len in context_lens[first_initial:]
    ):
        return None

    groups = []
    request_index = first_initial
    while request_index < num_reqs:
        group_first_request = request_index
        request_index += 1
        while (
            request_index < num_reqs and query_lens[request_index - 1] % block_size == 0
        ):
            request_index += 1
        group_request_end = request_index
        first_token = starts[group_first_request]
        token_end = starts[group_request_end]
        num_tokens = token_end - first_token
        rounded_tokens = (num_tokens + block_size - 1) // block_size * block_size
        group_query_lens = query_lens[group_first_request:group_request_end]
        local_starts = [
            starts[index] - first_token
            for index in range(group_first_request, group_request_end + 1)
        ]
        if any(local_start % block_size for local_start in local_starts[:-1]):
            return None
        blocks_per_request = [
            (query_len + block_size - 1) // block_size for query_len in group_query_lens
        ]
        max_blocks = max(blocks_per_request)
        local_block_table = torch.zeros(
            (len(group_query_lens), max_blocks),
            dtype=torch.int32,
            device=device,
        )
        for row, (local_start, num_blocks) in enumerate(
            zip(local_starts, blocks_per_request)
        ):
            first_page = local_start // block_size
            local_block_table[row, :num_blocks] = torch.arange(
                first_page,
                first_page + num_blocks,
                dtype=torch.int32,
                device=device,
            )
        local_query_start_loc = query_start_loc_device[
            group_first_request : group_request_end + 1
        ]
        if first_token:
            local_query_start_loc = local_query_start_loc - first_token
        groups.append(
            ByteV2DirectPrefillGroup(
                first_request=group_first_request,
                num_requests=len(group_query_lens),
                first_token=first_token,
                num_tokens=num_tokens,
                rounded_tokens=rounded_tokens,
                max_query_len=max(group_query_lens),
                query_start_loc=local_query_start_loc,
                block_table=local_block_table,
            )
        )

    return ByteV2DirectPrefillPlan(
        cached_request_count=first_initial,
        cached_token_count=starts[first_initial],
        cached_max_query_len=(max(query_lens[:first_initial], default=0)),
        cached_max_seq_len=max(seq_lens[:first_initial], default=0),
        groups=tuple(groups),
    )


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
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        seq_lens_cpu_upper_bound = getattr(
            common_attn_metadata,
            "seq_lens_cpu_upper_bound",
            None,
        )
        prefill_seq_lens_cpu = seq_lens_cpu
        if prefill_seq_lens_cpu is None:
            prefill_seq_lens_cpu = seq_lens_cpu_upper_bound
        if seq_lens_cpu_upper_bound is None:
            seq_lens_cpu_upper_bound = seq_lens_cpu
        num_reqs = int(
            getattr(
                common_attn_metadata,
                "num_reqs",
                0 if seq_lens_cpu is None else seq_lens_cpu.shape[0],
            )
        )
        seq_len_sum = (
            int(seq_lens_cpu[:num_reqs].sum().item())
            if seq_lens_cpu is not None and num_reqs > 0
            else None
        )
        direct_prefill_plan = None
        if (
            _direct_paged_prefill_enabled()
            and byte_v2_hybrid_raw_fallback_enabled()
            and common_attn_metadata.causal
            and common_attn_metadata.max_query_len > 1
        ):
            direct_prefill_plan = _build_byte_v2_direct_prefill_plan(
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                query_start_loc=common_attn_metadata.query_start_loc,
                seq_lens_cpu=prefill_seq_lens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=common_attn_metadata.num_actual_tokens,
                device=common_attn_metadata.query_start_loc.device,
                block_size=self.tile_policy.alloc_block_tokens,
            )
        return ByteV2AttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            is_prefilling=getattr(common_attn_metadata, "is_prefilling", None),
            seq_len_sum=seq_len_sum,
            common_prefix_len=int(common_prefix_len),
            tile_policy=self.tile_policy,
            direct_prefill_plan=direct_prefill_plan,
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
        layout = ByteV2PageLayoutV6(
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


@dataclass(frozen=True)
class ByteV2RawFallbackState:
    """Device tensors backing the experimental compact/raw page store."""

    raw_pages: torch.Tensor
    page_to_raw_slot: torch.Tensor
    free_slots: torch.Tensor
    free_count: torch.Tensor
    fatal: torch.Tensor
    forced_raw_diagnostic: torch.Tensor | None = None


class ByteV2RawFallbackStore:
    """Lazy persistent raw-page sidecar for an experimental runtime checkpoint.

    This store is deliberately separate from the transient update staging
    buffers. Its persistent allocation is included in the KV-cache plan and
    its block lifetime is driven by the scheduler's cache-zero/reset
    notifications. The feature remains experimental and default-off while
    unsupported runtime integrations are being made fail-closed. A KV-cache
    address, device, or shape change after first use is rejected because raw
    page state cannot be transferred safely to a new binding.
    """

    def __init__(
        self,
        *,
        raw_layout: ByteV2RawStagingLayout,
        forced_raw_diagnostic: bool = False,
    ) -> None:
        self.raw_layout = raw_layout
        self.forced_raw_diagnostic = forced_raw_diagnostic
        self._binding: tuple[int, torch.device, tuple[int, ...]] | None = None
        self._state: ByteV2RawFallbackState | None = None
        self._planned_num_blocks: int | None = None
        self._planned_num_raw_slots: int | None = None

    @property
    def current_state(self) -> ByteV2RawFallbackState | None:
        """Return the currently bound state without allocating it."""
        return self._state

    @staticmethod
    def _binding_for(
        kv_cache: torch.Tensor,
    ) -> tuple[int, torch.device, tuple[int, ...]]:
        return (
            kv_cache.data_ptr(),
            kv_cache.device,
            tuple(int(dim) for dim in kv_cache.shape),
        )

    @staticmethod
    def _num_raw_slots(num_blocks: int) -> int:
        return _positive_int_env(
            "BYTE_V2_FA2_RAW_FALLBACK_SLOTS",
            max(1, num_blocks // 256),
        )

    def _allocate_state(self, kv_cache: torch.Tensor) -> ByteV2RawFallbackState:
        num_blocks = int(kv_cache.shape[0])
        if (
            self._planned_num_blocks is not None
            and num_blocks != self._planned_num_blocks
        ):
            raise RuntimeError(
                "ByteV2 raw fallback plan expected "
                f"{self._planned_num_blocks} blocks, but the bound KV cache has "
                f"{num_blocks} blocks"
            )
        num_raw_slots = self._planned_num_raw_slots
        if num_raw_slots is None:
            num_raw_slots = self._num_raw_slots(num_blocks)
        device = kv_cache.device
        return ByteV2RawFallbackState(
            raw_pages=torch.empty(
                (num_raw_slots, self.raw_layout.slot_size_bytes),
                dtype=torch.uint8,
                device=device,
            ),
            page_to_raw_slot=torch.full(
                (num_blocks,),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            free_slots=torch.arange(
                num_raw_slots,
                dtype=torch.int32,
                device=device,
            ),
            free_count=torch.full(
                (1,),
                num_raw_slots,
                dtype=torch.int32,
                device=device,
            ),
            fatal=torch.zeros((1,), dtype=torch.int32, device=device),
            forced_raw_diagnostic=(
                torch.zeros((3,), dtype=torch.int32, device=device)
                if self.forced_raw_diagnostic
                else None
            ),
        )

    def bind_plan(self, *, num_blocks: int, num_raw_slots: int) -> None:
        """Bind the planner's persistent sidecar dimensions before first use."""
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_raw_slots <= 0:
            raise ValueError("num_raw_slots must be positive")
        if self._state is not None and (
            num_blocks != self._planned_num_blocks
            or num_raw_slots != self._planned_num_raw_slots
        ):
            raise RuntimeError(
                "Cannot change the ByteV2 raw fallback plan after sidecar state "
                "has been allocated"
            )
        self._planned_num_blocks = num_blocks
        self._planned_num_raw_slots = num_raw_slots

    def clear_binding(self) -> None:
        """Drop profiling-only state after synchronization and graph teardown."""
        self._binding = None
        self._state = None
        self._planned_num_blocks = None
        self._planned_num_raw_slots = None

    def state(self, kv_cache: torch.Tensor) -> ByteV2RawFallbackState:
        """Return state for ``kv_cache``, initializing on a new binding."""
        binding = self._binding_for(kv_cache)
        if self._state is not None and binding == self._binding:
            return self._state
        if self._binding is not None:
            raise RuntimeError(
                "ByteV2 hybrid raw fallback KV-cache binding changed after "
                "first use; cache pointer, device, and shape must remain fixed"
            )
        self._binding = binding
        self._state = self._allocate_state(kv_cache)
        return self._state

    def clear_forced_raw_diagnostic(self) -> None:
        """Clear test-only lifecycle evidence after warmup or capture."""
        state = self._state
        if state is None or state.forced_raw_diagnostic is None:
            raise RuntimeError("ByteV2 forced raw diagnostic state is unavailable")
        state.forced_raw_diagnostic.zero_()

    def arm_forced_raw_promotion(self) -> None:
        """Arm the next valid fused Q1 update after graph capture."""
        state = self._state
        if state is None or state.forced_raw_diagnostic is None:
            raise RuntimeError("ByteV2 forced raw diagnostic state is unavailable")
        state.forced_raw_diagnostic[0].fill_(1)

    def reset(self, physical_block_ids: torch.Tensor | None = None) -> None:
        """Reset all state, or release selected physical block IDs.

        The scheduler uses selective reset before a block ID is reused and when
        an uncached block becomes dead. Prefix-cached blocks are deliberately
        retained until they are evicted and subsequently reported for reuse.
        """
        state = self._state
        if state is None:
            return
        if physical_block_ids is not None:
            byte_v2_reset_raw_fallback_pages(
                state.page_to_raw_slot,
                state.free_slots,
                state.free_count,
                state.fatal,
                physical_block_ids,
            )
            return
        state.page_to_raw_slot.fill_(-1)
        state.free_slots.copy_(
            torch.arange(
                state.free_slots.shape[0],
                dtype=torch.int32,
                device=state.free_slots.device,
            )
        )
        state.free_count.fill_(state.free_slots.shape[0])
        state.fatal.zero_()
        if state.forced_raw_diagnostic is not None:
            state.forced_raw_diagnostic.zero_()


@dataclass(frozen=True)
class ByteV2RawStagingWorkspaceSpec:
    """Shape and placement of one runner-owned raw staging workspace."""

    num_blocks: int
    num_staging_slots: int
    slot_size_bytes: int
    device: torch.device

    def __post_init__(self) -> None:
        if self.num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if self.num_staging_slots <= 0:
            raise ValueError("num_staging_slots must be positive")
        if self.slot_size_bytes <= 0:
            raise ValueError("slot_size_bytes must be positive")

    @property
    def nbytes(self) -> int:
        """Return the exact storage size of all tensors in the workspace."""
        return (
            self.num_staging_slots * self.slot_size_bytes
            + 4 * self.num_blocks
            + 8 * self.num_staging_slots
            + 8
        )

    def allocate(self) -> ByteV2RawStagingWorkspace:
        """Allocate and initialize the fixed-address workspace tensors."""
        return ByteV2RawStagingWorkspace(
            spec=self,
            raw_staging=torch.empty(
                (self.num_staging_slots, self.slot_size_bytes),
                dtype=torch.uint8,
                device=self.device,
            ),
            block_to_staging_slot=torch.full(
                (self.num_blocks,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            staging_to_physical_block=torch.full(
                (self.num_staging_slots,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            valid_rows=torch.zeros(
                (self.num_staging_slots,),
                dtype=torch.int32,
                device=self.device,
            ),
            next_staging_slot=torch.zeros(
                (1,),
                dtype=torch.int32,
                device=self.device,
            ),
            overflow=torch.zeros(
                (1,),
                dtype=torch.int32,
                device=self.device,
            ),
        )


@dataclass(frozen=True)
class ByteV2RawStagingWorkspace:
    """Fixed-address tensors shared serially by all ByteV2 layers."""

    spec: ByteV2RawStagingWorkspaceSpec
    raw_staging: torch.Tensor
    block_to_staging_slot: torch.Tensor
    staging_to_physical_block: torch.Tensor
    valid_rows: torch.Tensor
    next_staging_slot: torch.Tensor
    overflow: torch.Tensor


@dataclass(frozen=True)
class ByteV2InitialPrefillStagingLease:
    """Transient exact pages retained between cache update and attention."""

    kv_cache_ptr: int
    num_tokens: int
    active_slot_capacity: int
    raw_staging: torch.Tensor
    block_to_staging_slot: torch.Tensor
    staging_to_physical_block: torch.Tensor
    valid_rows: torch.Tensor


@dataclass(frozen=True)
class ByteV2RawStagingWave:
    """A contiguous cache update whose unique-page upper bound fits staging."""

    start: int
    end: int
    max_unique_pages: int


def _trusted_byte_v2_query_start_locs(
    attn_metadata: object | None,
    num_tokens: int,
) -> list[int] | None:
    if attn_metadata is None:
        return None
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    if not isinstance(num_actual_tokens, int) or isinstance(num_actual_tokens, bool):
        return None
    if not 0 <= num_actual_tokens <= num_tokens:
        return None
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    if not isinstance(query_start_loc_cpu, torch.Tensor):
        return None
    if query_start_loc_cpu.device.type != "cpu" or query_start_loc_cpu.ndim != 1:
        return None
    if query_start_loc_cpu.dtype not in (torch.int32, torch.int64):
        return None
    query_start_locs = [int(value) for value in query_start_loc_cpu.tolist()]
    if not query_start_locs or query_start_locs[0] != 0:
        return None
    if query_start_locs[-1] != num_actual_tokens:
        return None
    if any(end < start for start, end in zip(query_start_locs, query_start_locs[1:])):
        return None
    return query_start_locs


def _trusted_byte_v2_initial_prefill_rows(
    attn_metadata: object | None,
    query_start_locs: list[int],
) -> list[bool]:
    """Identify rows whose first scheduled token is known to be page-aligned."""
    num_requests = len(query_start_locs) - 1
    unknown = [False] * num_requests
    if attn_metadata is None:
        return unknown

    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu_upper_bound", None)
    if (
        not isinstance(seq_lens_cpu, torch.Tensor)
        or seq_lens_cpu.device.type != "cpu"
        or seq_lens_cpu.dtype not in (torch.int32, torch.int64)
        or seq_lens_cpu.ndim != 1
        or seq_lens_cpu.shape[0] < num_requests
    ):
        return unknown

    # The exact sequence length cannot be shorter than its scheduled query,
    # while seq_lens_cpu_upper_bound cannot be shorter than the exact length.
    # Equality at both ends therefore proves context_len == 0 even when async
    # speculative decode makes the CPU value only an upper bound. All rows
    # with a positive or unknown context retain the conservative bound.
    return [
        int(seq_lens_cpu[index]) == request_end - request_start
        for index, (request_start, request_end) in enumerate(
            zip(query_start_locs, query_start_locs[1:])
        )
    ]


def plan_byte_v2_raw_staging_waves(
    num_tokens: int,
    num_staging_slots: int,
    attn_metadata: object | None = None,
    *,
    alloc_block_tokens: int = 16,
) -> list[ByteV2RawStagingWave]:
    """Plan page-bounded, stream-ordered staging waves.

    A request segment with ``n`` consecutive logical tokens can touch at most
    ``ceil((block_size - 1 + n) / block_size)`` pages when its first row is
    unknown. An initial-prefill row whose zero context is proven by trusted CPU
    metadata instead uses its exact page alignment. Bounds from adjacent
    requests are summed, which may over-count shared physical pages but can
    never under-count them.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if num_staging_slots <= 0:
        raise ValueError("num_staging_slots must be positive")
    if alloc_block_tokens <= 0:
        raise ValueError("alloc_block_tokens must be positive")

    query_start_locs = _trusted_byte_v2_query_start_locs(
        attn_metadata,
        num_tokens,
    )
    if query_start_locs is None:
        conservative_waves = []
        for start in range(0, num_tokens, num_staging_slots):
            end = min(start + num_staging_slots, num_tokens)
            conservative_waves.append(ByteV2RawStagingWave(start, end, end - start))
        return conservative_waves

    initial_prefill_rows = _trusted_byte_v2_initial_prefill_rows(
        attn_metadata,
        query_start_locs,
    )
    pieces: list[ByteV2RawStagingWave] = []
    conservative_token_limit = alloc_block_tokens * num_staging_slots
    conservative_token_limit -= alloc_block_tokens - 1
    aligned_token_limit = alloc_block_tokens * num_staging_slots
    for request_index, (request_start, request_end) in enumerate(
        zip(query_start_locs, query_start_locs[1:])
    ):
        initial_prefill = initial_prefill_rows[request_index]
        token_limit = (
            aligned_token_limit if initial_prefill else conservative_token_limit
        )
        start = request_start
        while start < request_end:
            end = min(start + token_limit, request_end)
            num_request_tokens = end - start
            if initial_prefill:
                first_row = (start - request_start) % alloc_block_tokens
                max_unique_pages = (
                    first_row + num_request_tokens + alloc_block_tokens - 1
                ) // alloc_block_tokens
            else:
                max_unique_pages = (
                    num_request_tokens + 2 * alloc_block_tokens - 2
                ) // alloc_block_tokens
            pieces.append(ByteV2RawStagingWave(start, end, max_unique_pages))
            start = end

    waves: list[ByteV2RawStagingWave] = []
    for piece in pieces:
        if waves and waves[-1].max_unique_pages + piece.max_unique_pages <= (
            num_staging_slots
        ):
            previous = waves[-1]
            waves[-1] = ByteV2RawStagingWave(
                previous.start,
                piece.end,
                previous.max_unique_pages + piece.max_unique_pages,
            )
        else:
            waves.append(piece)
    return waves


class ByteV2RawStagingManager:
    """Reusable raw staging buffers for ByteV2 cache updates."""

    def __init__(
        self,
        *,
        tile_policy: ByteV2TilePolicy,
        num_kv_heads: int,
        max_tokens_per_update: int = _BYTE_V2_MAX_RAW_STAGING_TOKENS,
        raw_fallback_store: ByteV2RawFallbackStore | None = None,
        hybrid_raw_mutable_tail_q1: bool = False,
        hybrid_raw_tail_fused_finalize: bool = True,
    ) -> None:
        self.tile_policy = tile_policy
        self.raw_layout = ByteV2RawStagingLayout(
            tile_policy=tile_policy,
            num_kv_heads=num_kv_heads,
        )
        self.max_tokens_per_update = max_tokens_per_update
        self.raw_fallback_store = raw_fallback_store
        self.hybrid_raw_mutable_tail_q1 = hybrid_raw_mutable_tail_q1
        self.hybrid_raw_tail_fused_finalize = hybrid_raw_tail_fused_finalize
        self.shared_workspace: ByteV2RawStagingWorkspace | None = None

        self.raw_staging: torch.Tensor | None = None
        self.block_to_staging_slot: torch.Tensor | None = None
        self.staging_to_physical_block: torch.Tensor | None = None
        self.valid_rows: torch.Tensor | None = None
        self.next_staging_slot: torch.Tensor | None = None
        self.overflow: torch.Tensor | None = None
        self._initial_prefill_lease: ByteV2InitialPrefillStagingLease | None = None

    def bind_shared_workspace(
        self,
        workspace: ByteV2RawStagingWorkspace,
    ) -> None:
        """Bind a runner-owned workspace without changing tensor addresses."""
        if self.raw_fallback_store is None:
            raise RuntimeError(
                "ByteV2 shared raw staging is only valid with hybrid fallback"
            )
        if workspace.spec.slot_size_bytes != self.raw_layout.slot_size_bytes:
            raise RuntimeError(
                "ByteV2 raw staging slot-size mismatch: workspace has "
                f"{workspace.spec.slot_size_bytes} bytes, layer requires "
                f"{self.raw_layout.slot_size_bytes} bytes"
            )
        self.shared_workspace = workspace
        self.raw_staging = workspace.raw_staging
        self.block_to_staging_slot = workspace.block_to_staging_slot
        self.staging_to_physical_block = workspace.staging_to_physical_block
        self.valid_rows = workspace.valid_rows
        self.next_staging_slot = workspace.next_staging_slot
        self.overflow = workspace.overflow

    def clear_shared_workspace(self) -> None:
        """Drop all references to a profiling-only shared workspace."""
        if self._initial_prefill_lease is not None:
            raise RuntimeError(
                "Cannot clear ByteV2 raw staging with an active initial-prefill lease"
            )
        self.shared_workspace = None
        self.raw_staging = None
        self.block_to_staging_slot = None
        self.staging_to_physical_block = None
        self.valid_rows = None
        self.next_staging_slot = None
        self.overflow = None

    def stage_initial_prefill(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Stage one complete initial prefill as exact raw BF16 pages.

        The returned physical-page map can be passed directly to the hybrid
        FA2 loader. If the shared workspace cannot hold every active page at
        once, return ``None`` so the caller can use the compact-cache reader.
        """
        lease = self._initial_prefill_lease
        if lease is not None:
            self._initial_prefill_lease = None
            if (
                lease.kv_cache_ptr != kv_cache.data_ptr()
                or lease.num_tokens != slot_mapping.shape[0]
            ):
                self._release_allocator_state(
                    active_slot_capacity=lease.active_slot_capacity
                )
                raise RuntimeError(
                    "ByteV2 retained initial-prefill staging does not match "
                    "the following attention call"
                )
            return (
                lease.raw_staging,
                lease.block_to_staging_slot,
                lease.staging_to_physical_block,
                lease.valid_rows,
            )

        workspace = self.shared_workspace
        if workspace is None or slot_mapping.numel() == 0:
            return None
        waves = plan_byte_v2_raw_staging_waves(
            slot_mapping.shape[0],
            workspace.spec.num_staging_slots,
            attn_metadata,
            alloc_block_tokens=self.tile_policy.alloc_block_tokens,
        )
        if (
            len(waves) != 1
            or waves[0].start != 0
            or waves[0].end != slot_mapping.shape[0]
        ):
            return None

        active_capacity = waves[0].max_unique_pages
        raw_staging = workspace.raw_staging[:active_capacity]
        staging_to_physical_block = workspace.staging_to_physical_block[
            :active_capacity
        ]
        valid_rows = workspace.valid_rows[:active_capacity]
        byte_v2_prepare_raw_staging(
            slot_mapping,
            workspace.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            workspace.next_staging_slot,
            workspace.overflow,
            alloc_block_tokens=self.tile_policy.alloc_block_tokens,
        )
        byte_v2_append_raw_staging(
            key,
            value,
            raw_staging,
            slot_mapping,
            workspace.block_to_staging_slot,
            codec_token_block=self.tile_policy.codec_token_block,
            codec_dim_block=self.tile_policy.codec_dim_block,
            alloc_block_tokens=self.tile_policy.alloc_block_tokens,
        )
        return (
            raw_staging,
            workspace.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
        )

    def stage_cached_prefill(
        self,
        *,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Hydrate one B1 cached sequence into exact raw BF16 pages.

        This is an experimental full-prefix path. It decodes each logical
        page once, then exposes a dense local block table to the original raw
        paged FA2 implementation. If the fixed workspace cannot hold the
        complete sequence, return ``None`` without mutating allocator state.
        """
        if self._initial_prefill_lease is not None:
            raise RuntimeError(
                "ByteV2 cached-prefill hydrate cannot reuse an active "
                "initial-prefill staging lease"
            )
        workspace = self.shared_workspace
        if (
            workspace is None
            or self.raw_fallback_store is None
            or seq_len <= 0
            or block_table.dtype != torch.int32
            or block_table.device != kv_cache.device
            or block_table.ndim != 2
            or block_table.shape[0] != 1
            or block_table.stride(1) != 1
        ):
            return None

        block_size = self.tile_policy.alloc_block_tokens
        num_pages = (seq_len + block_size - 1) // block_size
        if (
            num_pages <= 0
            or num_pages > workspace.spec.num_staging_slots
            or num_pages > block_table.shape[1]
            or num_pages > workspace.block_to_staging_slot.shape[0]
        ):
            return None

        raw_staging = workspace.raw_staging[:num_pages]
        valid_rows = workspace.valid_rows[:num_pages]
        physical_pages = block_table[0, :num_pages]
        hybrid_state = self.raw_fallback_store.state(kv_cache)
        valid_rows.fill_(block_size)
        final_rows = seq_len % block_size
        if final_rows:
            valid_rows[-1].fill_(final_rows)

        # This descriptor is not used by hydrate; it is a fixed-address dense
        # block table for the following raw FA2 call.
        local_page_ids = workspace.staging_to_physical_block[:num_pages]
        try:
            torch.arange(num_pages, out=local_page_ids)
            byte_v2_hydrate_raw_staging_from_hybrid_cache(
                raw_staging,
                kv_cache,
                hybrid_state.raw_pages,
                hybrid_state.page_to_raw_slot,
                physical_pages,
                valid_rows,
                codec_token_block=self.tile_policy.codec_token_block,
                codec_dim_block=self.tile_policy.codec_dim_block,
                alloc_block_tokens=block_size,
            )
        except Exception:
            self.release_cached_prefill(
                local_page_ids.view(1, num_pages),
                valid_rows,
            )
            raise
        return (
            raw_staging,
            workspace.block_to_staging_slot,
            local_page_ids.view(1, num_pages),
            valid_rows,
        )

    def release_initial_prefill(
        self,
        staging_to_physical_block: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> None:
        """Release page mappings created by :meth:`stage_initial_prefill`."""
        workspace = self.shared_workspace
        if workspace is None:
            raise RuntimeError("ByteV2 initial prefill staging is not bound")
        byte_v2_release_raw_staging(
            workspace.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            workspace.next_staging_slot,
            workspace.overflow,
        )

    def release_cached_prefill(
        self,
        local_page_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> None:
        """Release cached-prefill descriptors while ``block_to`` is quiescent."""
        self.release_initial_prefill(local_page_ids.view(-1), valid_rows)

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
        num_staging_slots: int,
    ) -> bool:
        num_blocks = kv_cache.shape[0]
        if num_staging_slots <= 0:
            return False

        if self.raw_fallback_store is not None:
            workspace = self.shared_workspace
            if workspace is None:
                raise RuntimeError(
                    "ByteV2 hybrid raw fallback requires a runner-owned raw "
                    "staging workspace"
                )
            spec = workspace.spec
            if spec.device != kv_cache.device or spec.num_blocks != num_blocks:
                raise RuntimeError(
                    "ByteV2 raw staging workspace does not match the bound KV "
                    f"cache: workspace=(device={spec.device}, blocks="
                    f"{spec.num_blocks}), cache=(device={kv_cache.device}, "
                    f"blocks={num_blocks})"
                )
            if num_staging_slots > spec.num_staging_slots:
                raise RuntimeError(
                    "ByteV2 raw staging wave requires "
                    f"{num_staging_slots} pages, but the shared workspace has "
                    f"only {spec.num_staging_slots} slots"
                )
            return True

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

    def _release_allocator_state(
        self,
        kv_cache: torch.Tensor | None = None,
        page_unsafe_flags: torch.Tensor | None = None,
        active_slot_capacity: int | None = None,
    ) -> bool:
        assert self.block_to_staging_slot is not None
        assert self.staging_to_physical_block is not None
        assert self.valid_rows is not None
        assert self.next_staging_slot is not None
        assert self.overflow is not None
        staging_to_physical_block = self.staging_to_physical_block
        valid_rows = self.valid_rows
        if active_slot_capacity is not None:
            staging_to_physical_block = staging_to_physical_block[:active_slot_capacity]
            valid_rows = valid_rows[:active_slot_capacity]
        if (
            kv_cache is not None
            and page_unsafe_flags is not None
            and _fused_staging_release_flags_enabled()
        ):
            byte_v2_release_raw_staging_and_update_flags(
                self.block_to_staging_slot,
                staging_to_physical_block,
                valid_rows,
                self.next_staging_slot,
                self.overflow,
                page_unsafe_flags,
                kv_cache,
                tile_policy=(
                    self.tile_policy.codec_token_block,
                    self.tile_policy.codec_dim_block,
                    self.tile_policy.alloc_block_tokens,
                    self.tile_policy.compute_block_n,
                    self.tile_policy.head_dim,
                    self.tile_policy.head_dim_v,
                ),
            )
            return True
        byte_v2_release_raw_staging(
            self.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            self.next_staging_slot,
            self.overflow,
        )
        return False

    def update(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        page_unsafe_flags: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        retain_initial_prefill: bool = False,
    ) -> tuple[bool, bool]:
        if self._initial_prefill_lease is not None:
            raise RuntimeError(
                "ByteV2 raw staging was reused before the retained "
                "initial-prefill lease was consumed"
            )
        num_tokens = slot_mapping.shape[0]
        if num_tokens == 0:
            return True, False
        if self.raw_fallback_store is not None:
            workspace = self.shared_workspace
            if workspace is None:
                raise RuntimeError(
                    "ByteV2 hybrid raw fallback requires a runner-owned raw "
                    "staging workspace"
                )
            hybrid_state = self.raw_fallback_store.state(kv_cache)
            waves = plan_byte_v2_raw_staging_waves(
                num_tokens,
                workspace.spec.num_staging_slots,
                attn_metadata,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            retain_single_wave = (
                retain_initial_prefill
                and page_unsafe_flags is None
                and len(waves) == 1
                and waves[0].start == 0
                and waves[0].end == num_tokens
            )
            flags_updated = page_unsafe_flags is not None and bool(waves)
            for wave in waves:
                handled, wave_flags_updated = self._update_one_wave(
                    key=key[wave.start : wave.end],
                    value=value[wave.start : wave.end],
                    kv_cache=kv_cache,
                    slot_mapping=slot_mapping[wave.start : wave.end],
                    page_unsafe_flags=page_unsafe_flags,
                    hybrid_state=hybrid_state,
                    active_slot_capacity=wave.max_unique_pages,
                    retain_initial_prefill=retain_single_wave,
                )
                if not handled:
                    raise RuntimeError(
                        "ByteV2 hybrid raw fallback update unexpectedly fell "
                        "through to the direct writer"
                    )
                flags_updated = flags_updated and wave_flags_updated
            return True, flags_updated

        return self._update_one_wave(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            page_unsafe_flags=page_unsafe_flags,
            hybrid_state=None,
            active_slot_capacity=min(kv_cache.shape[0], num_tokens),
        )

    def _update_one_wave(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        page_unsafe_flags: torch.Tensor | None,
        hybrid_state: ByteV2RawFallbackState | None,
        active_slot_capacity: int,
        retain_initial_prefill: bool = False,
    ) -> tuple[bool, bool]:
        if (
            hybrid_state is None
            and slot_mapping.shape[0] == 1
            and slot_mapping.is_cuda
            and _native_single_token_update_enabled()
        ):
            try:
                byte_v2_update_cache_single_token(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                    page_unsafe_flags=page_unsafe_flags,
                    codec_token_block=self.tile_policy.codec_token_block,
                    codec_dim_block=self.tile_policy.codec_dim_block,
                    alloc_block_tokens=self.tile_policy.alloc_block_tokens,
                )
                return True, page_unsafe_flags is not None
            except NotImplementedError:
                pass
        if hybrid_state is None and not self._should_stage(slot_mapping):
            return False, False
        if not self._ensure_capacity(kv_cache, active_slot_capacity):
            return False, False

        assert self.raw_staging is not None
        assert self.block_to_staging_slot is not None
        assert self.staging_to_physical_block is not None
        assert self.valid_rows is not None
        assert self.next_staging_slot is not None
        assert self.overflow is not None

        raw_staging = self.raw_staging[:active_slot_capacity]
        staging_to_physical_block = self.staging_to_physical_block[
            :active_slot_capacity
        ]
        valid_rows = self.valid_rows[:active_slot_capacity]

        if (
            hybrid_state is not None
            and slot_mapping.shape[0] == 1
            and active_slot_capacity == 1
            and slot_mapping.is_cuda
        ):
            if getattr(self, "hybrid_raw_mutable_tail_q1", False):
                try:
                    byte_v2_update_hybrid_cache_raw_tail_q1(
                        key,
                        value,
                        raw_staging,
                        kv_cache,
                        hybrid_state.raw_pages,
                        slot_mapping,
                        self.block_to_staging_slot,
                        staging_to_physical_block,
                        valid_rows,
                        self.next_staging_slot,
                        self.overflow,
                        hybrid_state.page_to_raw_slot,
                        hybrid_state.free_slots,
                        hybrid_state.free_count,
                        hybrid_state.fatal,
                        tile_policy=(
                            self.tile_policy.codec_token_block,
                            self.tile_policy.codec_dim_block,
                            self.tile_policy.alloc_block_tokens,
                            self.tile_policy.compute_block_n,
                            self.tile_policy.head_dim,
                            self.tile_policy.head_dim_v,
                        ),
                        page_unsafe_flags=page_unsafe_flags,
                        fuse_commit_finalize=self.hybrid_raw_tail_fused_finalize,
                    )
                except NotImplementedError:
                    pass
                else:
                    # Raw-tail sealing owns the row-15 demotion decision; the
                    # forced-promotion diagnostic must not remap that page.
                    return True, page_unsafe_flags is not None
            try:
                byte_v2_update_hybrid_cache_raw_staging_q1(
                    key,
                    value,
                    raw_staging,
                    kv_cache,
                    hybrid_state.raw_pages,
                    slot_mapping,
                    self.block_to_staging_slot,
                    staging_to_physical_block,
                    valid_rows,
                    self.next_staging_slot,
                    self.overflow,
                    hybrid_state.page_to_raw_slot,
                    hybrid_state.free_slots,
                    hybrid_state.free_count,
                    hybrid_state.fatal,
                    tile_policy=(
                        self.tile_policy.codec_token_block,
                        self.tile_policy.codec_dim_block,
                        self.tile_policy.alloc_block_tokens,
                        self.tile_policy.compute_block_n,
                        self.tile_policy.head_dim,
                        self.tile_policy.head_dim_v,
                    ),
                    page_unsafe_flags=page_unsafe_flags,
                )
            except NotImplementedError:
                pass
            else:
                forced_raw_diagnostic = getattr(
                    hybrid_state,
                    "forced_raw_diagnostic",
                    None,
                )
                if forced_raw_diagnostic is not None:
                    # This diagnostic launch must remain immediately after
                    # fused Q1 on the same stream: persist/release leaves the
                    # current valid raw-page prefix in raw_staging[0].
                    byte_v2_test_force_promote_raw_staging_q1(
                        raw_staging,
                        hybrid_state.raw_pages,
                        slot_mapping,
                        hybrid_state.page_to_raw_slot,
                        hybrid_state.free_slots,
                        hybrid_state.free_count,
                        hybrid_state.fatal,
                        forced_raw_diagnostic,
                    )
                return True, page_unsafe_flags is not None

        if (
            hybrid_state is not None
            and slot_mapping.shape[0] > 1
            and slot_mapping.is_cuda
        ):
            if retain_initial_prefill:
                try:
                    byte_v2_update_hybrid_cache_raw_staging_multi_token_retained(
                        key,
                        value,
                        raw_staging,
                        kv_cache,
                        hybrid_state.raw_pages,
                        slot_mapping,
                        self.block_to_staging_slot,
                        staging_to_physical_block,
                        valid_rows,
                        self.next_staging_slot,
                        self.overflow,
                        hybrid_state.page_to_raw_slot,
                        hybrid_state.free_slots,
                        hybrid_state.free_count,
                        hybrid_state.fatal,
                        tile_policy=(
                            self.tile_policy.codec_token_block,
                            self.tile_policy.codec_dim_block,
                            self.tile_policy.alloc_block_tokens,
                            self.tile_policy.compute_block_n,
                            self.tile_policy.head_dim,
                            self.tile_policy.head_dim_v,
                        ),
                        page_unsafe_flags=None,
                    )
                except NotImplementedError:
                    pass
                else:
                    self._initial_prefill_lease = ByteV2InitialPrefillStagingLease(
                        kv_cache_ptr=kv_cache.data_ptr(),
                        num_tokens=slot_mapping.shape[0],
                        active_slot_capacity=active_slot_capacity,
                        raw_staging=raw_staging,
                        block_to_staging_slot=self.block_to_staging_slot,
                        staging_to_physical_block=(staging_to_physical_block),
                        valid_rows=valid_rows,
                    )
                    return True, False
            try:
                byte_v2_update_hybrid_cache_raw_staging_multi_token(
                    key,
                    value,
                    raw_staging,
                    kv_cache,
                    hybrid_state.raw_pages,
                    slot_mapping,
                    self.block_to_staging_slot,
                    staging_to_physical_block,
                    valid_rows,
                    self.next_staging_slot,
                    self.overflow,
                    hybrid_state.page_to_raw_slot,
                    hybrid_state.free_slots,
                    hybrid_state.free_count,
                    hybrid_state.fatal,
                    tile_policy=(
                        self.tile_policy.codec_token_block,
                        self.tile_policy.codec_dim_block,
                        self.tile_policy.alloc_block_tokens,
                        self.tile_policy.compute_block_n,
                        self.tile_policy.head_dim,
                        self.tile_policy.head_dim_v,
                    ),
                    page_unsafe_flags=page_unsafe_flags,
                    demote_safe_raw_pages=self.hybrid_raw_mutable_tail_q1,
                )
                return True, page_unsafe_flags is not None
            except NotImplementedError:
                pass

        if (
            hybrid_state is None
            and page_unsafe_flags is not None
            and _fused_staging_release_flags_enabled()
            and _native_raw_staging_update_enabled()
            and not _debug_warmup_enabled()
        ):
            try:
                byte_v2_update_cache_raw_staging(
                    key,
                    value,
                    raw_staging,
                    kv_cache,
                    slot_mapping,
                    self.block_to_staging_slot,
                    staging_to_physical_block,
                    valid_rows,
                    self.next_staging_slot,
                    self.overflow,
                    page_unsafe_flags,
                    tile_policy=(
                        self.tile_policy.codec_token_block,
                        self.tile_policy.codec_dim_block,
                        self.tile_policy.alloc_block_tokens,
                        self.tile_policy.compute_block_n,
                        self.tile_policy.head_dim,
                        self.tile_policy.head_dim_v,
                    ),
                    fuse_metadata_clear=_fused_commit_metadata_clear_enabled(),
                    warp_parallel_histogram=(_warp_parallel_commit_histogram_enabled()),
                    fuse_single_token_staging=(_fused_single_token_staging_enabled()),
                    fuse_single_token_commit_release=(
                        _fused_single_token_commit_release_enabled()
                    ),
                    fuse_single_token_stage_metadata_clear=(
                        _fused_single_token_stage_metadata_clear_enabled()
                    ),
                )
                return True, True
            except NotImplementedError:
                pass

        try:
            _debug_warmup(
                "raw staging prepare start slot_mapping=%s raw_staging=%s capacity=%s",
                tuple(slot_mapping.shape),
                tuple(raw_staging.shape),
                tuple(self.raw_staging.shape),
            )
            byte_v2_prepare_raw_staging(
                slot_mapping,
                self.block_to_staging_slot,
                staging_to_physical_block,
                valid_rows,
                self.next_staging_slot,
                self.overflow,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            if hybrid_state is not None:
                torch._assert_async(
                    self.overflow == 0,
                    "ByteV2 raw staging page bound was exceeded",
                )
            _debug_sync("raw staging prepare")
            _debug_warmup("raw staging prepare done")
            _debug_warmup("raw staging hydrate start")
            if hybrid_state is None:
                byte_v2_hydrate_raw_staging_from_cache(
                    raw_staging,
                    kv_cache,
                    staging_to_physical_block,
                    valid_rows,
                    codec_token_block=self.tile_policy.codec_token_block,
                    codec_dim_block=self.tile_policy.codec_dim_block,
                    alloc_block_tokens=self.tile_policy.alloc_block_tokens,
                )
            else:
                byte_v2_hydrate_raw_staging_from_hybrid_cache(
                    raw_staging,
                    kv_cache,
                    hybrid_state.raw_pages,
                    hybrid_state.page_to_raw_slot,
                    staging_to_physical_block,
                    valid_rows,
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
                raw_staging,
                slot_mapping,
                self.block_to_staging_slot,
                codec_token_block=self.tile_policy.codec_token_block,
                codec_dim_block=self.tile_policy.codec_dim_block,
                alloc_block_tokens=self.tile_policy.alloc_block_tokens,
            )
            _debug_sync("raw staging append")
            _debug_warmup("raw staging append done")
            _debug_warmup("raw staging commit start")
            if hybrid_state is None:
                byte_v2_commit_raw_staging_to_cache(
                    raw_staging,
                    kv_cache,
                    staging_to_physical_block,
                    valid_rows,
                    codec_token_block=self.tile_policy.codec_token_block,
                    codec_dim_block=self.tile_policy.codec_dim_block,
                    alloc_block_tokens=self.tile_policy.alloc_block_tokens,
                )
            else:
                byte_v2_commit_raw_staging_to_hybrid_cache(
                    raw_staging,
                    kv_cache,
                    hybrid_state.raw_pages,
                    hybrid_state.page_to_raw_slot,
                    hybrid_state.free_slots,
                    hybrid_state.free_count,
                    hybrid_state.fatal,
                    staging_to_physical_block,
                    valid_rows,
                    codec_token_block=self.tile_policy.codec_token_block,
                    codec_dim_block=self.tile_policy.codec_dim_block,
                    alloc_block_tokens=self.tile_policy.alloc_block_tokens,
                )
            _debug_sync("raw staging commit")
            _debug_warmup("raw staging commit done")
            _debug_warmup("raw staging release start")
            flags_updated = self._release_allocator_state(
                kv_cache,
                page_unsafe_flags,
                active_slot_capacity,
            )
            _debug_sync("raw staging release")
            _debug_warmup("raw staging release done")
        except NotImplementedError as error:
            self._initialize_allocator_state()
            if hybrid_state is not None:
                raise RuntimeError(
                    "ByteV2 hybrid raw fallback op became unavailable after "
                    "runtime validation"
                ) from error
            return False, False
        return True, flags_updated


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
        self.fa2_hybrid_raw_fallback = byte_v2_hybrid_raw_fallback_enabled()
        self.test_forced_raw_promotion = byte_v2_test_forced_raw_promotion_enabled()
        raw_tail_requested = _hybrid_raw_mutable_tail_q1_enabled(
            hybrid_raw_fallback=(
                self.fa2_hybrid_raw_fallback and not self.test_forced_raw_promotion
            )
        )
        if raw_tail_requested and not self.fa2_hybrid_raw_fallback:
            raise RuntimeError(
                "BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1=1 requires "
                "BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1"
            )
        self.hybrid_raw_mutable_tail_q1 = resolve_byte_v2_hybrid_raw_mutable_tail_q1(
            native_available=byte_v2_hybrid_raw_tail_q1_is_available(),
            hybrid_raw_fallback=(
                self.fa2_hybrid_raw_fallback and not self.test_forced_raw_promotion
            ),
        )
        if self.hybrid_raw_mutable_tail_q1:
            logger.warning_once(
                "[ByteV2] experimental raw-tail Q1 is enabled; safe full pages "
                "written by Q>1 dynamic batches are demoted back to compact "
                "storage, cache writes must remain ordered on one CUDA stream, "
                "and resumable/streaming sessions are rejected"
            )
        if raw_tail_requested and not self.hybrid_raw_mutable_tail_q1:
            logger.warning_once(
                "[ByteV2] persistent raw-tail Q1 op is unavailable; using "
                "the existing hybrid Q1 cache update"
            )
        direct_paged_prefill_enabled = _direct_paged_prefill_enabled()
        if (
            os.environ.get("BYTE_V2_FA2_DIRECT_PREFILL") == "1"
            and not self.fa2_hybrid_raw_fallback
        ):
            raise RuntimeError(
                "BYTE_V2_FA2_DIRECT_PREFILL=1 requires "
                "BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1"
            )
        self.direct_paged_prefill = (
            self.fa2_hybrid_raw_fallback and direct_paged_prefill_enabled
        )
        cached_prefill_hydrate_mode = _cached_prefill_hydrate_to_raw_mode()
        if (
            cached_prefill_hydrate_mode == "enabled"
            and not self.fa2_hybrid_raw_fallback
        ):
            raise RuntimeError(
                "BYTE_V2_FA2_CACHED_PREFILL_HYDRATE_TO_RAW=1 requires "
                "BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1"
            )
        self.cached_prefill_hydrate_to_raw_mode = (
            cached_prefill_hydrate_mode if self.fa2_hybrid_raw_fallback else "disabled"
        )
        self.cached_prefill_hydrate_to_raw = (
            self.cached_prefill_hydrate_to_raw_mode != "disabled"
        )
        if self.cached_prefill_hydrate_to_raw:
            logger.warning_once(
                "[ByteV2] experimental B1 cached-prefill hydrate-to-raw mode "
                "is %s; auto mode requires Q >= %d, and sequences that exceed "
                "raw staging capacity retain the hybrid reader",
                self.cached_prefill_hydrate_to_raw_mode,
                _BYTE_V2_CACHED_PREFILL_AUTO_MIN_QUERY_LEN,
            )
        if self.test_forced_raw_promotion and not self.fa2_hybrid_raw_fallback:
            raise RuntimeError(
                "BYTE_V2_TEST_FORCE_RAW_PROMOTION=1 requires "
                "BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1"
            )
        if self.test_forced_raw_promotion and self.hybrid_raw_mutable_tail_q1:
            raise RuntimeError(
                "BYTE_V2_TEST_FORCE_RAW_PROMOTION=1 is incompatible with "
                "BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1=1"
            )
        if self.fa2_hybrid_raw_fallback:
            unsupported = []
            if (self.num_heads, self.num_kv_heads, self.head_size) != (32, 8, 128):
                unsupported.append("local Hq=32, Hkv=8, and D=128 are required")
            if self.tile_policy != DEFAULT_BYTE_V2_TILE_POLICY:
                unsupported.append("the default ByteV2 tile policy is required")
            if self.kv_sharing_target_layer_name is not None:
                unsupported.append("cross-layer KV-cache sharing")
            if self.alibi_slopes is not None:
                unsupported.append("ALiBi")
            if self.sliding_window is not None:
                unsupported.append("sliding-window/local attention")
            if self.logits_soft_cap is not None and self.logits_soft_cap > 0:
                unsupported.append("logit softcap")
            if self.attn_type != AttentionType.DECODER:
                unsupported.append("non-decoder attention")
            if unsupported:
                raise RuntimeError(
                    "ByteV2 hybrid raw fallback does not support this "
                    f"configuration: {', '.join(unsupported)}"
                )
        self.raw_fallback_store: ByteV2RawFallbackStore | None = None
        if self.fa2_hybrid_raw_fallback:
            self.raw_fallback_store = ByteV2RawFallbackStore(
                raw_layout=ByteV2RawStagingLayout(
                    tile_policy=self.tile_policy,
                    num_kv_heads=self.num_kv_heads,
                ),
                forced_raw_diagnostic=self.test_forced_raw_promotion,
            )
        self.raw_staging_manager = ByteV2RawStagingManager(
            tile_policy=self.tile_policy,
            num_kv_heads=self.num_kv_heads,
            raw_fallback_store=self.raw_fallback_store,
            hybrid_raw_mutable_tail_q1=self.hybrid_raw_mutable_tail_q1,
            hybrid_raw_tail_fused_finalize=(_hybrid_raw_tail_fused_finalize_enabled()),
        )
        self.prefill_backend = _prefill_backend()
        self.decode_kernel_mode = _decode_kernel_mode()
        self.decode_fa2_available = False
        if self.fa2_hybrid_raw_fallback:
            hybrid_reader_available = byte_v2_fa2_hybrid_decode_is_available()
            hybrid_writer_available = byte_v2_hybrid_cache_update_is_available()
            if not hybrid_reader_available or not hybrid_writer_available:
                missing = []
                if not hybrid_reader_available:
                    missing.append("FA2 hybrid reader")
                if not hybrid_writer_available:
                    missing.append("hybrid cache writer/reset ops")
                raise RuntimeError(
                    "BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1 requested the "
                    "experimental compact/raw checkpoint, but the following "
                    f"ops are unavailable: {', '.join(missing)}"
                )
            if (
                self.test_forced_raw_promotion
                and not byte_v2_test_forced_raw_promotion_is_available()
            ):
                raise RuntimeError(
                    "BYTE_V2_TEST_FORCE_RAW_PROMOTION=1 requested the "
                    "test-only forced raw lifecycle hook, but its CUDA op is "
                    "unavailable"
                )
            self.decode_fa2_available = True
            logger.info_once("[ByteV2] experimental FA2 hybrid raw fallback is enabled")
        elif self.decode_kernel_mode != "legacy":
            self.decode_fa2_available = byte_v2_fa2_decode_is_available()
            if self.decode_kernel_mode == "fa2" and not self.decode_fa2_available:
                raise RuntimeError(
                    "BYTE_V2_DECODE_KERNEL=fa2 was requested, but the ByteV2 "
                    "FA2 extension op is unavailable"
                )
            if self.decode_kernel_mode == "auto" and not self.decode_fa2_available:
                logger.warning_once(
                    "[ByteV2] ByteV2 FA2 decode is unavailable; using the "
                    "legacy decode kernel"
                )
        self.decode_raw_fallback = _decode_raw_fallback_enabled()
        self.decode_assume_no_outlier = _decode_assume_no_outlier_enabled()
        self.decode_gqa_packed = _decode_gqa_packed_enabled()
        self.decode_gqa_fa2_like = _decode_gqa_fa2_like_enabled()
        self.decode_gqa_fa2_direct = _decode_gqa_fa2_direct_enabled()
        self.decode_unnormalized_partition_output = (
            _decode_unnormalized_partition_output_enabled()
        )
        self.decode_validate_no_outlier = _decode_validate_no_outlier_enabled()
        self.decode_page_unsafe_flags = _decode_page_unsafe_flags_enabled()
        self.speculative_verify_q4 = _speculative_verify_q4_enabled()
        self.speculative_verify_gqa = _speculative_verify_gqa_enabled()
        self.speculative_verify_ragged_q4 = _speculative_verify_ragged_q4_enabled()
        self.cached_prefix_q16 = _cached_prefix_q16_enabled()
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
        self._decode_empty_max_logits: torch.Tensor | None = None
        self._decode_tmp_out: torch.Tensor | None = None
        self._speculative_ragged_exp_sums: torch.Tensor | None = None
        self._speculative_ragged_tmp_out: torch.Tensor | None = None
        self._decode_cache_stats: torch.Tensor | None = None
        self._decode_page_unsafe_flags: torch.Tensor | None = None
        self._decode_page_unsafe_flags_cache_ptr: int | None = None

    def raw_staging_workspace_spec(
        self,
        kv_cache: torch.Tensor,
        num_staging_slots: int,
    ) -> ByteV2RawStagingWorkspaceSpec | None:
        """Return this layer's requirement for the shared staging bundle."""
        if not self.fa2_hybrid_raw_fallback:
            return None
        return ByteV2RawStagingWorkspaceSpec(
            num_blocks=int(kv_cache.shape[0]),
            num_staging_slots=num_staging_slots,
            slot_size_bytes=self.raw_staging_manager.raw_layout.slot_size_bytes,
            device=kv_cache.device,
        )

    def bind_raw_staging_workspace(
        self,
        workspace: ByteV2RawStagingWorkspace,
    ) -> None:
        """Bind the runner-owned workspace used serially across layers."""
        if not self.fa2_hybrid_raw_fallback:
            raise RuntimeError(
                "Cannot bind a ByteV2 raw staging workspace when hybrid "
                "fallback is disabled"
            )
        self.raw_staging_manager.bind_shared_workspace(workspace)

    def bind_raw_fallback_plan(
        self,
        *,
        num_blocks: int,
        num_raw_slots: int,
    ) -> None:
        """Bind persistent sidecar dimensions resolved by the planner."""
        if self.raw_fallback_store is None:
            raise RuntimeError(
                "Cannot bind a raw fallback plan when hybrid fallback is disabled"
            )
        self.raw_fallback_store.bind_plan(
            num_blocks=num_blocks,
            num_raw_slots=num_raw_slots,
        )

    def initialize_raw_fallback_state(self, kv_cache: torch.Tensor) -> None:
        """Allocate planned persistent state before warmup or graph capture."""
        if self.raw_fallback_store is None:
            raise RuntimeError(
                "Cannot initialize raw fallback state when hybrid fallback is disabled"
            )
        self.raw_fallback_store.state(kv_cache)

    def clear_raw_fallback_runtime_state(self) -> None:
        """Release profiling-only workspace and sidecar tensor references."""
        self.raw_staging_manager.clear_shared_workspace()
        if self.raw_fallback_store is not None:
            self.raw_fallback_store.clear_binding()

    def reset_raw_fallback_pages(self, physical_block_ids: torch.Tensor) -> None:
        """Release raw sidecar slots for scheduler-reset physical blocks."""
        if self.raw_fallback_store is not None:
            self.raw_fallback_store.reset(physical_block_ids)

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
        use_unnormalized_partition_output: bool | None = None,
    ) -> tuple[int, ...]:
        if assume_no_outlier is None:
            assume_no_outlier = self.decode_assume_no_outlier
        if use_gqa_packed is None:
            use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        if use_unnormalized_partition_output is None:
            use_unnormalized_partition_output = self._use_unnormalized_partition_output(
                max_seq_len,
                use_gqa_packed=use_gqa_packed,
            )
        policy: tuple[int, ...] = (
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
                policy += (1, 1, 0, 0, 1)
                if use_unnormalized_partition_output:
                    policy += (_BYTE_V2_UNNORMALIZED_PARTITION_OUTPUT_MODE,)
                return policy
            return policy + (1,)
        return policy

    def _use_unnormalized_partition_output(
        self,
        max_seq_len: int,
        *,
        num_decode_tokens: int = 1,
        use_gqa_packed: bool | None = None,
    ) -> bool:
        if not self.decode_unnormalized_partition_output:
            return False
        if not self.decode_gqa_fa2_like or not self.decode_gqa_fa2_direct:
            return False
        if self.num_heads != self.num_kv_heads * 4:
            return False
        if use_gqa_packed is None:
            use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        if not use_gqa_packed:
            return False
        return self._use_split_k_decode(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        )

    def _decode_partition_size(
        self,
        max_seq_len: int,
        *,
        num_decode_tokens: int = 1,
        use_gqa_packed: bool | None = None,
        seq_len_sum: int | None = None,
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
                            seq_len_sum=seq_len_sum,
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
        *,
        seq_len_sum: int | None = None,
    ) -> int:
        # A fixed four-way split underutilizes the GPU when a B4 decode batch is
        # strongly ragged. The 0.8 threshold keeps the uniform/safe regime on
        # the existing heuristic while using the measured N64 sweet spot for
        # ragged 2K decode.
        if (
            num_decode_tokens == 4
            and 2048 <= max_seq_len < 4096
            and seq_len_sum is not None
            and seq_len_sum * 5 <= max_seq_len * num_decode_tokens * 4
        ):
            return self.tile_policy.compute_block_n
        num_splits = self._auto_gqa_fa2_direct_num_splits(
            max_seq_len,
            num_decode_tokens,
        )
        return self._partition_size_from_num_splits(max_seq_len, num_splits)

    def _speculative_gqa_partition_size(
        self,
        max_seq_len: int,
        query_len: int,
    ) -> int:
        assert query_len in (2, 4, 8, 16)
        if self.decode_gqa_packed_partition_size_explicit:
            return self.decode_gqa_packed_partition_size
        if query_len == 16:
            if max_seq_len >= 8192:
                return 512
            if max_seq_len >= 4096:
                return 256
            if max_seq_len >= 2048:
                return 128
            return 64
        if query_len == 8 and max_seq_len >= 8192:
            return 512
        if max_seq_len >= 16384:
            return 512
        if max_seq_len >= 4096:
            return 256
        if max_seq_len >= 2048:
            return 128
        return 64

    def _speculative_q4_partition_size(self, max_seq_len: int) -> int:
        return self._speculative_gqa_partition_size(max_seq_len, 4)

    def _speculative_ragged_q4_partition_size(self, max_seq_len: int) -> int:
        if self.decode_gqa_packed_partition_size_explicit:
            return self.decode_gqa_packed_partition_size
        if max_seq_len >= 16384:
            return 256
        if max_seq_len >= 8192:
            return 128
        return 64

    def _auto_gqa_fa2_direct_num_splits(
        self,
        max_seq_len: int,
        num_decode_tokens: int,
    ) -> int:
        # FA2-direct now runs multiple 64-token tiles inside a CTA. Keep small
        # split counts for medium context, then increase once per-CTA tile-loop
        # work dominates split/reduce overhead. This mirrors FA2's strategy of
        # choosing split count first, while using ByteV2-measured breakpoints.
        if num_decode_tokens >= 4:
            if max_seq_len >= 32768:
                return 64
            if max_seq_len >= 16384:
                return 32
            if max_seq_len >= 8192:
                return 32
            if max_seq_len >= 4096:
                return 4
            if max_seq_len >= 2048:
                return 4
            return max(1, (max_seq_len + 63) // 64)
        if num_decode_tokens >= 2:
            if max_seq_len >= 32768:
                return 64
            if max_seq_len >= 16384:
                return 64
            if max_seq_len >= 8192:
                return 8
            if max_seq_len >= 4096:
                return 8
            if max_seq_len >= 2048:
                return 8
            return max(1, (max_seq_len + 63) // 64)
        if max_seq_len >= 32768:
            return 16
        if max_seq_len >= 16384:
            return 16
        if max_seq_len >= 8192:
            return 16
        if max_seq_len >= 4096:
            return 16
        if max_seq_len >= 2048:
            return 16
        return max(1, (max_seq_len + 63) // 64)

    def _partition_size_from_num_splits(
        self,
        max_seq_len: int,
        num_splits: int,
    ) -> int:
        block = self.tile_policy.alloc_block_tokens
        num_splits = max(num_splits, 1)
        split_size = max(1, (max_seq_len + num_splits - 1) // num_splits)
        return max(block, ((split_size + block - 1) // block) * block)

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
        if not (
            self.decode_page_unsafe_flags
            and (
                self.decode_assume_no_outlier
                or self.speculative_verify_q4
                or self.speculative_verify_gqa
                or self.speculative_verify_ragged_q4
                or self.cached_prefix_q16
            )
        ):
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
    ) -> tuple[bool, bool, bool]:
        assume_no_outlier = self.decode_assume_no_outlier
        use_gqa_packed = self._use_gqa_packed_decode(max_seq_len)
        if not assume_no_outlier:
            return False, False, False
        if not self.decode_validate_no_outlier:
            return assume_no_outlier, use_gqa_packed, False
        if not (kv_cache.is_cuda and block_table.is_cuda and seq_lens.is_cuda):
            return False, False, False
        if self._decode_cache_has_fallback_or_outlier(
            kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
        ):
            return False, False, False
        return assume_no_outlier, use_gqa_packed, True

    def _get_split_k_workspace(
        self,
        output: torch.Tensor,
        max_seq_len: int,
        *,
        num_decode_tokens: int,
        use_gqa_packed: bool | None = None,
        use_unnormalized_partition_output: bool | None = None,
        partition_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if partition_size is None:
            partition_size = self._decode_partition_size(
                max_seq_len,
                num_decode_tokens=num_decode_tokens,
                use_gqa_packed=use_gqa_packed,
            )
        num_partitions = max(1, (max_seq_len + partition_size - 1) // partition_size)
        # CUDA graphs are captured in descending batch-size order. Reserve enough
        # partitions for every decode heuristic bucket on the first capture so a
        # later, smaller batch cannot replace storage referenced by an older graph.
        capacity_partitions = num_partitions
        for candidate_tokens in (1, 2, 4):
            candidate_partition_size = self._decode_partition_size(
                max_seq_len,
                num_decode_tokens=candidate_tokens,
                use_gqa_packed=use_gqa_packed,
                seq_len_sum=0 if candidate_tokens == 4 else None,
            )
            capacity_partitions = max(
                capacity_partitions,
                (max_seq_len + candidate_partition_size - 1)
                // candidate_partition_size,
            )
        num_output_tokens = output.shape[0]
        if use_unnormalized_partition_output is None:
            use_unnormalized_partition_output = self._use_unnormalized_partition_output(
                max_seq_len,
                num_decode_tokens=num_decode_tokens,
                use_gqa_packed=use_gqa_packed,
            )
        stats_shape = (num_output_tokens, self.num_heads, num_partitions)
        stats_numel = num_output_tokens * self.num_heads * num_partitions
        capacity_numel = (
            max(num_output_tokens, 4) * self.num_heads * capacity_partitions
        )
        needs_alloc = (
            self._decode_exp_sums is None
            or self._decode_exp_sums.device != output.device
            or self._decode_exp_sums.numel() < capacity_numel
        )
        if needs_alloc:
            allocation_numel = capacity_numel
            if (
                self._decode_exp_sums is not None
                and self._decode_exp_sums.device == output.device
            ):
                allocation_numel = max(
                    allocation_numel,
                    self._decode_exp_sums.numel(),
                )
            self._decode_exp_sums = torch.empty(
                (allocation_numel,),
                dtype=torch.float32,
                device=output.device,
            )
            self._decode_tmp_out = torch.empty(
                (allocation_numel * self.head_size,),
                dtype=torch.float32,
                device=output.device,
            )
        assert self._decode_exp_sums is not None
        assert self._decode_tmp_out is not None
        exp_sums = self._decode_exp_sums[:stats_numel].view(stats_shape)
        tmp_out = self._decode_tmp_out[: stats_numel * self.head_size].view(
            *stats_shape, self.head_size
        )

        if use_unnormalized_partition_output:
            if (
                self._decode_max_logits is None
                or self._decode_max_logits.device != output.device
                or self._decode_max_logits.numel() < capacity_numel
            ):
                self._decode_max_logits = torch.empty(
                    (capacity_numel,),
                    dtype=torch.float32,
                    device=output.device,
                )
            max_logits = self._decode_max_logits[:stats_numel].view(stats_shape)
        else:
            if (
                self._decode_empty_max_logits is None
                or self._decode_empty_max_logits.device != output.device
            ):
                self._decode_empty_max_logits = torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=output.device,
                )
            max_logits = self._decode_empty_max_logits

        return exp_sums, max_logits, tmp_out

    def _get_speculative_ragged_q4_workspace(
        self,
        output: torch.Tensor,
        max_seq_len: int,
        *,
        num_requests: int,
        partition_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_partitions = max(1, (max_seq_len + partition_size - 1) // partition_size)
        stats_shape = (num_requests, self.num_heads * 4, num_partitions)
        stats_numel = num_requests * self.num_heads * 4 * num_partitions
        needs_alloc = (
            self._speculative_ragged_exp_sums is None
            or self._speculative_ragged_exp_sums.device != output.device
            or self._speculative_ragged_exp_sums.numel() < stats_numel
        )
        if needs_alloc:
            allocation_numel = stats_numel
            if (
                self._speculative_ragged_exp_sums is not None
                and self._speculative_ragged_exp_sums.device == output.device
            ):
                allocation_numel = max(
                    allocation_numel,
                    self._speculative_ragged_exp_sums.numel() * 2,
                )
            self._speculative_ragged_exp_sums = torch.empty(
                (allocation_numel,),
                dtype=torch.float32,
                device=output.device,
            )
            self._speculative_ragged_tmp_out = torch.empty(
                (allocation_numel * self.head_size,),
                dtype=torch.float32,
                device=output.device,
            )
        assert self._speculative_ragged_exp_sums is not None
        assert self._speculative_ragged_tmp_out is not None
        exp_sums = self._speculative_ragged_exp_sums[:stats_numel].view(stats_shape)
        tmp_out = self._speculative_ragged_tmp_out[: stats_numel * self.head_size].view(
            *stats_shape, self.head_size
        )
        if (
            self._decode_empty_max_logits is None
            or self._decode_empty_max_logits.device != output.device
        ):
            self._decode_empty_max_logits = torch.empty(
                (0,),
                dtype=torch.float32,
                device=output.device,
            )
        return exp_sums, self._decode_empty_max_logits, tmp_out

    def _run_paged_decode(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        seq_len_sum: int | None = None,
    ) -> None:
        (
            assume_no_outlier,
            use_gqa_packed,
            validated_all_safe,
        ) = self._resolve_decode_fast_path(
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
        num_decode_tokens = int(output.shape[0])
        use_split_k = output.is_cuda and self._use_split_k_decode(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
        )
        use_unnormalized_partition_output = (
            use_split_k
            and self._use_unnormalized_partition_output(
                max_seq_len,
                num_decode_tokens=num_decode_tokens,
                use_gqa_packed=use_gqa_packed,
            )
        )
        tile_policy = self._decode_tile_policy_tuple(
            max_seq_len,
            assume_no_outlier=assume_no_outlier,
            use_gqa_packed=use_gqa_packed,
            use_unnormalized_partition_output=use_unnormalized_partition_output,
        )
        partition_size = self._decode_partition_size(
            max_seq_len,
            num_decode_tokens=num_decode_tokens,
            use_gqa_packed=use_gqa_packed,
            seq_len_sum=seq_len_sum,
        )
        if use_split_k:
            exp_sums, max_logits, tmp_out = self._get_split_k_workspace(
                output,
                max_seq_len,
                num_decode_tokens=num_decode_tokens,
                use_gqa_packed=use_gqa_packed,
                use_unnormalized_partition_output=(use_unnormalized_partition_output),
                partition_size=partition_size,
            )
            if (
                assume_no_outlier
                and self.decode_page_unsafe_flags
                and page_unsafe_flags is not None
                and not validated_all_safe
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

    @staticmethod
    def _metadata_seq_lens_cpu(
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = attn_metadata.seq_lens.detach().cpu()
        return seq_lens_cpu

    def _speculative_gqa_query_len(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> int | None:
        query_len = attn_metadata.max_query_len
        enabled = self.speculative_verify_gqa and query_len in (2, 4, 8)
        enabled = enabled or (self.speculative_verify_q4 and query_len == 4)
        enabled = enabled or (self.cached_prefix_q16 and query_len == 16)
        if not enabled:
            return None
        if not (query.is_cuda and output.is_cuda and kv_cache.is_cuda):
            return None
        if query.shape != output.shape or output.ndim != 3:
            return None
        if attn_metadata.num_actual_tokens != output.shape[0]:
            return None
        if output.shape[0] == 0 or output.shape[0] % query_len != 0:
            return None
        if not attn_metadata.causal:
            return None
        if (
            self.num_heads != 32
            or self.num_kv_heads != 8
            or self.head_size != 128
            or self.tile_policy.alloc_block_tokens != 16
            or self.tile_policy.compute_block_n != 64
        ):
            return None
        if self.alibi_slopes is not None or self.sliding_window is not None:
            return None
        if self.logits_soft_cap is not None:
            return None

        query_start_locs = attn_metadata.query_start_loc_cpu.tolist()
        num_requests = output.shape[0] // query_len
        if len(query_start_locs) != num_requests + 1:
            return None
        if any(
            int(end) - int(start) != query_len
            for start, end in zip(query_start_locs[:-1], query_start_locs[1:])
        ):
            return None
        if int(query_start_locs[-1]) != output.shape[0]:
            return None
        if (
            attn_metadata.block_table.ndim != 2
            or attn_metadata.block_table.shape[0] < num_requests
            or attn_metadata.seq_lens.shape[0] < num_requests
        ):
            return None
        seq_lens_cpu = self._metadata_seq_lens_cpu(attn_metadata)
        if seq_lens_cpu.shape[0] < num_requests:
            return None
        if not bool(torch.all(seq_lens_cpu[:num_requests] > query_len).item()):
            return None
        return query_len

    def _run_speculative_verify_gqa(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
        query_len: int,
    ) -> bool:
        page_unsafe_flags = self._usable_decode_page_unsafe_flags(kv_cache)
        if page_unsafe_flags is None:
            return False

        num_requests = output.shape[0] // query_len
        virtual_heads = query_len * self.num_heads
        partition_size = self._speculative_gqa_partition_size(
            attn_metadata.max_seq_len,
            query_len,
        )
        exp_sums, max_logits, tmp_out = self._get_split_k_workspace(
            output,
            attn_metadata.max_seq_len,
            num_decode_tokens=output.shape[0],
            use_gqa_packed=True,
            use_unnormalized_partition_output=False,
            partition_size=partition_size,
        )
        num_partitions = exp_sums.shape[2]
        virtual_exp_sums = exp_sums.view(
            num_requests,
            virtual_heads,
            num_partitions,
        )
        virtual_tmp_out = tmp_out.view(
            num_requests,
            virtual_heads,
            num_partitions,
            self.head_size,
        )
        byte_v2_speculative_verify_gqa(
            output,
            virtual_exp_sums,
            max_logits,
            virtual_tmp_out,
            query,
            kv_cache,
            page_unsafe_flags,
            attn_metadata.block_table[:num_requests],
            attn_metadata.seq_lens[:num_requests],
            speculative_query_len=query_len,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            block_size=self.tile_policy.alloc_block_tokens,
            max_seq_len=attn_metadata.max_seq_len,
            partition_size=partition_size,
            tile_policy=self._base_tile_policy_tuple(),
        )
        return True

    def _speculative_ragged_q4_num_requests(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> int | None:
        if not self.speculative_verify_ragged_q4:
            return None
        if not 1 < attn_metadata.max_query_len <= 4:
            return None
        if not (query.is_cuda and output.is_cuda and kv_cache.is_cuda):
            return None
        if query.shape != output.shape or output.ndim != 3:
            return None
        if not 0 < attn_metadata.num_actual_tokens <= output.shape[0]:
            return None
        if not attn_metadata.causal:
            return None
        if (
            self.num_heads != 32
            or self.num_kv_heads != 8
            or self.head_size != 128
            or self.tile_policy.alloc_block_tokens != 16
            or self.tile_policy.compute_block_n != 64
        ):
            return None
        if self.alibi_slopes is not None or self.sliding_window is not None:
            return None
        if self.logits_soft_cap is not None:
            return None

        query_start_locs = attn_metadata.query_start_loc_cpu.tolist()
        if len(query_start_locs) < 2 or int(query_start_locs[0]) != 0:
            return None
        if any(
            int(end) < int(start)
            for start, end in zip(query_start_locs[:-1], query_start_locs[1:])
        ):
            return None
        num_actual_tokens = attn_metadata.num_actual_tokens
        clipped_starts = [
            max(0, min(int(start), num_actual_tokens)) for start in query_start_locs
        ]
        if clipped_starts[-1] != num_actual_tokens:
            return None
        query_lens = [
            end - start for start, end in zip(clipped_starts[:-1], clipped_starts[1:])
        ]
        if any(query_len < 0 or query_len > 4 for query_len in query_lens):
            return None
        if not any(query_len > 1 for query_len in query_lens):
            return None

        num_requests = len(query_lens)
        if (
            attn_metadata.query_start_loc.ndim != 1
            or attn_metadata.query_start_loc.shape[0] < num_requests + 1
            or attn_metadata.query_start_loc.dtype != torch.int32
            or attn_metadata.block_table.ndim != 2
            or attn_metadata.block_table.shape[0] < num_requests
            or attn_metadata.seq_lens.shape[0] < num_requests
        ):
            return None
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return None
        if seq_lens_cpu.shape[0] < num_requests:
            return None
        if any(
            query_len > 0 and int(seq_lens_cpu[idx]) <= query_len
            for idx, query_len in enumerate(query_lens)
        ):
            return None
        return num_requests

    def _run_speculative_verify_ragged_q4(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
        num_requests: int,
    ) -> bool:
        page_unsafe_flags = self._usable_decode_page_unsafe_flags(kv_cache)
        if page_unsafe_flags is None:
            return False

        partition_size = self._speculative_ragged_q4_partition_size(
            attn_metadata.max_seq_len
        )
        exp_sums, max_logits, tmp_out = self._get_speculative_ragged_q4_workspace(
            output,
            attn_metadata.max_seq_len,
            num_requests=num_requests,
            partition_size=partition_size,
        )
        byte_v2_speculative_verify_ragged_q4(
            output,
            exp_sums,
            max_logits,
            tmp_out,
            query,
            kv_cache,
            page_unsafe_flags,
            attn_metadata.block_table[:num_requests],
            attn_metadata.seq_lens[:num_requests],
            attn_metadata.query_start_loc[: num_requests + 1],
            num_actual_tokens=attn_metadata.num_actual_tokens,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            block_size=self.tile_policy.alloc_block_tokens,
            max_seq_len=attn_metadata.max_seq_len,
            partition_size=partition_size,
            tile_policy=self._base_tile_policy_tuple(),
        )
        return True

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
        has_cached_context = self._prefill_has_cached_context(attn_metadata)
        if self.raw_fallback_store is not None:
            # Keep initial and cache-backed prefill on the same FA2 template as
            # raw serving. The hybrid loader is the only cache reader that
            # understands both compact pages and authoritative raw sidecars.
            # Routing before the legacy direct-QKV paths also prevents SDPA
            # reduction differences from changing an otherwise exact greedy
            # trajectory.
            if not has_cached_context and self._forward_initial_prefill_from_staging(
                query,
                key,
                value,
                kv_cache,
                output,
                attn_metadata,
            ):
                return output
            if self._forward_direct_paged_prefill(
                query,
                key,
                value,
                kv_cache,
                output,
                attn_metadata,
            ):
                return output
            return self._forward_prefill_from_cache(
                query,
                kv_cache,
                output,
                attn_metadata,
            )
        speculative_query_len = self._speculative_gqa_query_len(
            query,
            kv_cache,
            output,
            attn_metadata,
        )
        if speculative_query_len is not None and self._run_speculative_verify_gqa(
            query, kv_cache, output, attn_metadata, speculative_query_len
        ):
            return output
        ragged_num_requests = self._speculative_ragged_q4_num_requests(
            query,
            kv_cache,
            output,
            attn_metadata,
        )
        if ragged_num_requests is not None and self._run_speculative_verify_ragged_q4(
            query,
            kv_cache,
            output,
            attn_metadata,
            ragged_num_requests,
        ):
            return output
        if has_cached_context:
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

    def _forward_initial_prefill_from_staging(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        """Run an initial prefill from exact transient BF16 pages when they fit."""
        if not attn_metadata.causal:
            return False
        num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
        if num_actual_tokens <= 0:
            return True
        staged = self.raw_staging_manager.stage_initial_prefill(
            key=key[:num_actual_tokens],
            value=value[:num_actual_tokens],
            kv_cache=kv_cache,
            slot_mapping=attn_metadata.slot_mapping[:num_actual_tokens],
            attn_metadata=attn_metadata,
        )
        if staged is None:
            return False
        raw_staging, page_to_raw_slot, staging_to_physical_block, valid_rows = staged
        batch_size = min(
            attn_metadata.query_start_loc.numel() - 1,
            attn_metadata.block_table.shape[0],
            attn_metadata.seq_lens.shape[0],
        )
        try:
            byte_v2_fa2_raw_staging_prefill_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                raw_staging,
                page_to_raw_slot,
                attn_metadata.query_start_loc[: batch_size + 1],
                attn_metadata.block_table[:batch_size],
                attn_metadata.seq_lens[:batch_size],
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                block_size=self.tile_policy.alloc_block_tokens,
                head_dim=self.head_size,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                causal=True,
            )
        finally:
            self.raw_staging_manager.release_initial_prefill(
                staging_to_physical_block,
                valid_rows,
            )
        return True

    def _forward_direct_paged_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        """Split a safe cached-prefix/initial-suffix batch without K/V copies."""
        plan = getattr(attn_metadata, "direct_prefill_plan", None)
        if not getattr(self, "direct_paged_prefill", False) or plan is None:
            return False
        raw_fallback_store = self.raw_fallback_store
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        num_requests = plan.cached_request_count + sum(
            group.num_requests for group in plan.groups
        )
        if (
            not attn_metadata.causal
            or not plan.groups
            or num_actual_tokens <= 0
            or plan.cached_request_count < 0
            or not 0 <= plan.cached_token_count <= num_actual_tokens
            or query.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or output.device != query.device
            or key.device != query.device
            or value.device != query.device
            or query.ndim != 3
            or output.shape != query.shape
            or key.ndim != 3
            or value.shape != key.shape
            or query.shape[0] < num_actual_tokens
            or key.shape[0] < num_actual_tokens
            or value.shape[0] < num_actual_tokens
            or query.shape[1:] != (self.num_heads, self.head_size)
            or key.shape[1:] != (self.num_kv_heads, self.head_size)
            or query.stride()[1:] != (self.head_size, 1)
            or output.stride()[1:] != (self.head_size, 1)
            or key.stride()[1:] != (self.head_size, 1)
            or value.stride()[1:] != (self.head_size, 1)
            or query.stride(0) < self.num_heads * self.head_size
            or output.stride(0) < self.num_heads * self.head_size
            or key.stride(0) < self.num_kv_heads * self.head_size
            or value.stride(0) < self.num_kv_heads * self.head_size
            or attn_metadata.query_start_loc.dtype != torch.int32
            or attn_metadata.query_start_loc.device != query.device
            or attn_metadata.query_start_loc.ndim != 1
            or attn_metadata.query_start_loc.shape[0] < num_requests + 1
            or attn_metadata.block_table.dtype != torch.int32
            or attn_metadata.block_table.device != query.device
            or attn_metadata.block_table.ndim != 2
            or attn_metadata.block_table.shape[0] < num_requests
            or attn_metadata.seq_lens.dtype != torch.int32
            or attn_metadata.seq_lens.device != query.device
            or attn_metadata.seq_lens.ndim != 1
            or attn_metadata.seq_lens.shape[0] < num_requests
            or (plan.cached_request_count > 0 and raw_fallback_store is None)
        ):
            return False

        # Preflight every view before launching either side of the split. A
        # ragged group may borrow masked rows from the following group, but it
        # must never read beyond the padded projection tensor's storage.
        direct_calls = []
        expected_request = plan.cached_request_count
        expected_token = plan.cached_token_count
        for group in plan.groups:
            rounded_end = group.first_token + group.rounded_tokens
            if (
                group.first_request != expected_request
                or group.first_token != expected_token
                or group.num_requests <= 0
                or group.num_tokens <= 0
                or group.rounded_tokens < group.num_tokens
                or group.rounded_tokens % self.tile_policy.alloc_block_tokens != 0
                or group.max_query_len <= 0
                or group.query_start_loc.dtype != torch.int32
                or group.query_start_loc.device != query.device
                or group.query_start_loc.ndim != 1
                or group.query_start_loc.shape[0] != group.num_requests + 1
                or group.block_table.dtype != torch.int32
                or group.block_table.device != query.device
                or group.block_table.ndim != 2
                or group.block_table.shape[0] != group.num_requests
                or rounded_end > key.shape[0]
                or rounded_end > value.shape[0]
                or group.first_token + group.num_tokens > num_actual_tokens
            ):
                return False
            num_pages = group.rounded_tokens // self.tile_policy.alloc_block_tokens
            try:
                key_pages = key.narrow(
                    0,
                    group.first_token,
                    group.rounded_tokens,
                ).view(
                    num_pages,
                    self.tile_policy.alloc_block_tokens,
                    self.num_kv_heads,
                    self.head_size,
                )
                value_pages = value.narrow(
                    0,
                    group.first_token,
                    group.rounded_tokens,
                ).view_as(key_pages)
            except RuntimeError:
                return False
            request_end = group.first_request + group.num_requests
            direct_calls.append(
                (
                    group,
                    key_pages,
                    value_pages,
                    attn_metadata.seq_lens[group.first_request : request_end],
                )
            )
            expected_request = request_end
            expected_token = group.first_token + group.num_tokens

        if expected_request != num_requests or expected_token != num_actual_tokens:
            return False

        if plan.cached_request_count:
            assert raw_fallback_store is not None
            hybrid_state = raw_fallback_store.state(kv_cache)
            byte_v2_fa2_hybrid_paged_decode_attention(
                output[: plan.cached_token_count],
                query[: plan.cached_token_count],
                kv_cache,
                hybrid_state.raw_pages,
                hybrid_state.page_to_raw_slot,
                attn_metadata.query_start_loc[: plan.cached_request_count + 1],
                attn_metadata.block_table[: plan.cached_request_count],
                attn_metadata.seq_lens[: plan.cached_request_count],
                scale=self.scale,
                max_query_len=plan.cached_max_query_len,
                max_seq_len=plan.cached_max_seq_len,
                causal=True,
                preserve_mixed_dispatch=True,
            )

        for group, key_pages, value_pages, seq_lens in direct_calls:
            token_end = group.first_token + group.num_tokens
            byte_v2_fa2_direct_paged_prefill_attention(
                output[group.first_token : token_end],
                query[group.first_token : token_end],
                key_pages,
                value_pages,
                group.query_start_loc,
                group.block_table,
                seq_lens,
                scale=self.scale,
                max_query_len=group.max_query_len,
                max_seq_len=group.max_query_len,
                causal=True,
                preserve_mixed_dispatch=True,
            )
        return True

    @staticmethod
    def _prefill_has_cached_context(
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        # The common metadata's CPU upper bound is exact for prefill rows. In
        # async speculative decode it can only conservatively keep a decode row
        # on the cached path, which is the fail-closed routing decision. Reuse
        # it here instead of synchronously copying device seq_lens in every
        # layer.
        seq_lens_cpu = getattr(
            attn_metadata,
            "seq_lens_cpu_upper_bound",
            None,
        )
        if seq_lens_cpu is None:
            seq_lens_cpu = ByteV2AttentionImpl._metadata_seq_lens_cpu(attn_metadata)
        num_actual_tokens = attn_metadata.num_actual_tokens

        for seq_idx, (start, end) in enumerate(
            zip(query_start_locs[:-1], query_start_locs[1:])
        ):
            start = min(int(start), num_actual_tokens)
            end = min(int(end), num_actual_tokens)
            query_len = end - start
            if query_len <= 0 or seq_idx >= seq_lens_cpu.shape[0]:
                continue
            if int(seq_lens_cpu[seq_idx]) > query_len:
                return True
        return False

    def _forward_cached_prefill_from_hydrated_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        """Hydrate one complete B1 cached sequence, then run raw paged FA2."""
        hydrate_mode = getattr(
            self,
            "cached_prefill_hydrate_to_raw_mode",
            "enabled",
        )
        if (
            not getattr(self, "cached_prefill_hydrate_to_raw", False)
            or hydrate_mode == "disabled"
            or self.raw_fallback_store is None
            or not attn_metadata.causal
            or attn_metadata.max_query_len <= 1
            or (
                hydrate_mode == "auto"
                and attn_metadata.max_query_len
                < _BYTE_V2_CACHED_PREFILL_AUTO_MIN_QUERY_LEN
            )
        ):
            return False

        num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        is_prefilling = getattr(attn_metadata, "is_prefilling", None)
        if (
            num_actual_tokens <= 1
            or attn_metadata.max_query_len != num_actual_tokens
            or not isinstance(query_start_loc_cpu, torch.Tensor)
            or query_start_loc_cpu.device.type != "cpu"
            or query_start_loc_cpu.ndim != 1
            or query_start_loc_cpu.numel() < 2
            or not isinstance(is_prefilling, torch.Tensor)
            or is_prefilling.device.type != "cpu"
            or is_prefilling.dtype != torch.bool
            or is_prefilling.ndim != 1
            or is_prefilling.numel() < query_start_loc_cpu.numel() - 1
        ):
            return False
        query_start_locs = [int(value) for value in query_start_loc_cpu.tolist()]
        num_request_rows = len(query_start_locs) - 1
        if (
            query_start_locs[0] != 0
            or query_start_locs[1] != num_actual_tokens
            or any(value != num_actual_tokens for value in query_start_locs[2:])
            or not bool(is_prefilling[0])
            or bool(is_prefilling[1:num_request_rows].any())
        ):
            return False

        seq_len = int(attn_metadata.max_seq_len)
        seq_lens_cpu = getattr(
            attn_metadata,
            "seq_lens_cpu_upper_bound",
            None,
        )
        if (
            seq_len <= num_actual_tokens
            or not isinstance(seq_lens_cpu, torch.Tensor)
            or seq_lens_cpu.device.type != "cpu"
            or seq_lens_cpu.ndim != 1
            or seq_lens_cpu.numel() < 1
            or int(seq_lens_cpu[0]) != seq_len
            or attn_metadata.block_table.dtype != torch.int32
            or attn_metadata.block_table.device != query.device
            or attn_metadata.block_table.ndim != 2
            or attn_metadata.block_table.shape[0] < 1
            or attn_metadata.seq_lens.dtype != torch.int32
            or attn_metadata.seq_lens.device != query.device
            or attn_metadata.seq_lens.ndim != 1
            or attn_metadata.seq_lens.shape[0] < 1
        ):
            return False

        num_pages = (
            seq_len + self.tile_policy.alloc_block_tokens - 1
        ) // self.tile_policy.alloc_block_tokens
        if attn_metadata.block_table.shape[1] < num_pages:
            return False
        staged = self.raw_staging_manager.stage_cached_prefill(
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:1, :num_pages],
            seq_len=seq_len,
        )
        if staged is None:
            return False

        raw_staging, page_map, local_block_table, valid_rows = staged
        try:
            byte_v2_fa2_raw_staging_prefill_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                raw_staging,
                page_map,
                attn_metadata.query_start_loc[:2],
                local_block_table,
                attn_metadata.seq_lens[:1],
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                block_size=self.tile_policy.alloc_block_tokens,
                head_dim=self.head_size,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=seq_len,
                causal=True,
                block_tables_are_staging_slots=True,
            )
        finally:
            self.raw_staging_manager.release_cached_prefill(
                local_block_table,
                valid_rows,
            )
        return True

    def _forward_prefill_from_cache(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> torch.Tensor:
        common_prefix_len = int(getattr(attn_metadata, "common_prefix_len", 0))

        if self.raw_fallback_store is not None:
            if not attn_metadata.causal:
                raise RuntimeError(
                    "ByteV2 hybrid FA2 cache reads require causal attention"
                )
            num_actual_tokens = min(attn_metadata.num_actual_tokens, query.shape[0])
            if num_actual_tokens <= 0:
                return output
            batch_size = min(
                attn_metadata.query_start_loc.numel() - 1,
                attn_metadata.block_table.shape[0],
                attn_metadata.seq_lens.shape[0],
            )
            if batch_size <= 0:
                return output
            if self._forward_cached_prefill_from_hydrated_cache(
                query,
                kv_cache,
                output,
                attn_metadata,
            ):
                return output
            hybrid_state = self.raw_fallback_store.state(kv_cache)
            byte_v2_fa2_hybrid_paged_decode_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                kv_cache,
                hybrid_state.raw_pages,
                hybrid_state.page_to_raw_slot,
                attn_metadata.query_start_loc[: batch_size + 1],
                attn_metadata.block_table[:batch_size],
                attn_metadata.seq_lens[:batch_size],
                scale=self.scale,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=attn_metadata.max_seq_len,
                causal=attn_metadata.causal,
            )
            return output

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu()
        query_start_locs = query_start_loc_cpu.tolist()
        seq_lens_cpu = self._metadata_seq_lens_cpu(attn_metadata)
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
            if seq_idx < seq_lens_cpu.shape[0]:
                context_len = max(
                    int(seq_lens_cpu[seq_idx]) - query_len,
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

    def _fa2_decode_incompatibility(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> str | None:
        if not self.decode_fa2_available:
            return "the ByteV2 FA2 extension op is unavailable"
        if attn_metadata.max_query_len != 1:
            return "only Q1 decode is supported"
        if not attn_metadata.causal or self.attn_type != AttentionType.DECODER:
            return "only causal decoder Q1 attention is supported"
        if self.alibi_slopes is not None:
            return "ALiBi is unsupported"
        if self.sliding_window is not None:
            return "sliding-window/local attention is unsupported"
        if self.logits_soft_cap is not None and self.logits_soft_cap > 0:
            return "logit softcap is unsupported"
        if self.tile_policy != DEFAULT_BYTE_V2_TILE_POLICY:
            return "the default ByteV2 tile policy is required"
        if attn_metadata.tile_policy != DEFAULT_BYTE_V2_TILE_POLICY:
            return "metadata does not use the default ByteV2 tile policy"
        if self.num_heads != 32 or self.num_kv_heads != 8 or self.head_size != 128:
            return "local Hq=32, Hkv=8, and D=128 are required"
        if query.ndim != 3 or query.shape != output.shape:
            return "query and output must have the same rank-3 shape"
        if output.shape[0] <= 0 or tuple(output.shape[1:]) != (32, 128):
            return "query and output must have shape (total_q, 32, 128)"
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            return "query and output must use BF16"
        if not (query.is_cuda and output.is_cuda and kv_cache.is_cuda):
            return "query, output, and KV cache must be CUDA tensors"
        if not (query.device == output.device == kv_cache.device):
            return "query, output, and KV cache must share a CUDA device"
        if query.stride(-1) != 1 or output.stride(-1) != 1:
            return "query and output must have a contiguous head dimension"
        if (
            kv_cache.dtype != torch.uint8
            or kv_cache.ndim != 2
            or kv_cache.shape[0] <= 0
            or kv_cache.shape[1] != _BYTE_V2_FA2_PAGE_SIZE_BYTES
            or not kv_cache.is_contiguous()
        ):
            return "KV cache must be contiguous V6 uint8 pages of 50560 bytes"

        batch_size = output.shape[0]
        query_start_locs = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        if (
            query_start_locs.ndim != 1
            or query_start_locs.numel() != batch_size + 1
            or query_start_locs.dtype != torch.int32
            or not query_start_locs.is_cuda
            or not query_start_locs.is_contiguous()
            or query_start_locs.device != query.device
        ):
            return "query_start_loc must be CUDA int32 with total_q + 1 entries"
        if (
            seq_lens.ndim != 1
            or seq_lens.shape[0] < batch_size
            or seq_lens.dtype != torch.int32
            or not seq_lens.is_cuda
            or not seq_lens.is_contiguous()
            or seq_lens.device != query.device
        ):
            return "seq_lens must provide one contiguous CUDA int32 value per query"
        if (
            block_table.ndim != 2
            or block_table.shape[0] < batch_size
            or block_table.dtype != torch.int32
            or not block_table.is_cuda
            or block_table.stride(-1) != 1
            or block_table.device != query.device
        ):
            return "block_table must provide one CUDA int32 row per query"
        if attn_metadata.max_seq_len <= 0:
            return "max_seq_len must be positive"
        if (
            attn_metadata.max_seq_len
            > block_table.shape[1] * self.tile_policy.alloc_block_tokens
        ):
            return "max_seq_len exceeds block-table capacity"
        return None

    def _run_fa2_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: ByteV2AttentionMetadata,
    ) -> bool:
        if self.decode_kernel_mode == "legacy" and not self.fa2_hybrid_raw_fallback:
            return False
        incompatibility = self._fa2_decode_incompatibility(
            query,
            kv_cache,
            output,
            attn_metadata,
        )
        if incompatibility is not None:
            if self.decode_kernel_mode == "fa2" or self.fa2_hybrid_raw_fallback:
                raise RuntimeError(
                    f"ByteV2 FA2 decode cannot run this decode: {incompatibility}"
                )
            return False

        batch_size = output.shape[0]
        if self.raw_fallback_store is None:
            byte_v2_fa2_paged_decode_attention(
                output,
                query,
                kv_cache,
                attn_metadata.query_start_loc,
                attn_metadata.block_table[:batch_size],
                attn_metadata.seq_lens[:batch_size],
                scale=self.scale,
                max_seq_len=attn_metadata.max_seq_len,
                causal=attn_metadata.causal,
            )
        else:
            hybrid_state = self.raw_fallback_store.state(kv_cache)
            byte_v2_fa2_hybrid_paged_decode_attention(
                output,
                query,
                kv_cache,
                hybrid_state.raw_pages,
                hybrid_state.page_to_raw_slot,
                attn_metadata.query_start_loc,
                attn_metadata.block_table[:batch_size],
                attn_metadata.seq_lens[:batch_size],
                scale=self.scale,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                causal=attn_metadata.causal,
            )
        return True

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
        if self._run_fa2_decode(query, kv_cache, output, attn_metadata):
            _debug_warmup("decode forward done via FA2 template")
            return output
        self._run_paged_decode(
            output,
            query,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            seq_len_sum=getattr(attn_metadata, "seq_len_sum", None),
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
        self._do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
            None,
        )

    def do_kv_cache_update_with_metadata(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: object | None,
    ) -> None:
        self._do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
            attn_metadata,
        )

    def _do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: object | None,
    ) -> None:
        del layer
        query_start_locs = _trusted_byte_v2_query_start_locs(
            attn_metadata,
            slot_mapping.shape[0],
        )
        if query_start_locs is not None:
            num_actual_tokens = query_start_locs[-1]
            key = key[:num_actual_tokens]
            value = value[:num_actual_tokens]
            slot_mapping = slot_mapping[:num_actual_tokens]
        _debug_warmup(
            "kv update start key=%s kv_cache=%s slot_mapping=%s",
            tuple(key.shape),
            tuple(kv_cache.shape),
            tuple(slot_mapping.shape),
        )
        page_unsafe_flags = None
        if (
            self.decode_page_unsafe_flags
            and (
                self.decode_assume_no_outlier
                or self.speculative_verify_q4
                or self.speculative_verify_gqa
                or self.speculative_verify_ragged_q4
                or self.cached_prefix_q16
            )
            and kv_cache.is_cuda
            and slot_mapping.is_cuda
        ):
            page_unsafe_flags = self._get_decode_page_unsafe_flags(kv_cache)
        if (
            page_unsafe_flags is None
            and not self.fa2_hybrid_raw_fallback
            and _fused_single_token_staging_enabled()
            and slot_mapping.shape[0] == 1
            and kv_cache.is_cuda
            and slot_mapping.is_cuda
        ):
            page_unsafe_flags = self._get_decode_page_unsafe_flags(kv_cache)
        byte_v2_attn_metadata = (
            attn_metadata
            if isinstance(attn_metadata, ByteV2AttentionMetadata)
            else None
        )
        handled, flags_updated = self.raw_staging_manager.update(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            page_unsafe_flags=page_unsafe_flags,
            attn_metadata=attn_metadata,
            retain_initial_prefill=(
                self.raw_fallback_store is not None
                and byte_v2_attn_metadata is not None
                and slot_mapping.shape[0] > 1
                and byte_v2_attn_metadata.causal
                and byte_v2_attn_metadata.max_query_len > 1
                and not self._prefill_has_cached_context(byte_v2_attn_metadata)
            ),
        )
        if handled:
            if not flags_updated:
                self._update_decode_page_unsafe_flags(kv_cache, slot_mapping)
            _debug_warmup("kv update done via cache update manager")
            return
        if self.fa2_hybrid_raw_fallback:
            raise RuntimeError(
                "ByteV2 hybrid raw fallback cannot use the direct cache writer"
            )
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
