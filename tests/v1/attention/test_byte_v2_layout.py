# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends import byte_v2_attn as byte_v2_attn_module
from vllm.v1.attention.backends import byte_v2_ops as byte_v2_ops_module
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_MACRO_DESCRIPTOR_BYTES,
    BYTE_V2_MAX_MACRO_PAGES,
    DEFAULT_BYTE_V2_TILE_POLICY,
    ByteV2CodecPayloadPolicy,
    ByteV2OutlierEntryPolicy,
    ByteV2PagedKVManager,
    ByteV2PageLayoutV4,
    ByteV2PageLayoutV5,
    ByteV2RawStagingLayout,
    ByteV2TilePolicy,
    byte_v2_tile_policy_from_env,
    collect_byte_v2_fallback_stats,
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
    byte_v2_release_raw_staging_and_update_flags,
    byte_v2_reshape_and_cache,
    byte_v2_reshape_and_cache_sideband_high,
    byte_v2_speculative_verify_gqa,
    byte_v2_speculative_verify_ragged_q4,
    byte_v2_test_force_promote_raw_staging_q1,
    byte_v2_update_cache_raw_staging,
    byte_v2_update_cache_single_token,
    byte_v2_update_cache_unsafe_flags,
    byte_v2_update_hybrid_cache_raw_staging_multi_token,
    byte_v2_update_hybrid_cache_raw_staging_multi_token_retained,
    byte_v2_update_hybrid_cache_raw_staging_q1,
    missing_byte_v2_custom_ops,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec


def _has_torch_op(namespace: str, op_name: str) -> bool:
    byte_v2_custom_ops_are_available()
    op_namespace = getattr(torch.ops, namespace, None)
    return op_namespace is not None and getattr(op_namespace, op_name, None) is not None


def _bf16_bits(value: torch.Tensor) -> int:
    return int(value.detach().cpu().view(torch.int16).item()) & 0xFFFF


def _float_from_bf16_bits(bits: int) -> float:
    signed_bits = bits if bits < 0x8000 else bits - 0x10000
    return (
        torch.tensor([signed_bits], dtype=torch.int16)
        .view(torch.bfloat16)
        .float()
        .item()
    )


def _load_u16_bytes(cache: torch.Tensor, physical_block: int, offset: int) -> int:
    return int(cache[physical_block, offset]) | (
        int(cache[physical_block, offset + 1]) << 8
    )


def _load_u32_bytes(cache: torch.Tensor, physical_block: int, offset: int) -> int:
    return (
        int(cache[physical_block, offset])
        | (int(cache[physical_block, offset + 1]) << 8)
        | (int(cache[physical_block, offset + 2]) << 16)
        | (int(cache[physical_block, offset + 3]) << 24)
    )


def _v5_outlier_payload_offset(
    cache: torch.Tensor,
    layout: ByteV2PageLayoutV5,
    *,
    physical_block: int,
    kv_head: int,
    dim_tile: int,
    is_value: bool,
    entry_idx: int = 0,
) -> int:
    pool_offset = (
        layout.v_outlier_pool_index_offset(kv_head=kv_head, dim_tile=dim_tile)
        if is_value
        else layout.k_outlier_pool_index_offset(kv_head=kv_head, dim_tile=dim_tile)
    )
    pool_entry_index = _load_u16_bytes(cache, physical_block, pool_offset)
    return layout.k_outlier_payload_offset(
        kv_head=kv_head,
        dim_tile=dim_tile,
        pool_entry_index=pool_entry_index,
        entry_idx=entry_idx,
    )


def _raw_payload_layout(*, outlier_entries_per_tile: int = 128) -> ByteV2PageLayoutV4:
    return ByteV2PageLayoutV4(
        include_raw_payload=True,
        outlier_entries_per_tile=outlier_entries_per_tile,
    )


def _decode_current_byte_v2_payload(
    cache: torch.Tensor,
    layout: ByteV2PageLayoutV4 | ByteV2PageLayoutV5,
    *,
    physical_block: int,
    kv_head: int,
    row: int,
    dim: int,
    is_value: bool,
) -> float:
    policy = layout.tile_policy
    dim_tile = dim // policy.codec_dim_block
    dim_in_tile = dim % policy.codec_dim_block
    token_tile = row // policy.codec_token_block
    row_in_tile = row % policy.codec_token_block
    tile_offset = (
        layout.v_payload_offset(
            kv_head=kv_head,
            dim_tile=dim_tile,
            token_tile=token_tile,
        )
        if is_value
        else layout.k_payload_offset(
            kv_head=kv_head,
            dim_tile=dim_tile,
            token_tile=token_tile,
        )
    )
    tile_index = (
        token_tile * policy.v_dim_tiles + dim_tile
        if is_value
        else dim_tile * policy.codec_token_tiles_per_alloc_block + token_tile
    )
    fallback_mask = _load_u32_bytes(
        cache,
        physical_block,
        (
            layout.v_fallback_mask_offset(kv_head=kv_head)
            if is_value
            else layout.k_fallback_mask_offset(kv_head=kv_head)
        ),
    )
    if fallback_mask & (1 << tile_index):
        raw_offset = (
            layout.raw_value_offset(kv_head=kv_head, row=row, dim=dim)
            if is_value
            else layout.raw_key_offset(kv_head=kv_head, row=row, dim=dim)
        )
        return _float_from_bf16_bits(_load_u16_bytes(cache, physical_block, raw_offset))

    elem_idx = row_in_tile * policy.codec_dim_block + dim_in_tile
    low = int(cache[physical_block, tile_offset + elem_idx])
    packed_code = int(
        cache[physical_block, tile_offset + policy.codec_tile_elems + elem_idx // 2]
    )
    code = (packed_code >> 4) if elem_idx % 2 else (packed_code & 0x0F)
    base = int(
        cache[
            physical_block,
            (
                layout.v_base_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
                if is_value
                else layout.k_base_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
            ),
        ]
    )
    high = ((code >> 3) << 7) | (base + (code & 0x07))

    outlier_mask = _load_u32_bytes(
        cache,
        physical_block,
        (
            layout.v_outlier_mask_offset(kv_head=kv_head)
            if is_value
            else layout.k_outlier_mask_offset(kv_head=kv_head)
        ),
    )
    if outlier_mask & (1 << tile_index):
        count_offset = (
            layout.v_outlier_count_offset(
                kv_head=kv_head,
                dim_tile=dim_tile,
                token_tile=token_tile,
            )
            if is_value
            else layout.k_outlier_count_offset(
                kv_head=kv_head,
                dim_tile=dim_tile,
                token_tile=token_tile,
            )
        )
        outlier_count = (
            _load_u16_bytes(cache, physical_block, count_offset)
            if isinstance(layout, ByteV2PageLayoutV5)
            else int(cache[physical_block, count_offset])
        )
        pool_entry_index = 0
        if isinstance(layout, ByteV2PageLayoutV5):
            pool_index_offset = (
                layout.v_outlier_pool_index_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
                if is_value
                else layout.k_outlier_pool_index_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
            )
            pool_entry_index = _load_u16_bytes(cache, physical_block, pool_index_offset)
        outlier_payload_offset = (
            layout.v_outlier_payload_offset(
                kv_head=kv_head,
                dim_tile=dim_tile,
                token_tile=token_tile,
                **(
                    {"pool_entry_index": pool_entry_index}
                    if isinstance(layout, ByteV2PageLayoutV5)
                    else {}
                ),
            )
            if is_value
            else layout.k_outlier_payload_offset(
                kv_head=kv_head,
                dim_tile=dim_tile,
                token_tile=token_tile,
                **(
                    {"pool_entry_index": pool_entry_index}
                    if isinstance(layout, ByteV2PageLayoutV5)
                    else {}
                ),
            )
        )
        outlier_policy = layout.outlier_entry_policy
        if outlier_count == layout.outlier_entries_per_tile - 1 or (
            outlier_count > elem_idx
        ):
            entry = _load_u16_bytes(
                cache,
                physical_block,
                outlier_payload_offset + elem_idx * layout.outlier_entry_bytes,
            )
            if outlier_policy.decode_elem_index(entry) == elem_idx:
                high = outlier_policy.decode_value_bits(entry)
                return _float_from_bf16_bits((high << 8) | low)
        for entry_idx in range(outlier_count):
            entry = _load_u16_bytes(
                cache,
                physical_block,
                outlier_payload_offset + entry_idx * layout.outlier_entry_bytes,
            )
            if outlier_policy.decode_elem_index(entry) == elem_idx:
                high = outlier_policy.decode_value_bits(entry)
                break
    return _float_from_bf16_bits((high << 8) | low)


def _assert_byte_v2_caches_decode_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    layout: ByteV2PageLayoutV5,
    *,
    num_tokens: int,
) -> None:
    actual = actual.cpu()
    expected = expected.cpu()
    block_size = layout.tile_policy.alloc_block_tokens
    for token_idx in range(num_tokens):
        physical_block, row = divmod(token_idx, block_size)
        for kv_head in range(layout.num_kv_heads):
            for dim in range(layout.tile_policy.head_dim):
                for is_value in (False, True):
                    actual_value = _decode_current_byte_v2_payload(
                        actual,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=is_value,
                    )
                    expected_value = _decode_current_byte_v2_payload(
                        expected,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=is_value,
                    )
                    assert actual_value == expected_value


def _decode_sideband_high_byte_v2_payload(
    cache: torch.Tensor,
    layout: ByteV2PageLayoutV4 | ByteV2PageLayoutV5,
    *,
    physical_block: int,
    kv_head: int,
    row: int,
    dim: int,
    is_value: bool,
) -> float:
    policy = layout.tile_policy
    dim_tile = dim // policy.codec_dim_block
    dim_in_tile = dim % policy.codec_dim_block
    token_tile = row // policy.codec_token_block
    row_in_tile = row % policy.codec_token_block
    tile_offset = (
        layout.v_payload_offset(
            kv_head=kv_head,
            dim_tile=dim_tile,
            token_tile=token_tile,
        )
        if is_value
        else layout.k_payload_offset(
            kv_head=kv_head,
            dim_tile=dim_tile,
            token_tile=token_tile,
        )
    )
    tile_index = (
        token_tile * policy.v_dim_tiles + dim_tile
        if is_value
        else dim_tile * policy.codec_token_tiles_per_alloc_block + token_tile
    )
    elem_idx = row_in_tile * policy.codec_dim_block + dim_in_tile
    low = int(cache[physical_block, tile_offset + elem_idx])
    outlier_mask = _load_u32_bytes(
        cache,
        physical_block,
        (
            layout.v_outlier_mask_offset(kv_head=kv_head)
            if is_value
            else layout.k_outlier_mask_offset(kv_head=kv_head)
        ),
    )
    if outlier_mask & (1 << tile_index):
        if isinstance(layout, ByteV2PageLayoutV5):
            outlier_payload_offset = _v5_outlier_payload_offset(
                cache,
                layout,
                physical_block=physical_block,
                kv_head=kv_head,
                dim_tile=dim_tile,
                is_value=is_value,
            )
        else:
            outlier_payload_offset = (
                layout.v_outlier_payload_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
                if is_value
                else layout.k_outlier_payload_offset(
                    kv_head=kv_head,
                    dim_tile=dim_tile,
                    token_tile=token_tile,
                )
            )
        high = int(cache[physical_block, outlier_payload_offset + elem_idx])
        return _float_from_bf16_bits((high << 8) | low)
    return _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=physical_block,
        kv_head=kv_head,
        row=row,
        dim=dim,
        is_value=is_value,
    )


def _reference_current_byte_v2_decode(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    layout: ByteV2PageLayoutV4,
) -> torch.Tensor:
    query = query.cpu().float()
    block_tables = block_tables.cpu()
    seq_lens = seq_lens.cpu()
    batch, num_heads, head_dim = query.shape
    q_per_kv = num_heads // num_kv_heads
    output = torch.empty_like(query)

    for seq_idx in range(batch):
        seq_len = int(seq_lens[seq_idx])
        for head_idx in range(num_heads):
            kv_head = head_idx // q_per_kv
            keys = torch.empty((seq_len, head_dim), dtype=torch.float32)
            values = torch.empty((seq_len, head_dim), dtype=torch.float32)
            for token_idx in range(seq_len):
                block_idx = token_idx // layout.tile_policy.alloc_block_tokens
                row = token_idx % layout.tile_policy.alloc_block_tokens
                physical_block = int(block_tables[seq_idx, block_idx])
                for dim in range(head_dim):
                    keys[token_idx, dim] = _decode_current_byte_v2_payload(
                        cache,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=False,
                    )
                    values[token_idx, dim] = _decode_current_byte_v2_payload(
                        cache,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=True,
                    )
            scores = torch.matmul(keys, query[seq_idx, head_idx]) * scale
            probs = torch.softmax(scores, dim=0)
            output[seq_idx, head_idx] = torch.matmul(probs, values)
    return output.to(torch.bfloat16).float()


def _assert_byte_v2_cache_decodes_tokens(
    cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    layout: ByteV2PageLayoutV4,
) -> None:
    cache_cpu = cache.cpu()
    key_cpu = key.cpu()
    value_cpu = value.cpu()
    slot_mapping_cpu = slot_mapping.cpu().tolist()
    block_size = layout.tile_policy.alloc_block_tokens

    for token_idx, slot_idx in enumerate(slot_mapping_cpu):
        if slot_idx < 0:
            continue
        physical_block = int(slot_idx) // block_size
        row = int(slot_idx) % block_size
        for kv_head in range(key_cpu.shape[1]):
            for dim in range(key_cpu.shape[2]):
                expected_key = key_cpu[token_idx, kv_head, dim].float().item()
                expected_value = value_cpu[token_idx, kv_head, dim].float().item()
                actual_key = _decode_current_byte_v2_payload(
                    cache_cpu,
                    layout,
                    physical_block=physical_block,
                    kv_head=kv_head,
                    row=row,
                    dim=dim,
                    is_value=False,
                )
                actual_value = _decode_current_byte_v2_payload(
                    cache_cpu,
                    layout,
                    physical_block=physical_block,
                    kv_head=kv_head,
                    row=row,
                    dim=dim,
                    is_value=True,
                )
                assert actual_key == expected_key
                assert actual_value == expected_value


def _reference_raw_paged_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
) -> torch.Tensor:
    query = query.cpu().float()
    key = key.cpu().float()
    value = value.cpu().float()
    seq_lens = seq_lens.cpu()
    batch, num_heads, _ = query.shape
    q_per_kv = num_heads // num_kv_heads
    output = torch.empty_like(query)

    token_cursor = 0
    for seq_idx in range(batch):
        seq_len = int(seq_lens[seq_idx])
        seq_key = key[token_cursor : token_cursor + seq_len]
        seq_value = value[token_cursor : token_cursor + seq_len]
        token_cursor += seq_len
        for head_idx in range(num_heads):
            kv_head = head_idx // q_per_kv
            scores = torch.matmul(seq_key[:, kv_head], query[seq_idx, head_idx])
            probs = torch.softmax(scores * scale, dim=0)
            output[seq_idx, head_idx] = torch.matmul(probs, seq_value[:, kv_head])
    return output.to(torch.bfloat16).float()


def _attention_diff_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _reference_byte_v2_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float,
    num_kv_heads: int,
    causal: bool,
) -> torch.Tensor:
    query = query.cpu().float()
    key = key.cpu().float()
    value = value.cpu().float()
    query_start_locs = query_start_loc.cpu().tolist()
    _, num_heads, _ = query.shape
    q_per_kv = num_heads // num_kv_heads
    output = torch.empty_like(query)

    for start, end in zip(query_start_locs[:-1], query_start_locs[1:]):
        start = int(start)
        end = int(end)
        for head_idx in range(num_heads):
            kv_head = head_idx // q_per_kv
            q = query[start:end, head_idx]
            k = key[start:end, kv_head]
            v = value[start:end, kv_head]
            scores = torch.matmul(q, k.transpose(0, 1)) * scale
            if causal:
                q_len = end - start
                causal_mask = torch.triu(
                    torch.ones((q_len, q_len), dtype=torch.bool),
                    diagonal=1,
                )
                scores.masked_fill_(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            output[start:end, head_idx] = torch.matmul(probs, v)
    return output.to(torch.bfloat16).float()


def test_default_tile_policy_matches_phase1_defaults():
    policy = DEFAULT_BYTE_V2_TILE_POLICY

    assert policy.codec_token_block == 16
    assert policy.codec_dim_block == 16
    assert policy.alloc_block_tokens == 16
    assert policy.compute_block_n == 64
    assert policy.head_dim == 128
    assert policy.head_dim_v == 128

    assert policy.codec_tile_elems == 256
    assert policy.codec_packed_elems == 128
    assert policy.k_dim_tiles == 8
    assert policy.v_dim_tiles == 8
    assert policy.codec_token_tiles_per_alloc_block == 1
    assert policy.alloc_blocks_per_compute_tile == 4
    assert policy.codec_token_tiles_per_compute_tile == 4
    assert policy.codec_tiles_per_k_page == 8
    assert policy.codec_tiles_per_v_page == 8


def test_tile_policy_can_be_modified_without_changing_allocator_granularity():
    policy = DEFAULT_BYTE_V2_TILE_POLICY.with_updates(
        codec_dim_block=32,
        compute_block_n=128,
    )

    assert policy.codec_token_block == 16
    assert policy.alloc_block_tokens == 16
    assert policy.codec_dim_block == 32
    assert policy.compute_block_n == 128
    assert policy.codec_tile_elems == 512
    assert policy.k_dim_tiles == 4
    assert policy.v_dim_tiles == 4
    assert policy.alloc_blocks_per_compute_tile == 8


def test_tile_policy_from_env_parameterizes_compute_block_n(monkeypatch):
    monkeypatch.setenv("BYTE_V2_COMPUTE_BLOCK_N", "128")

    policy = byte_v2_tile_policy_from_env()

    assert policy.compute_block_n == 128
    assert policy.codec_token_block == DEFAULT_BYTE_V2_TILE_POLICY.codec_token_block
    assert policy.alloc_block_tokens == DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"codec_token_block": 0}, "codec_token_block must be positive"),
        (
            {"alloc_block_tokens": 24},
            "alloc_block_tokens must be divisible by codec_token_block",
        ),
        ({"head_dim": 120}, "head_dim must be divisible by codec_dim_block"),
        ({"head_dim_v": 120}, "head_dim_v must be divisible by codec_dim_block"),
        (
            {"compute_block_n": 72},
            "compute_block_n must be divisible by alloc_block_tokens",
        ),
    ],
)
def test_tile_policy_rejects_invalid_shapes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ByteV2TilePolicy(**kwargs)


def test_payload_policy_derives_codec_tile_bytes():
    policy = DEFAULT_BYTE_V2_TILE_POLICY
    payload = ByteV2CodecPayloadPolicy()

    assert payload.low_bytes_per_codec_tile(policy) == 256
    assert payload.code_bytes_per_codec_tile(policy) == 128
    assert payload.bytes_per_codec_tile(policy) == 384


def test_outlier_entry_policy_tracks_codec_tile_size():
    default_outlier = ByteV2OutlierEntryPolicy.from_tile_policy()
    wider_dim_policy = DEFAULT_BYTE_V2_TILE_POLICY.with_updates(codec_dim_block=32)
    wider_outlier = ByteV2OutlierEntryPolicy.from_tile_policy(wider_dim_policy)

    assert default_outlier.elem_bits == 8
    assert default_outlier.max_elem_index == 255
    assert default_outlier.supports_tile_policy(DEFAULT_BYTE_V2_TILE_POLICY)

    assert wider_outlier.elem_bits == 9
    assert wider_outlier.max_elem_index == 511
    assert wider_outlier.supports_tile_policy(wider_dim_policy)
    assert not default_outlier.supports_tile_policy(wider_dim_policy)


def test_outlier_entry_policy_encodes_raw_value_bits():
    default_outlier = ByteV2OutlierEntryPolicy.from_tile_policy()
    wider_outlier = ByteV2OutlierEntryPolicy.from_tile_policy(
        DEFAULT_BYTE_V2_TILE_POLICY.with_updates(codec_dim_block=32)
    )

    entry = default_outlier.encode(elem_index=255, value_bits=0xBEEF)

    assert default_outlier.decode(entry) == (255, 0xBEEF)
    with pytest.raises(ValueError, match="elem_index must fit"):
        default_outlier.encode(elem_index=256, value_bits=0)
    assert wider_outlier.decode(wider_outlier.encode(511, 0x1234)) == (
        511,
        0x1234,
    )


def test_outlier_entry_policy_can_estimate_high_byte_overlay():
    high_byte_outlier = ByteV2OutlierEntryPolicy.from_tile_policy(value_bits=8)

    entry = high_byte_outlier.encode(elem_index=255, value_bits=0x7F)

    assert high_byte_outlier.entry_bits == 16
    assert high_byte_outlier.entry_bytes == 2
    assert high_byte_outlier.decode(entry) == (255, 0x7F)


def test_page_layout_v4_derives_sizes_from_policy():
    layout = ByteV2PageLayoutV4()

    assert layout.include_raw_payload is False
    assert layout.macro_pages == 4
    assert layout.metadata_bytes == 640
    assert layout.aligned_metadata_bytes == 640
    assert layout.codec_payload_bytes_per_tile == 384
    assert layout.k_payload_bytes_per_kv_head == 3072
    assert layout.v_payload_bytes_per_kv_head == 3072
    assert layout.compressed_payload_bytes == 49152
    assert layout.outlier_entry_bytes == 2
    assert layout.outlier_payload_bytes_per_tile == 512
    assert layout.outlier_payload_bytes == 65536
    assert layout.raw_payload_bytes == 0
    assert layout.payload_bytes == 114688
    assert layout.page_size_bytes == 115328


def test_page_layout_v4_derives_payload_offsets():
    layout = ByteV2PageLayoutV4()

    assert layout.k_payload_base_bytes == 640
    assert layout.v_payload_base_bytes == 25216
    assert layout.k_outlier_payload_base_bytes == 49792
    assert layout.v_outlier_payload_base_bytes == 82560
    assert layout.raw_k_payload_base_bytes == 115328
    assert layout.raw_v_payload_base_bytes == 115328
    assert layout.k_fallback_mask_offset(kv_head=0) == 128
    assert layout.v_fallback_mask_offset(kv_head=0) == 132
    assert layout.k_outlier_mask_offset(kv_head=0) == 152
    assert layout.v_outlier_mask_offset(kv_head=0) == 156
    assert layout.k_base_offset(kv_head=0, dim_tile=0) == 136
    assert layout.v_base_offset(kv_head=0, dim_tile=0) == 144
    assert layout.k_outlier_count_offset(kv_head=0, dim_tile=0) == 160
    assert layout.v_outlier_count_offset(kv_head=0, dim_tile=0) == 168
    assert layout.k_payload_offset(kv_head=0, dim_tile=0) == 640
    assert layout.k_payload_offset(kv_head=7, dim_tile=7) == 24832
    assert layout.v_payload_offset(kv_head=0, dim_tile=0) == 25216
    assert layout.v_payload_offset(kv_head=7, dim_tile=7) == 49408
    assert layout.k_outlier_payload_offset(kv_head=0, dim_tile=0) == 49792
    assert (
        layout.k_outlier_payload_offset(kv_head=7, dim_tile=7, entry_idx=255) == 82558
    )
    assert layout.v_outlier_payload_offset(kv_head=0, dim_tile=0) == 82560
    assert (
        layout.v_outlier_payload_offset(kv_head=7, dim_tile=7, entry_idx=255) == 115326
    )
    with pytest.raises(ValueError, match="raw payload is disabled"):
        layout.raw_key_offset(kv_head=0, row=0, dim=0)
    with pytest.raises(ValueError, match="raw payload is disabled"):
        layout.raw_value_offset(kv_head=0, row=0, dim=0)


def test_page_layout_v5_uses_compact_page_outlier_pool():
    layout = ByteV2PageLayoutV5()

    assert layout.metadata_bytes == 896
    assert layout.aligned_metadata_bytes == 896
    assert layout.compressed_payload_bytes == 49152
    assert layout.outlier_pool_entries == 1024
    assert layout.outlier_pool_bytes == 2048
    assert layout.k_payload_base_bytes == 896
    assert layout.v_payload_base_bytes == 25472
    assert layout.outlier_pool_base_bytes == 50048
    assert layout.page_size_bytes == 52096
    assert layout.page_size_bytes < 2 * 16 * 8 * 128 * 2


def test_page_layout_v5_derives_tile_descriptors():
    layout = ByteV2PageLayoutV5()

    assert layout.k_outlier_count_offset(kv_head=0, dim_tile=0) == 160
    assert layout.v_outlier_count_offset(kv_head=0, dim_tile=0) == 176
    assert layout.k_outlier_pool_index_offset(kv_head=0, dim_tile=0) == 192
    assert layout.v_outlier_pool_index_offset(kv_head=0, dim_tile=0) == 208
    assert (
        layout.k_outlier_payload_offset(
            kv_head=0,
            dim_tile=0,
            pool_entry_index=17,
            entry_idx=3,
        )
        == 50088
    )


def test_page_layout_v4_derives_raw_enabled_offsets():
    layout = _raw_payload_layout()

    assert layout.include_raw_payload is True
    assert layout.outlier_entries_per_tile == 128
    assert layout.page_size_bytes == 148096
    assert (
        layout.k_outlier_payload_offset(kv_head=7, dim_tile=7, entry_idx=127) == 66174
    )
    assert layout.v_outlier_payload_offset(kv_head=0, dim_tile=0) == 66176
    assert (
        layout.v_outlier_payload_offset(kv_head=7, dim_tile=7, entry_idx=127) == 82558
    )
    assert layout.raw_key_offset(kv_head=0, row=0, dim=0) == 82560
    assert layout.raw_key_offset(kv_head=7, row=15, dim=127) == 115326
    assert layout.raw_value_offset(kv_head=0, row=0, dim=0) == 115328
    assert layout.raw_value_offset(kv_head=7, row=15, dim=127) == 148094


def test_page_layout_v4_requires_full_tile_overlay_without_raw_payload():
    with pytest.raises(ValueError, match="full codec tile"):
        ByteV2PageLayoutV4(outlier_entries_per_tile=128)


def test_page_layout_v4_uses_modified_compute_tile():
    layout = ByteV2PageLayoutV4(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY.with_updates(compute_block_n=128)
    )

    assert layout.macro_pages == 8
    assert layout.page_size_bytes == ByteV2PageLayoutV4().page_size_bytes


def test_byte_v2_fallback_stats_estimates_overlay_bytes_from_page():
    layout = _raw_payload_layout()
    page = bytearray(layout.page_size_bytes)
    k_fallback_mask_offset = layout.k_fallback_mask_offset(kv_head=0)
    page[k_fallback_mask_offset : k_fallback_mask_offset + 4] = (1).to_bytes(
        4,
        "little",
    )

    elem_idx = 0
    for row in range(layout.tile_policy.codec_token_block):
        for dim in range(layout.tile_policy.codec_dim_block):
            high_byte = 0 if elem_idx < 128 else 32
            offset = layout.raw_key_offset(kv_head=0, row=row, dim=dim)
            page[offset] = 0
            page[offset + 1] = high_byte
            elem_idx += 1

    stats = collect_byte_v2_fallback_stats([page], layout=layout)

    assert stats.total_tiles == 128
    assert stats.fallback_tiles == 1
    assert stats.outlier_entries == 128
    assert stats.raw_tile_bytes == 512
    assert stats.estimated_overlay_bytes == 256
    assert stats.overlay_tiles == 0
    assert stats.overlay_entries == 0
    assert stats.overlay_bytes == 0
    assert stats.fallback_ratio == pytest.approx(1 / 128)
    assert stats.outlier_entries_per_fallback_tile == 128
    assert stats.overlay_to_raw_tile_bytes_ratio == 0.5


def test_byte_v2_fallback_stats_rejects_too_narrow_outlier_policy():
    layout = ByteV2PageLayoutV4()
    page = bytearray(layout.page_size_bytes)
    outlier_policy = ByteV2OutlierEntryPolicy(elem_bits=7, value_bits=8)

    with pytest.raises(ValueError, match="outlier_policy must support"):
        collect_byte_v2_fallback_stats(
            [page],
            layout=layout,
            outlier_policy=outlier_policy,
        )


def test_byte_v2_reference_decode_applies_outlier_overlay():
    layout = ByteV2PageLayoutV4()
    page = bytearray(layout.page_size_bytes)
    page[layout.k_base_offset(kv_head=0, dim_tile=0)] = 0
    k_outlier_mask_offset = layout.k_outlier_mask_offset(kv_head=0)
    page[k_outlier_mask_offset : k_outlier_mask_offset + 4] = (1).to_bytes(
        4,
        "little",
    )
    page[layout.k_outlier_count_offset(kv_head=0, dim_tile=0)] = 1

    elem_idx = 0
    payload_offset = layout.k_payload_offset(kv_head=0, dim_tile=0)
    page[payload_offset + elem_idx] = 0x55
    entry = layout.outlier_entry_policy.encode(elem_index=elem_idx, value_bits=32)
    overlay_offset = layout.k_outlier_payload_offset(kv_head=0, dim_tile=0)
    page[overlay_offset : overlay_offset + 2] = entry.to_bytes(2, "little")

    cache = torch.tensor([list(page)], dtype=torch.uint8)

    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=0,
        dim=0,
        is_value=False,
    ) == pytest.approx(_float_from_bf16_bits((32 << 8) | 0x55))


def test_raw_staging_layout_derives_sizes_and_offsets():
    layout = ByteV2RawStagingLayout()

    assert layout.key_bytes == 32768
    assert layout.value_bytes == 32768
    assert layout.aligned_key_bytes == 32768
    assert layout.value_base_bytes == 32768
    assert layout.slot_size_bytes == 65536
    assert layout.key_offset(kv_head=0, row=0, dim=0) == 0
    assert layout.key_offset(kv_head=7, row=15, dim=127) == 32766
    assert layout.value_offset(kv_head=0, row=0, dim=0) == 32768
    assert layout.value_offset(kv_head=7, row=15, dim=127) == 65534


def test_paged_kv_manager_builds_fixed_width_macro_descriptor():
    layout = ByteV2PageLayoutV4()
    manager = ByteV2PagedKVManager(layout=layout)

    desc = manager.build_macro_descriptor(
        [10, 11, 12, 13, 14],
        first_block_idx=1,
        seq_len=55,
        kv_head=2,
        outlier_page_mask=0b100,
    )

    assert desc.active_pages == 4
    assert len(desc.physical_blocks) == BYTE_V2_MAX_MACRO_PAGES
    assert desc.descriptor_bytes == BYTE_V2_MACRO_DESCRIPTOR_BYTES
    assert desc.physical_blocks == (11, 12, 13, -1, -1, -1, -1, -1)
    assert desc.valid_rows == (16, 16, 7, 0, 0, 0, 0, 0)
    assert desc.compressed_mask == 0b111
    assert desc.outlier_page_mask == 0b100
    assert desc.k_payload_offsets[:4] == (
        layout.k_payload_offset(kv_head=2, dim_tile=0),
    ) * 3 + (0,)
    assert desc.v_payload_offsets[:4] == (
        layout.v_payload_offset(kv_head=2, dim_tile=0),
    ) * 3 + (0,)


def test_paged_kv_manager_rejects_invalid_descriptor_masks():
    manager = ByteV2PagedKVManager()

    with pytest.raises(ValueError, match="cannot include invalid pages"):
        manager.build_macro_descriptor(
            [0],
            first_block_idx=0,
            seq_len=16,
            kv_head=0,
            outlier_page_mask=0b10,
        )


def test_byte_v2_backend_is_registered_but_kernel_gated(monkeypatch):
    backend_cls = AttentionBackendEnum.BYTE_V2.get_class()

    assert backend_cls.get_name() == "BYTE_V2"
    assert backend_cls.get_kv_cache_shape(
        num_blocks=3,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
    ) == (3, ByteV2PageLayoutV5().page_size_bytes)

    invalid_reasons = backend_cls.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type="decoder",
    )

    if byte_v2_custom_ops_are_available():
        assert invalid_reasons == []
    else:
        assert invalid_reasons == ["ByteV2 native CUDA kernels are not registered yet"]
        assert missing_byte_v2_custom_ops()

    monkeypatch.setattr(
        byte_v2_attn_module, "byte_v2_custom_ops_are_available", lambda: True
    )

    assert (
        backend_cls.validate_configuration(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(9, 0),
            attn_type="decoder",
        )
        == []
    )


def test_byte_v2_metadata_builder_supports_single_token_decode_cudagraph():
    support = byte_v2_attn_module.ByteV2AttentionMetadataBuilder.get_cudagraph_support(
        vllm_config=None,
        kv_cache_spec=None,
    )

    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_byte_v2_decode_raw_fallback_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_RAW_FALLBACK", raising=False)
    assert byte_v2_attn_module._decode_raw_fallback_enabled() is False

    monkeypatch.setenv("BYTE_V2_DECODE_RAW_FALLBACK", "1")
    assert byte_v2_attn_module._decode_raw_fallback_enabled() is True


def test_byte_v2_test_forced_raw_promotion_defaults_off(monkeypatch):
    env_name = "BYTE_V2_TEST_FORCE_RAW_PROMOTION"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module.byte_v2_test_forced_raw_promotion_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert byte_v2_attn_module.byte_v2_test_forced_raw_promotion_enabled() is True


def test_byte_v2_raw_fallback_store_only_allocates_test_diagnostic_when_enabled():
    raw_layout = ByteV2RawStagingLayout()
    kv_cache = torch.empty((2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    default_store = byte_v2_attn_module.ByteV2RawFallbackStore(raw_layout=raw_layout)
    diagnostic_store = byte_v2_attn_module.ByteV2RawFallbackStore(
        raw_layout=raw_layout,
        forced_raw_diagnostic=True,
    )

    assert default_store.state(kv_cache).forced_raw_diagnostic is None
    diagnostic = diagnostic_store.state(kv_cache.clone()).forced_raw_diagnostic
    assert diagnostic is not None
    assert diagnostic.dtype == torch.int32
    assert diagnostic.tolist() == [0, 0, 0]


def test_byte_v2_test_forced_raw_promotion_requires_hybrid(monkeypatch):
    monkeypatch.delenv("BYTE_V2_FA2_HYBRID_RAW_FALLBACK", raising=False)
    monkeypatch.setenv("BYTE_V2_TEST_FORCE_RAW_PROMOTION", "1")

    with pytest.raises(RuntimeError, match="requires.*HYBRID_RAW_FALLBACK"):
        byte_v2_attn_module.ByteV2AttentionImpl(
            num_heads=32,
            head_size=128,
            scale=0.125,
            num_kv_heads=8,
        )


def test_byte_v2_decode_kernel_mode_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_KERNEL", raising=False)

    assert byte_v2_attn_module._decode_kernel_mode() == "legacy"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("legacy", "legacy"),
        ("fa2", "fa2"),
        ("auto", "auto"),
        ("FA2", "fa2"),
    ],
)
def test_byte_v2_decode_kernel_mode_accepts_valid_values(
    monkeypatch,
    value,
    expected,
):
    monkeypatch.setenv("BYTE_V2_DECODE_KERNEL", value)

    assert byte_v2_attn_module._decode_kernel_mode() == expected


@pytest.mark.parametrize("value", ["", "raw", "fa3"])
def test_byte_v2_decode_kernel_mode_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("BYTE_V2_DECODE_KERNEL", value)

    assert byte_v2_attn_module._decode_kernel_mode() == "legacy"


def test_byte_v2_explicit_fa2_decode_fails_closed_when_op_is_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_DECODE_KERNEL", "fa2")
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_fa2_decode_is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="FA2 extension op is unavailable"):
        byte_v2_attn_module.ByteV2AttentionImpl(
            num_heads=32,
            head_size=128,
            scale=0.125,
            num_kv_heads=8,
        )


def test_byte_v2_fa2_decode_rejects_non_q1_metadata(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_KERNEL", "fa2")
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_fa2_decode_is_available",
        lambda: True,
    )
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    reason = impl._fa2_decode_incompatibility(
        torch.empty((2, 32, 128), dtype=torch.bfloat16),
        torch.empty((1,), dtype=torch.uint8),
        torch.empty((2, 32, 128), dtype=torch.bfloat16),
        SimpleNamespace(max_query_len=2),
    )

    assert reason == "only Q1 decode is supported"


def test_byte_v2_forward_prefers_fa2_decode_and_slices_metadata(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_KERNEL", "fa2")
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_fa2_decode_is_available",
        lambda: True,
    )
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    monkeypatch.setattr(
        impl,
        "_fa2_decode_incompatibility",
        lambda *args: None,
    )

    captured: dict[str, Any] = {}

    def fake_fa2_decode(
        output_arg,
        query_arg,
        kv_cache_arg,
        query_start_locs_arg,
        block_tables_arg,
        seq_lens_arg,
        *,
        scale,
        max_seq_len,
        causal,
    ):
        captured.update(
            output=output_arg,
            query=query_arg,
            kv_cache=kv_cache_arg,
            query_start_locs=query_start_locs_arg,
            block_tables=block_tables_arg,
            seq_lens=seq_lens_arg,
            scale=scale,
            max_seq_len=max_seq_len,
            causal=causal,
        )
        output_arg.fill_(7)

    def fail_legacy_decode(*args, **kwargs):
        del args, kwargs
        pytest.fail("legacy decode must not run after FA2 handles the request")

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_fa2_paged_decode_attention",
        fake_fa2_decode,
    )
    monkeypatch.setattr(impl, "_run_paged_decode", fail_legacy_decode)

    batch_size = 2
    query = torch.empty((batch_size, 32, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    kv_cache = torch.empty(
        (4, ByteV2PageLayoutV5().page_size_bytes),
        dtype=torch.uint8,
    )
    query_start_locs = torch.tensor([0, 1, 2], dtype=torch.int32)
    block_table = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([33, 65, 97], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        max_query_len=1,
        query_start_loc=query_start_locs,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=65,
        causal=True,
    )

    result = impl.forward(
        None,
        query,
        torch.empty((0,), dtype=torch.bfloat16),
        torch.empty((0,), dtype=torch.bfloat16),
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert captured["output"] is output
    assert captured["query"] is query
    assert captured["kv_cache"] is kv_cache
    assert captured["query_start_locs"] is query_start_locs
    torch.testing.assert_close(captured["block_tables"], block_table[:batch_size])
    torch.testing.assert_close(captured["seq_lens"], seq_lens[:batch_size])
    assert tuple(captured["block_tables"].shape) == (batch_size, 3)
    assert tuple(captured["seq_lens"].shape) == (batch_size,)
    assert captured["scale"] == 0.125
    assert captured["max_seq_len"] == 65
    assert captured["causal"] is True
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_byte_v2_decode_no_outlier_fast_path_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", raising=False)
    assert byte_v2_attn_module._decode_assume_no_outlier_enabled() is False

    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    assert byte_v2_attn_module._decode_assume_no_outlier_enabled() is True


def test_byte_v2_decode_gqa_packed_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED", raising=False)
    assert byte_v2_attn_module._decode_gqa_packed_enabled() is False

    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    assert byte_v2_attn_module._decode_gqa_packed_enabled() is True


def test_byte_v2_decode_unnormalized_partition_output_defaults_off(monkeypatch):
    env_name = "BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._decode_unnormalized_partition_output_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert byte_v2_attn_module._decode_unnormalized_partition_output_enabled() is True


def test_byte_v2_decode_validate_no_outlier_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_VALIDATE_NO_OUTLIER", raising=False)
    assert byte_v2_attn_module._decode_validate_no_outlier_enabled() is False

    monkeypatch.setenv("BYTE_V2_DECODE_VALIDATE_NO_OUTLIER", "1")
    assert byte_v2_attn_module._decode_validate_no_outlier_enabled() is True


def test_byte_v2_decode_page_unsafe_flags_defaults_on(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_PAGE_UNSAFE_FLAGS", raising=False)
    assert byte_v2_attn_module._decode_page_unsafe_flags_enabled() is True

    monkeypatch.setenv("BYTE_V2_DECODE_PAGE_UNSAFE_FLAGS", "0")
    assert byte_v2_attn_module._decode_page_unsafe_flags_enabled() is False


def test_byte_v2_fused_staging_release_flags_defaults_on(monkeypatch):
    env_name = "BYTE_V2_FUSED_STAGING_RELEASE_FLAGS"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._fused_staging_release_flags_enabled() is True

    monkeypatch.setenv(env_name, "0")
    assert byte_v2_attn_module._fused_staging_release_flags_enabled() is False


def test_byte_v2_native_raw_staging_update_defaults_on(monkeypatch):
    env_name = "BYTE_V2_NATIVE_RAW_STAGING_UPDATE"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._native_raw_staging_update_enabled() is True

    monkeypatch.setenv(env_name, "0")
    assert byte_v2_attn_module._native_raw_staging_update_enabled() is False


def test_byte_v2_native_single_token_update_defaults_off(monkeypatch):
    env_name = "BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._native_single_token_update_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert byte_v2_attn_module._native_single_token_update_enabled() is True


def test_byte_v2_fused_single_token_staging_defaults_on(monkeypatch):
    env_name = "BYTE_V2_FUSED_SINGLE_TOKEN_STAGING"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._fused_single_token_staging_enabled() is True

    monkeypatch.setenv(env_name, "0")
    assert byte_v2_attn_module._fused_single_token_staging_enabled() is False


def test_byte_v2_fused_single_token_commit_release_defaults_off(monkeypatch):
    env_name = "BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._fused_single_token_commit_release_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert byte_v2_attn_module._fused_single_token_commit_release_enabled() is True


def test_byte_v2_fused_single_token_stage_metadata_clear_defaults_off(monkeypatch):
    env_name = "BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR"
    monkeypatch.delenv(env_name, raising=False)
    assert (
        byte_v2_attn_module._fused_single_token_stage_metadata_clear_enabled() is False
    )

    monkeypatch.setenv(env_name, "1")
    assert (
        byte_v2_attn_module._fused_single_token_stage_metadata_clear_enabled() is True
    )


def test_byte_v2_fused_commit_metadata_clear_defaults_on(monkeypatch):
    env_name = "BYTE_V2_FUSED_COMMIT_METADATA_CLEAR"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._fused_commit_metadata_clear_enabled() is True

    monkeypatch.setenv(env_name, "0")
    assert byte_v2_attn_module._fused_commit_metadata_clear_enabled() is False


def test_byte_v2_warp_parallel_commit_histogram_defaults_on(monkeypatch):
    env_name = "BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._warp_parallel_commit_histogram_enabled() is True

    monkeypatch.setenv(env_name, "0")
    assert byte_v2_attn_module._warp_parallel_commit_histogram_enabled() is False


def test_byte_v2_speculative_verify_q4_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_SPECULATIVE_VERIFY_Q4", raising=False)
    assert byte_v2_attn_module._speculative_verify_q4_enabled() is False

    monkeypatch.setenv("BYTE_V2_SPECULATIVE_VERIFY_Q4", "1")
    assert byte_v2_attn_module._speculative_verify_q4_enabled() is True


def test_byte_v2_speculative_verify_gqa_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_SPECULATIVE_VERIFY_GQA", raising=False)
    assert byte_v2_attn_module._speculative_verify_gqa_enabled() is False

    monkeypatch.setenv("BYTE_V2_SPECULATIVE_VERIFY_GQA", "1")
    assert byte_v2_attn_module._speculative_verify_gqa_enabled() is True


def test_byte_v2_speculative_verify_ragged_q4_defaults_off(monkeypatch):
    env_name = "BYTE_V2_SPECULATIVE_VERIFY_RAGGED_Q4"
    monkeypatch.delenv(env_name, raising=False)
    assert byte_v2_attn_module._speculative_verify_ragged_q4_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert byte_v2_attn_module._speculative_verify_ragged_q4_enabled() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_speculative_verify_ragged_q4_accepts_compiled_padding(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_SPECULATIVE_VERIFY_RAGGED_Q4", "1")
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    query_start_locs_cpu = torch.tensor([0, 1, 5, 6, 7], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=7,
        max_query_len=4,
        query_start_loc=query_start_locs_cpu.cuda(),
        query_start_loc_cpu=query_start_locs_cpu,
        seq_lens=torch.full((4,), 64, dtype=torch.int32, device="cuda"),
        seq_lens_cpu=torch.full((4,), 64, dtype=torch.int32),
        block_table=torch.zeros((4, 4), dtype=torch.int32, device="cuda"),
        causal=True,
    )
    query = torch.empty((8, 32, 128), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    kv_cache = torch.empty((1,), dtype=torch.uint8, device="cuda")

    assert (
        impl._speculative_ragged_q4_num_requests(
            query,
            kv_cache,
            output,
            attn_metadata,
        )
        == 4
    )
    attn_metadata.seq_lens_cpu = None
    assert (
        impl._speculative_ragged_q4_num_requests(
            query,
            kv_cache,
            output,
            attn_metadata,
        )
        is None
    )


def test_byte_v2_cached_prefix_q16_defaults_off(monkeypatch):
    monkeypatch.delenv("BYTE_V2_CACHED_PREFIX_Q16", raising=False)
    assert byte_v2_attn_module._cached_prefix_q16_enabled() is False

    monkeypatch.setenv("BYTE_V2_CACHED_PREFIX_Q16", "1")
    assert byte_v2_attn_module._cached_prefix_q16_enabled() is True


def test_byte_v2_hybrid_cached_prefill_uses_generic_fa2_reader(monkeypatch):
    impl = object.__new__(byte_v2_attn_module.ByteV2AttentionImpl)
    impl.scale = 0.125
    state = SimpleNamespace(
        raw_pages=torch.empty((2, 65_536), dtype=torch.uint8),
        page_to_raw_slot=torch.full((4,), -1, dtype=torch.int32),
    )
    impl.raw_fallback_store = SimpleNamespace(state=lambda _: state)
    query = torch.empty((5, 32, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    kv_cache = torch.empty((4, 52_096), dtype=torch.uint8)
    metadata = SimpleNamespace(
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([18, 35], dtype=torch.int32),
        max_query_len=3,
        max_seq_len=35,
        causal=True,
        common_prefix_len=0,
    )
    calls = []

    def fake_hybrid_fa2(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_fa2_hybrid_paged_decode_attention",
        fake_hybrid_fa2,
    )

    result = impl._forward_prefill_from_cache(query, kv_cache, output, metadata)

    assert result is output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].data_ptr() == output.data_ptr()
    assert args[1].data_ptr() == query.data_ptr()
    assert args[3] is state.raw_pages
    assert args[4] is state.page_to_raw_slot
    assert torch.equal(args[5], metadata.query_start_loc)
    assert torch.equal(args[6], metadata.block_table)
    assert torch.equal(args[7], metadata.seq_lens)
    assert kwargs == {
        "scale": 0.125,
        "max_query_len": 3,
        "max_seq_len": 35,
        "causal": True,
    }


def test_byte_v2_hybrid_cached_prefill_rejects_noncausal_metadata():
    impl = object.__new__(byte_v2_attn_module.ByteV2AttentionImpl)
    impl.raw_fallback_store = object()
    metadata = SimpleNamespace(causal=False, common_prefix_len=0)

    with pytest.raises(RuntimeError, match="require causal attention"):
        impl._forward_prefill_from_cache(
            torch.empty((2, 32, 128), dtype=torch.bfloat16),
            torch.empty((1, 52_096), dtype=torch.uint8),
            torch.empty((2, 32, 128), dtype=torch.bfloat16),
            metadata,
        )


def test_byte_v2_hybrid_fa2_wrapper_forwards_generic_query_length(monkeypatch):
    calls = []

    def fake_op(*args):
        calls.append(args)

    monkeypatch.setattr(byte_v2_ops_module, "_require_fa2_op", lambda _: fake_op)
    query = torch.empty((6, 32, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    kv_cache = torch.empty((4, 52_096), dtype=torch.uint8)
    raw_pages = torch.empty((1, 65_536), dtype=torch.uint8)
    page_map = torch.full((4,), -1, dtype=torch.int32)
    query_start_locs = torch.tensor([0, 2, 6], dtype=torch.int32)

    byte_v2_ops_module.byte_v2_fa2_hybrid_paged_decode_attention(
        output,
        query,
        kv_cache,
        raw_pages,
        page_map,
        query_start_locs,
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        torch.tensor([18, 36], dtype=torch.int32),
        scale=0.125,
        max_query_len=4,
        max_seq_len=36,
        causal=True,
    )

    assert len(calls) == 1
    assert calls[0][12] == 4
    assert calls[0][13] == 36


def test_byte_v2_raw_staging_fa2_wrapper_builds_paged_views(monkeypatch):
    calls = []

    def fake_require_fa2_op(op_name):
        assert op_name == "varlen_fwd"

        def fake_op(*args):
            calls.append(args)

        return fake_op

    monkeypatch.setattr(byte_v2_ops_module, "_require_fa2_op", fake_require_fa2_op)
    raw_staging = torch.empty((3, 65_536), dtype=torch.uint8)
    page_map = torch.tensor([2, 0, 1, -1], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    query_start_locs = torch.tensor([0, 2, 6], dtype=torch.int32)
    query = torch.empty((6, 32, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)

    byte_v2_ops_module.byte_v2_fa2_raw_staging_prefill_attention(
        output,
        query,
        raw_staging,
        page_map,
        query_start_locs,
        block_tables,
        torch.tensor([18, 36], dtype=torch.int32),
        scale=0.125,
        num_kv_heads=8,
        block_size=16,
        head_dim=128,
        max_query_len=4,
        max_seq_len=36,
        causal=True,
    )

    assert len(calls) == 1
    key_cache, value_cache = calls[0][1:3]
    assert key_cache.shape == (3, 16, 8, 128)
    assert value_cache.shape == (3, 16, 8, 128)
    assert key_cache.stride() == (32_768, 128, 2_048, 1)
    assert value_cache.stride() == (32_768, 128, 2_048, 1)
    torch.testing.assert_close(
        calls[0][8],
        torch.tensor([[2, 0], [1, 2]], dtype=torch.int32),
    )
    assert calls[0][10] == 4
    assert calls[0][11] == 36


def test_byte_v2_hybrid_q1_update_wrapper_forwards_optional_flags(monkeypatch):
    captured = {}

    def fake_require_op(namespace, op_name):
        assert namespace == "_C_cache_ops"
        assert op_name == "byte_v2_update_hybrid_cache_raw_staging_q1"

        def fake_op(*args):
            captured["args"] = args

        return fake_op

    monkeypatch.setattr(byte_v2_ops_module, "_require_op", fake_require_op)
    tensors = [torch.empty(1) for _ in range(15)]
    flags = torch.empty(1)

    byte_v2_ops_module.byte_v2_update_hybrid_cache_raw_staging_q1(
        *tensors,
        tile_policy=(16, 16, 16, 64, 128, 128),
        page_unsafe_flags=flags,
    )

    assert captured["args"][:15] == tuple(tensors)
    assert captured["args"][15] == [16, 16, 16, 64, 128, 128]
    assert captured["args"][16] is flags


def test_byte_v2_hybrid_multi_token_update_wrapper_forwards_optional_flags(
    monkeypatch,
):
    captured = {}

    def fake_require_op(namespace, op_name):
        assert namespace == "_C_cache_ops"
        assert op_name == "byte_v2_update_hybrid_cache_raw_staging_multi_token"

        def fake_op(*args):
            captured["args"] = args

        return fake_op

    monkeypatch.setattr(byte_v2_ops_module, "_require_op", fake_require_op)
    tensors = [torch.empty(1) for _ in range(15)]
    flags = torch.empty(1)

    byte_v2_ops_module.byte_v2_update_hybrid_cache_raw_staging_multi_token(
        *tensors,
        tile_policy=(16, 16, 16, 64, 128, 128),
        page_unsafe_flags=flags,
    )

    assert captured["args"][:15] == tuple(tensors)
    assert captured["args"][15] == [16, 16, 16, 64, 128, 128]
    assert captured["args"][16] is flags


def test_byte_v2_test_forced_raw_wrapper_forwards_diagnostic(monkeypatch):
    captured = {}

    def fake_require_op(namespace, op_name):
        assert namespace == "_C_cache_ops"
        assert op_name == "byte_v2_test_force_promote_raw_staging_q1"

        def fake_op(*args):
            captured["args"] = args

        return fake_op

    monkeypatch.setattr(byte_v2_ops_module, "_require_op", fake_require_op)
    tensors = [torch.empty(1) for _ in range(8)]

    byte_v2_ops_module.byte_v2_test_force_promote_raw_staging_q1(*tensors)

    assert captured["args"] == tuple(tensors)


@pytest.mark.parametrize("has_cached_context", [False, True])
def test_byte_v2_hybrid_prefill_always_uses_fa2_cache_reader(
    has_cached_context,
):
    impl = object.__new__(byte_v2_attn_module.ByteV2AttentionImpl)
    impl.raw_fallback_store = object()
    impl._prefill_has_cached_context = lambda _: has_cached_context
    impl._forward_initial_prefill_from_staging = lambda *args: False
    expected = torch.empty((1, 1, 1))
    impl._forward_prefill_from_cache = lambda *args: expected

    def unexpected_spec_route(*_args):
        raise AssertionError("legacy speculative cache reader was used")

    impl._speculative_gqa_query_len = unexpected_spec_route
    query = torch.empty((4, 32, 128), dtype=torch.bfloat16)
    key = torch.empty((4, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)

    result = impl._forward_prefill(
        query,
        key,
        value,
        torch.empty((1, 52_096), dtype=torch.uint8),
        torch.empty_like(query),
        SimpleNamespace(),
    )

    assert result is expected


def test_byte_v2_hybrid_initial_prefill_prefers_exact_raw_staging():
    impl = object.__new__(byte_v2_attn_module.ByteV2AttentionImpl)
    impl.raw_fallback_store = object()
    impl._prefill_has_cached_context = lambda _: False
    impl._forward_initial_prefill_from_staging = lambda *args: True

    def unexpected_compact_route(*_args):
        raise AssertionError("compact-cache prefill reader was used")

    impl._forward_prefill_from_cache = unexpected_compact_route
    query = torch.empty((4, 32, 128), dtype=torch.bfloat16)
    key = torch.empty((4, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)

    result = impl._forward_prefill(
        query,
        key,
        value,
        torch.empty((1, 52_096), dtype=torch.uint8),
        output,
        SimpleNamespace(),
    )

    assert result is output


def test_byte_v2_hybrid_rejects_cross_layer_kv_sharing(monkeypatch):
    monkeypatch.setenv("BYTE_V2_FA2_HYBRID_RAW_FALLBACK", "1")

    with pytest.raises(RuntimeError, match="cross-layer KV-cache sharing"):
        byte_v2_attn_module.ByteV2AttentionImpl(
            num_heads=32,
            head_size=128,
            scale=0.125,
            num_kv_heads=8,
            kv_sharing_target_layer_name="model.layers.0.self_attn",
        )


def test_byte_v2_hybrid_rejects_unsupported_local_head_shape(monkeypatch):
    monkeypatch.setenv("BYTE_V2_FA2_HYBRID_RAW_FALLBACK", "1")

    with pytest.raises(RuntimeError, match="Hq=32, Hkv=8"):
        byte_v2_attn_module.ByteV2AttentionImpl(
            num_heads=16,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
        )


def test_byte_v2_hybrid_rejects_nondefault_tile_policy(monkeypatch):
    monkeypatch.setenv("BYTE_V2_FA2_HYBRID_RAW_FALLBACK", "1")
    monkeypatch.setenv("BYTE_V2_COMPUTE_BLOCK_N", "128")

    with pytest.raises(RuntimeError, match="default ByteV2 V5 tile policy"):
        byte_v2_attn_module.ByteV2AttentionImpl(
            num_heads=32,
            head_size=128,
            scale=0.125,
            num_kv_heads=8,
        )


def test_byte_v2_decode_partition_size_uses_long_context_default(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._decode_partition_size(2048) == 16
    assert impl._decode_partition_size(4096) == 32
    assert impl._decode_partition_size(8192) == 64
    assert impl._decode_partition_size(16384) == 128
    assert impl._decode_partition_size(4096, num_decode_tokens=2) == 64
    assert impl._decode_partition_size(8192, num_decode_tokens=2) == 64
    assert impl._decode_partition_size(16384, num_decode_tokens=2) == 64
    assert impl._decode_partition_size(4096, num_decode_tokens=4) == 64
    assert impl._decode_partition_size(8192, num_decode_tokens=4) == 64
    assert impl._decode_partition_size(16384, num_decode_tokens=4) == 128


def test_byte_v2_speculative_q4_uses_dedicated_partition_sizes(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._speculative_q4_partition_size(1024) == 64
    assert impl._speculative_q4_partition_size(2048) == 128
    assert impl._speculative_q4_partition_size(4096) == 256
    assert impl._speculative_q4_partition_size(8192) == 256
    assert impl._speculative_q4_partition_size(16384) == 512


def test_byte_v2_cached_prefix_q16_uses_dedicated_partition_sizes(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._speculative_gqa_partition_size(512, 16) == 64
    assert impl._speculative_gqa_partition_size(1024, 16) == 64
    assert impl._speculative_gqa_partition_size(2048, 16) == 128
    assert impl._speculative_gqa_partition_size(4096, 16) == 256
    assert impl._speculative_gqa_partition_size(8192, 16) == 512
    assert impl._speculative_gqa_partition_size(16384, 16) == 512


def test_byte_v2_speculative_q8_uses_long_context_partition(monkeypatch):
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._speculative_gqa_partition_size(4096, 8) == 256
    assert impl._speculative_gqa_partition_size(8192, 8) == 512
    assert impl._speculative_gqa_partition_size(8192, 4) == 256


def test_byte_v2_decode_partition_size_keeps_explicit_base(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", "64")
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._decode_partition_size(4096) == 64
    assert impl._decode_partition_size(4096, num_decode_tokens=4) == 64


def test_byte_v2_gqa_packed_requires_no_outlier(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl.decode_gqa_packed is False
    assert impl._decode_tile_policy_tuple(4096)[-1] == 0
    assert impl._decode_partition_size(4096) == 32


def test_byte_v2_gqa_packed_uses_auto_partition(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_FA2_LIKE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._use_gqa_packed_decode(1024) is False
    assert impl._decode_tile_policy_tuple(1024)[-1] == 0
    assert impl._decode_partition_size(1024) == 16
    assert impl._use_gqa_packed_decode(2048) is True
    assert impl._decode_tile_policy_tuple(2048)[-1] == 1
    assert impl._decode_partition_size(2048) == 32
    assert impl._decode_partition_size(4096) == 64
    assert impl._decode_partition_size(8192) == 128
    assert impl._decode_partition_size(16384) == 64
    assert impl._decode_partition_size(4096, num_decode_tokens=2) == 32
    assert impl._decode_partition_size(8192, num_decode_tokens=2) == 32
    assert impl._decode_partition_size(16384, num_decode_tokens=2) == 64
    assert impl._decode_partition_size(4096, num_decode_tokens=4) == 32
    assert impl._decode_partition_size(8192, num_decode_tokens=4) == 64
    assert impl._decode_partition_size(16384, num_decode_tokens=4) == 64


def test_byte_v2_gqa_fa2_direct_uses_auto_partition(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._decode_tile_policy_tuple(2048)[-5:] == (1, 1, 0, 0, 1)
    assert impl._decode_partition_size(2048) == 128
    assert impl._decode_partition_size(4096) == 256
    assert impl._decode_partition_size(8192) == 512
    assert impl._decode_partition_size(16384) == 1024
    assert impl._decode_partition_size(32768) == 2048
    assert impl._auto_gqa_fa2_direct_num_splits(2048, 1) == 16
    assert impl._auto_gqa_fa2_direct_num_splits(8192, 1) == 16
    assert impl._auto_gqa_fa2_direct_num_splits(32768, 1) == 16
    assert impl._decode_partition_size(2048, num_decode_tokens=2) == 256
    assert impl._decode_partition_size(4096, num_decode_tokens=2) == 512
    assert impl._decode_partition_size(8192, num_decode_tokens=2) == 1024
    assert impl._decode_partition_size(16384, num_decode_tokens=2) == 256
    assert impl._decode_partition_size(32768, num_decode_tokens=2) == 512
    assert impl._auto_gqa_fa2_direct_num_splits(2048, 2) == 8
    assert impl._auto_gqa_fa2_direct_num_splits(4096, 2) == 8
    assert impl._auto_gqa_fa2_direct_num_splits(8192, 2) == 8
    assert impl._auto_gqa_fa2_direct_num_splits(32768, 2) == 64
    assert impl._decode_partition_size(2048, num_decode_tokens=4) == 512
    assert impl._decode_partition_size(2072, num_decode_tokens=4) == 528
    assert (
        impl._decode_partition_size(
            2072,
            num_decode_tokens=4,
            seq_len_sum=6630,
        )
        == 64
    )
    assert (
        impl._decode_partition_size(
            2072,
            num_decode_tokens=4,
            seq_len_sum=6631,
        )
        == 528
    )
    assert impl._decode_partition_size(4096, num_decode_tokens=4) == 1024
    assert impl._decode_partition_size(8192, num_decode_tokens=4) == 256
    assert impl._decode_partition_size(16384, num_decode_tokens=4) == 512
    assert impl._decode_partition_size(32768, num_decode_tokens=4) == 512
    assert impl._auto_gqa_fa2_direct_num_splits(2048, 4) == 4
    assert impl._auto_gqa_fa2_direct_num_splits(4096, 4) == 4
    assert impl._auto_gqa_fa2_direct_num_splits(8192, 4) == 32
    assert impl._auto_gqa_fa2_direct_num_splits(32768, 4) == 64


def test_byte_v2_unnormalized_partition_output_is_opt_in_gqa4(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._decode_tile_policy_tuple(2048)[-6:] == (1, 1, 0, 0, 1, 27)
    output = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    _, max_logits, _ = impl._get_split_k_workspace(
        output,
        2048,
        num_decode_tokens=1,
        use_gqa_packed=True,
    )
    assert max_logits.shape == (1, 32, 16)


def test_byte_v2_unnormalized_partition_output_fails_closed_for_non_gqa4(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=64,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._decode_tile_policy_tuple(2048)[-5:] == (1, 1, 0, 0, 1)
    output = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    _, max_logits, _ = impl._get_split_k_workspace(
        output,
        2048,
        num_decode_tokens=1,
        use_gqa_packed=True,
    )
    assert max_logits.shape == (0,)


def test_byte_v2_split_k_workspace_keeps_backing_storage_for_smaller_batches(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    large_output = torch.empty((4, 32, 128), dtype=torch.bfloat16)
    large = impl._get_split_k_workspace(
        large_output,
        4096,
        num_decode_tokens=4,
        use_gqa_packed=True,
    )
    large_ptrs = tuple(tensor.data_ptr() for tensor in large)

    for num_tokens in (2, 1):
        output = torch.empty((num_tokens, 32, 128), dtype=torch.bfloat16)
        exp_sums, max_logits, tmp_out = impl._get_split_k_workspace(
            output,
            4096,
            num_decode_tokens=num_tokens,
            use_gqa_packed=True,
        )

        assert (exp_sums.shape[0], max_logits.shape[0], tmp_out.shape[0]) == (
            num_tokens,
            num_tokens,
            num_tokens,
        )
        assert (
            tuple(tensor.data_ptr() for tensor in (exp_sums, max_logits, tmp_out))
            == large_ptrs
        )
        assert exp_sums.is_contiguous()
        assert max_logits.is_contiguous()
        assert tmp_out.is_contiguous()


def test_byte_v2_split_k_workspace_reserves_ragged_b4_partitions(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_UNNORMALIZED_PARTITION_OUTPUT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    output = torch.empty((4, 32, 128), dtype=torch.bfloat16)
    uniform = impl._get_split_k_workspace(
        output,
        2072,
        num_decode_tokens=4,
        use_gqa_packed=True,
        partition_size=528,
    )
    uniform_ptrs = tuple(tensor.data_ptr() for tensor in uniform)
    assert tuple(tensor.shape for tensor in uniform) == (
        (4, 32, 4),
        (4, 32, 4),
        (4, 32, 4, 128),
    )

    ragged = impl._get_split_k_workspace(
        output,
        2072,
        num_decode_tokens=4,
        use_gqa_packed=True,
        partition_size=64,
    )

    assert tuple(tensor.shape for tensor in ragged) == (
        (4, 32, 33),
        (4, 32, 33),
        (4, 32, 33, 128),
    )
    assert tuple(tensor.data_ptr() for tensor in ragged) == uniform_ptrs


def test_byte_v2_ragged_q4_uses_independent_stable_workspace(monkeypatch):
    monkeypatch.setenv("BYTE_V2_SPECULATIVE_VERIFY_RAGGED_Q4", "1")
    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    decode_output = torch.empty((4, 32, 128), dtype=torch.bfloat16)
    decode_workspace = impl._get_split_k_workspace(
        decode_output,
        4096,
        num_decode_tokens=4,
        use_gqa_packed=True,
    )
    decode_ptrs = tuple(tensor.data_ptr() for tensor in decode_workspace)

    padded_output = torch.empty((8, 32, 128), dtype=torch.bfloat16)
    exp_sums, max_logits, tmp_out = impl._get_speculative_ragged_q4_workspace(
        padded_output,
        4096,
        num_requests=4,
        partition_size=64,
    )

    assert exp_sums.shape == (4, 128, 64)
    assert max_logits.shape == (0,)
    assert tmp_out.shape == (4, 128, 64, 128)
    assert exp_sums.data_ptr() != decode_workspace[0].data_ptr()
    assert tmp_out.data_ptr() != decode_workspace[2].data_ptr()
    repeated_decode_workspace = impl._get_split_k_workspace(
        decode_output,
        4096,
        num_decode_tokens=4,
        use_gqa_packed=True,
    )
    assert tuple(tensor.data_ptr() for tensor in repeated_decode_workspace) == (
        decode_ptrs
    )


@pytest.mark.parametrize("q_heads_per_kv", [1, 2, 4, 8, 16, 32])
def test_byte_v2_gqa_fa2_direct_allows_grouped_q_per_kv(
    monkeypatch,
    q_heads_per_kv,
):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_LIKE", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_FA2_DIRECT", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=8 * q_heads_per_kv,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._use_gqa_packed_decode(2048) is True
    assert impl._decode_tile_policy_tuple(2048)[-5:] == (1, 1, 0, 0, 1)


def test_byte_v2_gqa_packed_keeps_explicit_partition(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", "128")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl._use_gqa_packed_decode(2048) is True
    assert impl._decode_partition_size(2048) == 128
    assert impl._decode_partition_size(4096) == 128
    assert impl._decode_partition_size(8192, num_decode_tokens=4) == 128
    assert (
        impl._decode_partition_size(
            2072,
            num_decode_tokens=4,
            seq_len_sum=0,
        )
        == 128
    )


def test_byte_v2_decode_validation_disables_unsafe_fast_path(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_VALIDATE_NO_OUTLIER", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    monkeypatch.setattr(
        impl,
        "_decode_cache_has_fallback_or_outlier",
        lambda *args: True,
    )

    class _CudaLike:
        is_cuda = True

    kv_cache = _CudaLike()
    block_table = _CudaLike()
    seq_lens = _CudaLike()

    (
        assume_no_outlier,
        use_gqa_packed,
        validated_all_safe,
    ) = impl._resolve_decode_fast_path(
        kv_cache,
        block_table,
        seq_lens,
        4096,
    )

    assert assume_no_outlier is False
    assert use_gqa_packed is False
    assert validated_all_safe is False
    assert impl._decode_tile_policy_tuple(
        4096,
        assume_no_outlier=assume_no_outlier,
        use_gqa_packed=use_gqa_packed,
    )[-2:] == (0, 0)
    assert impl._decode_partition_size(4096, use_gqa_packed=use_gqa_packed) == 32


def test_byte_v2_decode_validation_marks_all_safe_fast_path(monkeypatch):
    monkeypatch.setenv("BYTE_V2_DECODE_GQA_PACKED", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_ASSUME_NO_OUTLIER", "1")
    monkeypatch.setenv("BYTE_V2_DECODE_VALIDATE_NO_OUTLIER", "1")
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("BYTE_V2_DECODE_SPLIT_K_LONG_MIN_SEQ_LEN", raising=False)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    monkeypatch.setattr(
        impl,
        "_decode_cache_has_fallback_or_outlier",
        lambda *args: False,
    )

    class _CudaLike:
        is_cuda = True

    kv_cache = _CudaLike()
    block_table = _CudaLike()
    seq_lens = _CudaLike()

    (
        assume_no_outlier,
        use_gqa_packed,
        validated_all_safe,
    ) = impl._resolve_decode_fast_path(
        kv_cache,
        block_table,
        seq_lens,
        4096,
    )

    assert assume_no_outlier is True
    assert use_gqa_packed is True
    assert validated_all_safe is True


def test_byte_v2_kv_cache_spec_uses_byte_page_layout():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
    )
    backend_cls = AttentionBackendEnum.BYTE_V2.get_class()

    assert spec.dtype == torch.uint8
    assert spec.tile_policy == DEFAULT_BYTE_V2_TILE_POLICY
    assert spec.outlier_value_bits == 8
    assert spec.outlier_entries_per_tile == 256
    assert spec.include_raw_payload is False
    assert spec.real_page_size_bytes == ByteV2PageLayoutV5().page_size_bytes
    assert spec.page_size_bytes == ByteV2PageLayoutV5().page_size_bytes
    assert backend_cls.get_kv_cache_shape(
        num_blocks=3,
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
    ) == (3, spec.page_size_bytes)


def test_page_layout_v5_parameterizes_outlier_pool_capacity():
    layout = ByteV2PageLayoutV5(outlier_pool_entries=2048)

    assert layout.outlier_pool_entries == 2048
    assert layout.page_size_bytes < _raw_payload_layout().page_size_bytes


def test_byte_v2_kv_cache_spec_rejects_uncompiled_layout_capacity():
    with pytest.raises(ValueError, match="compiled V5 CUDA layout"):
        ByteV2FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            outlier_pool_entries=2048,
        )


def test_byte_v2_kv_cache_spec_rejects_non_byte_storage():
    with pytest.raises(ValueError, match="requires dtype=torch.uint8"):
        ByteV2FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )


def test_attention_layer_uses_byte_v2_kv_cache_spec():
    class ByteV2BackendForTest:
        @staticmethod
        def get_name():
            return "BYTE_V2"

    attn = Attention.__new__(Attention)
    attn.attn_type = AttentionType.DECODER
    attn.attn_backend = ByteV2BackendForTest
    attn.kv_cache_dtype = "auto"
    attn.sliding_window = None
    attn.num_kv_heads = 8
    attn.head_size = 128
    attn.head_size_v = 128
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))

    spec = Attention.get_kv_cache_spec(attn, vllm_config)

    assert isinstance(spec, ByteV2FullAttentionSpec)
    assert spec.dtype == torch.uint8
    assert spec.outlier_entries_per_tile == 256
    assert spec.include_raw_payload is False
    assert spec.page_size_bytes == ByteV2PageLayoutV5().page_size_bytes


def test_byte_v2_backend_rejects_unsupported_shape_before_kernel_gate():
    backend_cls = AttentionBackendEnum.BYTE_V2.get_class()

    invalid_reasons = backend_cls.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=32,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(7, 5),
        attn_type="decoder",
    )

    assert "head_size not supported" in invalid_reasons
    assert "dtype not supported" in invalid_reasons
    assert "block_size not supported" in invalid_reasons
    assert "compute capability not supported" not in invalid_reasons
    assert "ByteV2 requires CUDA compute capability >= 8.0" in invalid_reasons


def test_byte_v2_ops_fail_cleanly_without_registered_kernels():
    cache_op_registered = _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache")
    append_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_append_raw_staging"
    )
    prepare_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_prepare_raw_staging"
    )
    hydrate_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_hydrate_raw_staging_from_cache"
    )
    release_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_release_raw_staging"
    )
    fused_release_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_release_raw_staging_and_update_flags"
    )
    commit_staging_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_commit_raw_staging_to_cache"
    )
    collect_stats_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_collect_cache_stats"
    )
    update_flags_op_registered = _has_torch_op(
        "_C_cache_ops", "byte_v2_update_cache_unsafe_flags"
    )
    decode_op_registered = _has_torch_op("_C", "byte_v2_paged_decode_attention")
    guarded_decode_op_registered = _has_torch_op(
        "_C", "byte_v2_paged_decode_attention_split_k_guarded"
    )
    if (
        cache_op_registered
        and append_staging_op_registered
        and prepare_staging_op_registered
        and hydrate_staging_op_registered
        and release_staging_op_registered
        and fused_release_staging_op_registered
        and commit_staging_op_registered
        and collect_stats_op_registered
        and update_flags_op_registered
        and decode_op_registered
        and guarded_decode_op_registered
    ):
        pytest.skip("ByteV2 custom ops are registered in this environment")

    key = torch.empty((1, 1, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty(
        (1, ByteV2PageLayoutV5().page_size_bytes),
        dtype=torch.uint8,
    )
    slot_mapping = torch.zeros((1,), dtype=torch.int64)
    raw_staging = torch.empty(
        (1, ByteV2RawStagingLayout().slot_size_bytes), dtype=torch.uint8
    )
    block_to_staging_slot = torch.zeros((1,), dtype=torch.int32)
    staging_to_physical_block = torch.zeros((1,), dtype=torch.int32)
    valid_rows = torch.ones((1,), dtype=torch.int32)
    next_staging_slot = torch.zeros((1,), dtype=torch.int32)
    overflow = torch.zeros((1,), dtype=torch.int32)
    stats = torch.empty((4,), dtype=torch.int32)
    page_unsafe_flags = torch.empty((1,), dtype=torch.int32)
    block_tables = torch.zeros((1, 1), dtype=torch.int32)
    seq_lens = torch.ones((1,), dtype=torch.int32)

    if not cache_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_reshape_and_cache(
                key,
                value,
                kv_cache,
                slot_mapping,
                codec_token_block=16,
                codec_dim_block=16,
                alloc_block_tokens=16,
            )

    if not append_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_append_raw_staging(
                key,
                value,
                raw_staging,
                slot_mapping,
                block_to_staging_slot,
                codec_token_block=16,
                codec_dim_block=16,
                alloc_block_tokens=16,
            )

    if not prepare_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_prepare_raw_staging(
                slot_mapping,
                block_to_staging_slot,
                staging_to_physical_block,
                valid_rows,
                next_staging_slot,
                overflow,
                alloc_block_tokens=16,
            )

    if not hydrate_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_hydrate_raw_staging_from_cache(
                raw_staging,
                kv_cache,
                staging_to_physical_block,
                valid_rows,
                codec_token_block=16,
                codec_dim_block=16,
                alloc_block_tokens=16,
            )

    if not release_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_release_raw_staging(
                block_to_staging_slot,
                staging_to_physical_block,
                valid_rows,
                next_staging_slot,
                overflow,
            )

    if not fused_release_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_release_raw_staging_and_update_flags(
                block_to_staging_slot,
                staging_to_physical_block,
                valid_rows,
                next_staging_slot,
                overflow,
                page_unsafe_flags,
                kv_cache,
                tile_policy=(16, 16, 16, 64, 128, 128),
            )

    if not commit_staging_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_commit_raw_staging_to_cache(
                raw_staging,
                kv_cache,
                block_to_staging_slot,
                valid_rows,
                codec_token_block=16,
                codec_dim_block=16,
                alloc_block_tokens=16,
            )

    if not collect_stats_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_collect_cache_stats(
                stats,
                kv_cache,
                block_tables,
                seq_lens,
                max_seq_len=16,
                tile_policy=(16, 16, 16, 64, 128, 128),
            )

    if not update_flags_op_registered:
        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_update_cache_unsafe_flags(
                page_unsafe_flags,
                kv_cache,
                slot_mapping,
                tile_policy=(16, 16, 16, 64, 128, 128),
            )

    if not decode_op_registered:
        output = torch.empty((1, 32, 128), dtype=torch.bfloat16)
        query = torch.empty_like(output)

        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_paged_decode_attention(
                output,
                query,
                kv_cache,
                block_tables,
                seq_lens,
                scale=1.0,
                num_kv_heads=8,
                block_size=16,
                max_seq_len=1,
                tile_policy=(16, 16, 16, 64, 128, 128),
            )

    if not guarded_decode_op_registered:
        output = torch.empty((1, 32, 128), dtype=torch.bfloat16)
        query = torch.empty_like(output)
        exp_sums = torch.empty((1, 32, 1), dtype=torch.float32)
        max_logits = torch.empty((0,), dtype=torch.float32)
        tmp_out = torch.empty((1, 32, 1, 128), dtype=torch.float32)

        with pytest.raises(NotImplementedError, match="ByteV2 custom kernels"):
            byte_v2_paged_decode_attention_split_k_guarded(
                output,
                exp_sums,
                max_logits,
                tmp_out,
                query,
                kv_cache,
                page_unsafe_flags,
                block_tables,
                seq_lens,
                scale=1.0,
                num_kv_heads=8,
                block_size=16,
                max_seq_len=16,
                partition_size=16,
                tile_policy=(16, 16, 16, 64, 128, 128, 0, 1),
            )


def test_byte_v2_kv_cache_update_uses_and_reuses_raw_staging(monkeypatch):
    calls = []
    active_slot_capacities = []

    def prepare(
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        *,
        alloc_block_tokens,
    ):
        calls.append("prepare")
        active_slot_capacities.append(staging_to_physical_block.shape[0])
        assert alloc_block_tokens == 16
        assert staging_to_physical_block.shape == slot_mapping.shape
        assert valid_rows.shape == slot_mapping.shape
        assert block_to_staging_slot.tolist() == [-1, -1, -1, -1]
        block_to_staging_slot[0] = 0
        staging_to_physical_block[0] = 0
        valid_rows[0] = int(slot_mapping.shape[0])
        next_staging_slot[0] = 1
        overflow[0] = 0

    def append(
        key,
        value,
        raw_staging,
        slot_mapping,
        block_to_staging_slot,
        *,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    ):
        del value, slot_mapping, block_to_staging_slot
        calls.append("append")
        assert key.shape[1:] == (8, 128)
        active_slot_capacities.append(raw_staging.shape[0])
        assert raw_staging.shape == (
            key.shape[0],
            ByteV2RawStagingLayout().slot_size_bytes,
        )
        assert (codec_token_block, codec_dim_block, alloc_block_tokens) == (
            16,
            16,
            16,
        )

    def hydrate(
        raw_staging,
        kv_cache,
        staging_to_physical_block,
        valid_rows,
        *,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    ):
        del kv_cache
        calls.append("hydrate")
        active_slot_capacities.append(staging_to_physical_block.shape[0])
        assert raw_staging.shape[0] == staging_to_physical_block.shape[0]
        assert valid_rows.shape == staging_to_physical_block.shape
        assert (codec_token_block, codec_dim_block, alloc_block_tokens) == (
            16,
            16,
            16,
        )

    def commit(
        raw_staging,
        kv_cache,
        staging_to_physical_block,
        valid_rows,
        *,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    ):
        del kv_cache
        calls.append("commit")
        active_slot_capacities.append(staging_to_physical_block.shape[0])
        assert raw_staging.shape[0] == staging_to_physical_block.shape[0]
        assert valid_rows.shape == staging_to_physical_block.shape
        assert (codec_token_block, codec_dim_block, alloc_block_tokens) == (
            16,
            16,
            16,
        )

    def release(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
    ):
        calls.append("release")
        active_slot_capacities.append(staging_to_physical_block.shape[0])
        assert valid_rows.shape == staging_to_physical_block.shape
        block_to_staging_slot.fill_(-1)
        staging_to_physical_block.fill_(-1)
        valid_rows.zero_()
        next_staging_slot.zero_()
        overflow.zero_()

    def direct(*args, **kwargs):
        del args, kwargs
        calls.append("direct")

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_prepare_raw_staging", prepare)
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_hydrate_raw_staging_from_cache",
        hydrate,
    )
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_append_raw_staging", append)
    monkeypatch.setattr(
        byte_v2_attn_module, "byte_v2_commit_raw_staging_to_cache", commit
    )
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_release_raw_staging", release)
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_reshape_and_cache", direct)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
    )
    key = torch.empty((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((4, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)

    impl.do_kv_cache_update(None, key, value, kv_cache, slot_mapping)
    first_staging_buffer = impl.raw_staging_manager.raw_staging

    assert calls == ["prepare", "hydrate", "append", "commit", "release"]
    assert active_slot_capacities == [2, 2, 2, 2, 2]
    assert first_staging_buffer is not None
    assert impl.raw_staging_manager.block_to_staging_slot.tolist() == [-1, -1, -1, -1]
    assert impl.raw_staging_manager.next_staging_slot.item() == 0

    calls.clear()
    active_slot_capacities.clear()
    impl.do_kv_cache_update(None, key[:1], value[:1], kv_cache, slot_mapping[:1])

    assert calls == ["prepare", "hydrate", "append", "commit", "release"]
    assert active_slot_capacities == [1, 1, 1, 1, 1]
    assert impl.raw_staging_manager.raw_staging is first_staging_buffer


def test_byte_v2_raw_staging_manager_native_single_token_update_is_opt_in(
    monkeypatch,
):
    calls = []

    def single_token_update(*args, **kwargs):
        del args, kwargs
        calls.append("single_token")

    def raw_staging_update(*args, **kwargs):
        del args, kwargs
        calls.append("raw_staging")

    env_name = "BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE"
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("BYTE_V2_NATIVE_RAW_STAGING_UPDATE", "1")
    monkeypatch.setenv("BYTE_V2_FUSED_STAGING_RELEASE_FLAGS", "1")
    monkeypatch.delenv("BYTE_V2_DEBUG_WARMUP", raising=False)
    monkeypatch.delenv("VLLM_BYTE_V2_DEBUG_WARMUP", raising=False)
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_cache_single_token",
        single_token_update,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_cache_raw_staging",
        raw_staging_update,
    )

    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
    )
    key = torch.empty((1, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty(
        (1, ByteV2PageLayoutV5().page_size_bytes),
        dtype=torch.uint8,
    )
    slot_mapping = SimpleNamespace(shape=(1,), is_cuda=True)
    page_unsafe_flags = torch.zeros((1,), dtype=torch.int32)

    result = manager.update(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        page_unsafe_flags=page_unsafe_flags,
    )
    assert result == (True, True)
    assert calls == ["raw_staging"]

    calls.clear()
    monkeypatch.setenv(env_name, "1")
    result = manager.update(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        page_unsafe_flags=page_unsafe_flags,
    )
    assert result == (True, True)
    assert calls == ["single_token"]


def test_byte_v2_raw_staging_manager_uses_fused_hybrid_q1_update(monkeypatch):
    layout = ByteV2RawStagingLayout()
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=2,
        num_staging_slots=1,
        slot_size_bytes=layout.slot_size_bytes,
        device=torch.device("cpu"),
    ).allocate()
    hybrid_state = SimpleNamespace(
        raw_pages=torch.empty((1, layout.slot_size_bytes), dtype=torch.uint8),
        page_to_raw_slot=torch.full((2,), -1, dtype=torch.int32),
        free_slots=torch.zeros((1,), dtype=torch.int32),
        free_count=torch.ones((1,), dtype=torch.int32),
        fatal=torch.zeros((1,), dtype=torch.int32),
        forced_raw_diagnostic=None,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
    )
    manager.bind_shared_workspace(workspace)
    calls = []

    def fused_update(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_q1",
        fused_update,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_test_force_promote_raw_staging_q1",
        lambda *args, **kwargs: pytest.fail(
            "default hybrid Q1 launched the test-only forced promotion op"
        ),
    )
    slot_mapping = SimpleNamespace(shape=(1,), is_cuda=True)
    page_flags = torch.zeros((2,), dtype=torch.int32)

    result = manager._update_one_wave(
        key=torch.empty((1, 8, 128), dtype=torch.bfloat16),
        value=torch.empty((1, 8, 128), dtype=torch.bfloat16),
        kv_cache=torch.empty(
            (2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8
        ),
        slot_mapping=slot_mapping,
        page_unsafe_flags=page_flags,
        hybrid_state=hybrid_state,
        active_slot_capacity=1,
    )

    assert result == (True, True)
    assert len(calls) == 1
    assert calls[0][0][4] is hybrid_state.raw_pages
    assert calls[0][0][5] is slot_mapping
    assert calls[0][1]["page_unsafe_flags"] is page_flags


def test_byte_v2_raw_staging_manager_routes_test_forced_raw_only_after_q1(
    monkeypatch,
):
    layout = ByteV2RawStagingLayout()
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=2,
        num_staging_slots=1,
        slot_size_bytes=layout.slot_size_bytes,
        device=torch.device("cpu"),
    ).allocate()
    diagnostic = torch.zeros((3,), dtype=torch.int32)
    hybrid_state = SimpleNamespace(
        raw_pages=torch.empty((1, layout.slot_size_bytes), dtype=torch.uint8),
        page_to_raw_slot=torch.full((2,), -1, dtype=torch.int32),
        free_slots=torch.zeros((1,), dtype=torch.int32),
        free_count=torch.ones((1,), dtype=torch.int32),
        fatal=torch.zeros((1,), dtype=torch.int32),
        forced_raw_diagnostic=diagnostic,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
    )
    manager.bind_shared_workspace(workspace)
    calls = []

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_q1",
        lambda *args, **kwargs: calls.append("fused_q1"),
    )

    def forced(*args, **kwargs):
        del kwargs
        assert args[-1] is diagnostic
        calls.append("forced_raw")

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_test_force_promote_raw_staging_q1",
        forced,
    )

    result = manager._update_one_wave(
        key=torch.empty((1, 8, 128), dtype=torch.bfloat16),
        value=torch.empty((1, 8, 128), dtype=torch.bfloat16),
        kv_cache=torch.empty(
            (2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8
        ),
        slot_mapping=SimpleNamespace(shape=(1,), is_cuda=True),
        page_unsafe_flags=None,
        hybrid_state=hybrid_state,
        active_slot_capacity=1,
    )

    assert result == (True, False)
    assert calls == ["fused_q1", "forced_raw"]


def test_byte_v2_raw_staging_manager_uses_fused_hybrid_multi_token_update(
    monkeypatch,
):
    layout = ByteV2RawStagingLayout()
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=2,
        num_staging_slots=2,
        slot_size_bytes=layout.slot_size_bytes,
        device=torch.device("cpu"),
    ).allocate()
    hybrid_state = SimpleNamespace(
        raw_pages=torch.empty((1, layout.slot_size_bytes), dtype=torch.uint8),
        page_to_raw_slot=torch.full((2,), -1, dtype=torch.int32),
        free_slots=torch.zeros((1,), dtype=torch.int32),
        free_count=torch.ones((1,), dtype=torch.int32),
        fatal=torch.zeros((1,), dtype=torch.int32),
        forced_raw_diagnostic=None,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
    )
    manager.bind_shared_workspace(workspace)
    calls = []

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_q1",
        lambda *args, **kwargs: pytest.fail("Q2 used the fused hybrid Q1 op"),
    )

    def fused_update(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
        fused_update,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_test_force_promote_raw_staging_q1",
        lambda *args, **kwargs: pytest.fail("Q2 used forced raw promotion"),
    )
    slot_mapping = SimpleNamespace(shape=(2,), is_cuda=True)
    page_flags = torch.zeros((2,), dtype=torch.int32)

    result = manager._update_one_wave(
        key=torch.empty((2, 8, 128), dtype=torch.bfloat16),
        value=torch.empty((2, 8, 128), dtype=torch.bfloat16),
        kv_cache=torch.empty(
            (2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8
        ),
        slot_mapping=slot_mapping,
        page_unsafe_flags=page_flags,
        hybrid_state=hybrid_state,
        active_slot_capacity=2,
    )

    assert result == (True, True)
    assert len(calls) == 1
    assert calls[0][0][4] is hybrid_state.raw_pages
    assert calls[0][0][5] is slot_mapping
    assert calls[0][1]["page_unsafe_flags"] is page_flags


def test_byte_v2_raw_staging_manager_retains_initial_prefill(monkeypatch):
    layout = ByteV2RawStagingLayout()
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=2,
        num_staging_slots=2,
        slot_size_bytes=layout.slot_size_bytes,
        device=torch.device("cpu"),
    ).allocate()
    hybrid_state = SimpleNamespace(
        raw_pages=torch.empty((1, layout.slot_size_bytes), dtype=torch.uint8),
        page_to_raw_slot=torch.full((2,), -1, dtype=torch.int32),
        free_slots=torch.zeros((1,), dtype=torch.int32),
        free_count=torch.ones((1,), dtype=torch.int32),
        fatal=torch.zeros((1,), dtype=torch.int32),
        forced_raw_diagnostic=None,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
    )
    manager.bind_shared_workspace(workspace)
    calls = []

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_multi_token_retained",
        lambda *args, **kwargs: calls.append("retained"),
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
        lambda *args, **kwargs: pytest.fail("retained update used releasing op"),
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_release_raw_staging",
        lambda *args, **kwargs: calls.append("release"),
    )
    kv_cache = torch.empty((2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    key = torch.empty((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)

    result = manager._update_one_wave(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=SimpleNamespace(shape=(2,), is_cuda=True),
        page_unsafe_flags=None,
        hybrid_state=hybrid_state,
        active_slot_capacity=2,
        retain_initial_prefill=True,
    )
    staged = manager.stage_initial_prefill(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([0, 16], dtype=torch.int64),
        attn_metadata=SimpleNamespace(),
    )

    assert result == (True, False)
    assert staged is not None
    assert staged[0].data_ptr() == workspace.raw_staging.data_ptr()
    assert staged[1] is workspace.block_to_staging_slot
    manager.release_initial_prefill(staged[2], staged[3])
    assert calls == ["retained", "release"]


def test_byte_v2_raw_staging_manager_falls_back_to_generic_multi_token(
    monkeypatch,
):
    layout = ByteV2RawStagingLayout()
    workspace = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=2,
        num_staging_slots=2,
        slot_size_bytes=layout.slot_size_bytes,
        device=torch.device("cpu"),
    ).allocate()
    hybrid_state = SimpleNamespace(
        raw_pages=torch.empty((1, layout.slot_size_bytes), dtype=torch.uint8),
        page_to_raw_slot=torch.full((2,), -1, dtype=torch.int32),
        free_slots=torch.zeros((1,), dtype=torch.int32),
        free_count=torch.ones((1,), dtype=torch.int32),
        fatal=torch.zeros((1,), dtype=torch.int32),
        forced_raw_diagnostic=torch.zeros((3,), dtype=torch.int32),
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=SimpleNamespace(state=lambda _: hybrid_state),
    )
    manager.bind_shared_workspace(workspace)
    calls = []

    def record(name):
        def implementation(*args, **kwargs):
            del args, kwargs
            calls.append(name)

        return implementation

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_q1",
        lambda *args, **kwargs: pytest.fail("Q2 used the fused hybrid Q1 op"),
    )

    def missing_multi_token(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_hybrid_cache_raw_staging_multi_token",
        missing_multi_token,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_test_force_promote_raw_staging_q1",
        lambda *args, **kwargs: pytest.fail("Q2 used forced raw promotion"),
    )
    monkeypatch.setattr(
        byte_v2_attn_module, "byte_v2_prepare_raw_staging", record("prepare")
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_hydrate_raw_staging_from_hybrid_cache",
        record("hydrate"),
    )
    monkeypatch.setattr(
        byte_v2_attn_module, "byte_v2_append_raw_staging", record("append")
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_commit_raw_staging_to_hybrid_cache",
        record("commit"),
    )

    def release(*args, **kwargs):
        del args, kwargs
        calls.append("release")
        return False

    monkeypatch.setattr(
        manager,
        "_release_allocator_state",
        release,
    )
    monkeypatch.setattr(torch, "_assert_async", lambda *args, **kwargs: None)

    result = manager._update_one_wave(
        key=torch.empty((2, 8, 128), dtype=torch.bfloat16),
        value=torch.empty((2, 8, 128), dtype=torch.bfloat16),
        kv_cache=torch.empty(
            (2, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8
        ),
        slot_mapping=torch.tensor([0, 16], dtype=torch.int64),
        page_unsafe_flags=None,
        hybrid_state=hybrid_state,
        active_slot_capacity=2,
    )

    assert result == (True, False)
    assert calls == ["prepare", "hydrate", "append", "commit", "release"]


@pytest.mark.parametrize("fused_stage", [False, True])
def test_byte_v2_kv_update_supplies_flags_for_fused_single_token_staging(
    monkeypatch,
    fused_stage,
):
    env_name = "BYTE_V2_FUSED_SINGLE_TOKEN_STAGING"
    if fused_stage:
        monkeypatch.setenv(env_name, "1")
    else:
        monkeypatch.setenv(env_name, "0")

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
    )
    impl.decode_page_unsafe_flags = False
    supplied_flags = []
    flags = object()

    def get_flags(kv_cache):
        del kv_cache
        return flags

    def update(**kwargs):
        supplied_flags.append(kwargs["page_unsafe_flags"])
        return True, kwargs["page_unsafe_flags"] is not None

    monkeypatch.setattr(impl, "_get_decode_page_unsafe_flags", get_flags)
    monkeypatch.setattr(impl.raw_staging_manager, "update", update)
    key = SimpleNamespace(shape=(1, 8, 128))
    value = SimpleNamespace(shape=(1, 8, 128))
    kv_cache = SimpleNamespace(shape=(4, 1), is_cuda=True)
    slot_mapping = SimpleNamespace(shape=(1,), is_cuda=True)

    impl.do_kv_cache_update(None, key, value, kv_cache, slot_mapping)

    assert supplied_flags == ([flags] if fused_stage else [None])


def test_byte_v2_raw_staging_manager_uses_native_update_with_flags(monkeypatch):
    calls = []
    monkeypatch.delenv("BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE", raising=False)
    monkeypatch.delenv("BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR", raising=False)

    def native_update(
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
        *,
        tile_policy,
        fuse_metadata_clear,
        warp_parallel_histogram,
        fuse_single_token_staging,
        fuse_single_token_commit_release,
        fuse_single_token_stage_metadata_clear,
    ):
        del key, value, kv_cache, slot_mapping, page_unsafe_flags
        calls.append(
            (
                raw_staging.shape[0],
                tuple(tile_policy),
                fuse_metadata_clear,
                warp_parallel_histogram,
                fuse_single_token_staging,
                fuse_single_token_commit_release,
                fuse_single_token_stage_metadata_clear,
            )
        )
        block_to_staging_slot.fill_(-1)
        staging_to_physical_block.fill_(-1)
        valid_rows.zero_()
        next_staging_slot.zero_()
        overflow.zero_()

    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_update_cache_raw_staging",
        native_update,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
    )
    key = torch.empty((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((4, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    page_unsafe_flags = torch.zeros((4,), dtype=torch.int32)

    handled, flags_updated = manager.update(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        page_unsafe_flags=page_unsafe_flags,
    )

    assert handled is True
    assert flags_updated is True
    assert calls == [(2, (16, 16, 16, 64, 128, 128), True, True, True, False, False)]


def test_byte_v2_kv_cache_update_falls_back_when_raw_staging_is_missing(monkeypatch):
    calls = []

    def prepare(*args, **kwargs):
        del args, kwargs
        calls.append("prepare")
        raise NotImplementedError("raw staging op is unavailable")

    def direct(
        key,
        value,
        kv_cache,
        slot_mapping,
        *,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    ):
        del key, value, kv_cache, slot_mapping
        calls.append("direct")
        assert (codec_token_block, codec_dim_block, alloc_block_tokens) == (
            16,
            16,
            16,
        )

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_prepare_raw_staging", prepare)
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_reshape_and_cache", direct)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
    )
    key = torch.empty((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((4, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)

    impl.do_kv_cache_update(None, key, value, kv_cache, slot_mapping)

    assert calls == ["prepare", "direct"]


def test_byte_v2_kv_cache_update_skips_raw_staging_above_threshold(monkeypatch):
    calls = []

    def prepare(*args, **kwargs):
        del args, kwargs
        calls.append("prepare")

    def direct(
        key,
        value,
        kv_cache,
        slot_mapping,
        *,
        codec_token_block,
        codec_dim_block,
        alloc_block_tokens,
    ):
        del key, value, kv_cache, slot_mapping
        calls.append("direct")
        assert (codec_token_block, codec_dim_block, alloc_block_tokens) == (
            16,
            16,
            16,
        )

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_prepare_raw_staging", prepare)
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_reshape_and_cache", direct)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
    )
    impl.raw_staging_manager.max_tokens_per_update = 1
    key = torch.empty((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((4, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)

    impl.do_kv_cache_update(None, key, value, kv_cache, slot_mapping)

    assert calls == ["direct"]
    assert impl.raw_staging_manager.raw_staging is None


def test_byte_v2_attention_forward_zeros_output_for_profile_run(monkeypatch):
    calls = []

    def decode(*args, **kwargs):
        del args, kwargs
        calls.append("decode")

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_paged_decode_attention", decode)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
    )
    query = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    key = torch.empty((1, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((1, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    output = torch.empty_like(query)

    result = impl.forward(
        None,
        query,
        key,
        value,
        kv_cache,
        None,
        output,
    )

    assert result is output
    assert calls == []
    assert torch.all(output == 0)


def test_byte_v2_attention_forward_uses_native_prefill_when_not_decode_compatible(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_PREFILL_BACKEND", "native")
    calls = []
    prefill_kwargs = {}

    def decode(*args, **kwargs):
        del args, kwargs
        calls.append("decode")

    def prefill(output, query, key, value, query_start_loc, **kwargs):
        del query, key, value, query_start_loc
        calls.append("prefill")
        prefill_kwargs.update(kwargs)
        output.fill_(5)

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_paged_decode_attention", decode)
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_prefill_attention", prefill)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    query = torch.zeros((2, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((1, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    output = torch.empty_like(query)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones((1,), dtype=torch.int32),
        causal=True,
    )

    result = impl.forward(
        None,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["prefill"]
    assert prefill_kwargs["max_query_len"] == 2
    assert tuple(prefill_kwargs["tile_policy"]) == (
        DEFAULT_BYTE_V2_TILE_POLICY.codec_token_block,
        DEFAULT_BYTE_V2_TILE_POLICY.codec_dim_block,
        DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens,
        DEFAULT_BYTE_V2_TILE_POLICY.compute_block_n,
        DEFAULT_BYTE_V2_TILE_POLICY.head_dim,
        DEFAULT_BYTE_V2_TILE_POLICY.head_dim_v,
    )
    assert torch.all(output == 5)


def test_byte_v2_attention_forward_uses_sdpa_prefill_by_default(monkeypatch):
    monkeypatch.delenv("BYTE_V2_PREFILL_BACKEND", raising=False)
    calls = []
    sdpa_kwargs = {}

    def native_prefill(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native prefill should not be called by default")

    def sdpa(query, key, value, **kwargs):
        del key, value
        calls.append("sdpa")
        sdpa_kwargs.update(kwargs)
        return torch.full_like(query, 3)

    monkeypatch.setattr(
        byte_v2_attn_module, "byte_v2_prefill_attention", native_prefill
    )
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    query = torch.zeros((2, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((1, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    output = torch.empty_like(query)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones((1,), dtype=torch.int32),
        causal=True,
    )

    result = impl.forward(
        None,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["sdpa"]
    assert sdpa_kwargs["dropout_p"] == 0.0
    assert sdpa_kwargs["is_causal"] is True
    assert sdpa_kwargs["scale"] == 0.125
    assert sdpa_kwargs["enable_gqa"] is True
    assert torch.all(output == 3)


def test_byte_v2_attention_metadata_builder_preserves_common_prefix_len():
    builder = object.__new__(byte_v2_attn_module.ByteV2AttentionMetadataBuilder)
    builder.tile_policy = DEFAULT_BYTE_V2_TILE_POLICY
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_reqs=1,
        max_query_len=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        max_seq_len=50,
        seq_lens=torch.tensor([50], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([50], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 4), dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int64),
        causal=True,
    )

    metadata = builder.build(
        common_prefix_len=48,
        common_attn_metadata=common_attn_metadata,
    )

    assert metadata.common_prefix_len == 48
    assert metadata.seq_len_sum == 50
    assert metadata.tile_policy == DEFAULT_BYTE_V2_TILE_POLICY


def test_byte_v2_attention_forward_uses_paged_decode_for_prefix_prefill(
    monkeypatch,
):
    calls = []
    seq_lens_seen = []

    def prefill(*args, **kwargs):
        del args, kwargs
        calls.append("prefill")

    def decode(output, query, kv_cache, block_tables, seq_lens, **kwargs):
        del query, kv_cache, block_tables, kwargs
        calls.append("decode")
        seq_lens_seen.extend(seq_lens.tolist())
        values = torch.arange(1, output.shape[0] + 1, dtype=torch.bfloat16).view(
            -1, 1, 1
        )
        output.copy_(values.expand_as(output))

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_paged_decode_attention", decode)
    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_prefill_attention", prefill)

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    query = torch.zeros((2, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    kv_cache = torch.empty((4, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    output = torch.empty_like(query)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=50,
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        seq_lens=torch.tensor([50], dtype=torch.int32),
        causal=True,
        common_prefix_len=0,
    )

    result = impl.forward(
        None,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["decode"]
    assert seq_lens_seen == [49, 50]
    torch.testing.assert_close(output[0], torch.full_like(output[0], 1))
    torch.testing.assert_close(output[1], torch.full_like(output[1], 2))


def test_byte_v2_attention_forward_uses_prefill_fallback_when_not_decode_compatible(
    monkeypatch,
):
    monkeypatch.setenv("BYTE_V2_PREFILL_BACKEND", "fallback")
    calls = []

    def decode(*args, **kwargs):
        del args, kwargs
        calls.append("decode")

    def missing_prefill(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError

    monkeypatch.setattr(byte_v2_attn_module, "byte_v2_paged_decode_attention", decode)
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_prefill_attention",
        missing_prefill,
    )

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    query = torch.zeros((2, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    value[0].fill_(1)
    value[1].fill_(3)
    kv_cache = torch.empty((1, ByteV2PageLayoutV5().page_size_bytes), dtype=torch.uint8)
    output = torch.empty_like(query)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones((1,), dtype=torch.int32),
        causal=True,
    )

    result = impl.forward(
        None,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == []
    torch.testing.assert_close(
        output.float(),
        torch.stack(
            (
                torch.full_like(output[0], 1),
                torch.full_like(output[1], 2),
            )
        ).float(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "seq_lens",
        "force_outliers",
        "permute_pages",
        "shared_prefix_blocks",
        "num_splits",
    ),
    [
        pytest.param((73,), True, False, 0, 0, id="q1-seq73-outliers"),
        pytest.param(
            (73, 128, 4105),
            True,
            True,
            4,
            4,
            id="q1-ragged-permuted-shared-prefix-split4",
        ),
    ],
)
def test_byte_v2_fa2_cuda_matches_raw_fa2_bitwise(
    seq_lens,
    force_outliers,
    permute_pages,
    shared_prefix_blocks,
    num_splits,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")

    from scripts.byte_v2_fa2_oracle import (
        compare_byte_v2_and_raw,
        fa2_oracle_ops_are_available,
        make_inputs,
    )

    if not fa2_oracle_ops_are_available():
        pytest.skip("ByteV2 and raw FA2 extension ops are not registered")

    tensors = make_inputs(
        seq_lens,
        query_len=1,
        force_outliers=force_outliers,
        permute_pages=permute_pages,
        shared_prefix_blocks=shared_prefix_blocks,
    )
    block_table = tensors["block_table"]
    assert isinstance(block_table, torch.Tensor)
    block_table_cpu = block_table.cpu()
    if permute_pages:
        active_rows = [
            block_table_cpu[seq_idx, : (seq_len + 15) // 16].tolist()
            for seq_idx, seq_len in enumerate(seq_lens)
        ]
        assert any(row != sorted(row) for row in active_rows)
    if shared_prefix_blocks:
        expected_prefix = block_table_cpu[0, :shared_prefix_blocks]
        for seq_idx in range(1, len(seq_lens)):
            torch.testing.assert_close(
                block_table_cpu[seq_idx, :shared_prefix_blocks],
                expected_prefix,
            )

    result = compare_byte_v2_and_raw(
        tensors,
        query_len=1,
        iterations=1,
        num_splits=num_splits,
    )

    assert result["out_mismatch"] == 0
    assert result["lse_mismatch"] == 0
    assert result["out_max_abs"] == 0.0
    assert result["lse_max_abs"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "seq_lens",
        "query_len",
        "raw_page_mode",
        "permute_pages",
        "shared_prefix_blocks",
        "num_splits",
    ),
    [
        pytest.param((73,), 1, "none", False, 0, 0, id="all-compact"),
        pytest.param((73,), 1, "tail", False, 0, 0, id="raw-tail"),
        pytest.param((128,), 1, "mixed", False, 0, 0, id="mixed-n128"),
        pytest.param((128,), 1, "all", False, 0, 1, id="all-raw-nonsplit"),
        pytest.param(
            (73, 128, 4105),
            1,
            "ragged",
            True,
            4,
            4,
            id="ragged-permuted-shared-prefix-split4",
        ),
        pytest.param(
            (33, 47),
            4,
            "ragged",
            True,
            1,
            1,
            id="generic-q4-ragged-mixed-partial",
        ),
    ],
)
def test_byte_v2_hybrid_fa2_cuda_matches_raw_fa2_bitwise(
    seq_lens,
    query_len,
    raw_page_mode,
    permute_pages,
    shared_prefix_blocks,
    num_splits,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")

    from scripts.byte_v2_fa2_oracle import (
        add_hybrid_raw_pages,
        compare_hybrid_byte_v2_and_raw,
        hybrid_fa2_oracle_ops_are_available,
        make_inputs,
    )

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")

    tensors = make_inputs(
        seq_lens,
        query_len=query_len,
        force_outliers=True,
        permute_pages=permute_pages,
        shared_prefix_blocks=shared_prefix_blocks,
    )
    block_table = tensors["block_table"]
    assert isinstance(block_table, torch.Tensor)
    table = block_table.cpu()
    active_rows = [
        table[seq_idx, : (seq_len + 15) // 16].tolist()
        for seq_idx, seq_len in enumerate(seq_lens)
    ]
    if raw_page_mode == "none":
        raw_pages = []
    elif raw_page_mode == "tail":
        raw_pages = [active_rows[0][-1]]
    elif raw_page_mode == "mixed":
        raw_pages = active_rows[0][1::3]
    elif raw_page_mode == "all":
        raw_pages = active_rows[0]
    else:
        raw_pages = sorted(
            {page for row in active_rows for page in (row[len(row) // 2], row[-1])}
        )
    if query_len > 1:
        active_pages = {page for row in active_rows for page in row}
        assert len(set(seq_lens)) > 1
        assert any(seq_len % 16 for seq_len in seq_lens)
        assert 0 < len(set(raw_pages)) < len(active_pages)
    add_hybrid_raw_pages(tensors, raw_pages, reverse_raw_slots=True)

    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(page_to_raw_slot, torch.Tensor)
    assert int((page_to_raw_slot >= 0).sum()) == len(set(raw_pages))
    if raw_pages:
        page_to_raw_slot_cpu = page_to_raw_slot.cpu()
        assert any(
            int(page_to_raw_slot_cpu[physical_page]) != physical_page
            for physical_page in raw_pages
        )
    else:
        assert bool((page_to_raw_slot == -1).all())
    result = compare_hybrid_byte_v2_and_raw(
        tensors,
        query_len=query_len,
        iterations=1,
        num_splits=num_splits,
    )

    assert result["out_mismatch"] == 0
    assert result["lse_mismatch"] == 0
    assert result["out_max_abs"] == 0.0
    assert result["lse_max_abs"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_fa2_python_wrapper_q4_matches_raw_bitwise():
    from scripts.byte_v2_fa2_oracle import (
        add_hybrid_raw_pages,
        hybrid_fa2_oracle_ops_are_available,
        make_inputs,
    )

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")

    query_len = 4
    tensors = make_inputs(
        (33, 47),
        query_len=query_len,
        force_outliers=True,
        permute_pages=True,
        shared_prefix_blocks=1,
    )
    block_table = tensors["block_table"]
    query = tensors["query"]
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(query, torch.Tensor)
    active_pages = block_table[:, :3].detach().cpu().flatten().tolist()
    add_hybrid_raw_pages(tensors, sorted(set(active_pages[1::2])))

    hybrid_out = torch.empty_like(query)
    byte_v2_ops_module.byte_v2_fa2_hybrid_paged_decode_attention(
        hybrid_out,
        query,
        tensors["byte_cache"],
        tensors["raw_staging"],
        tensors["page_to_raw_slot"],
        tensors["cu_seqlens_q"],
        block_table,
        tensors["seq_lens"],
        scale=float(tensors["scale"]),
        max_query_len=query_len,
        max_seq_len=int(tensors["max_seq_len"]),
        causal=True,
    )
    raw_out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        query,
        tensors["key"],
        tensors["value"],
        torch.empty_like(query),
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        block_table,
        None,
        query_len,
        int(tensors["max_seq_len"]),
        0.0,
        float(tensors["scale"]),
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )
    torch.accelerator.synchronize()

    assert int((hybrid_out.view(torch.int16) != raw_out.view(torch.int16)).sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_initial_prefill_staging_matches_raw_fa2_bitwise():
    from scripts.byte_v2_fa2_oracle import (
        compare_hybrid_byte_v2_and_raw,
        hybrid_fa2_oracle_ops_are_available,
        make_inputs,
    )

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")

    seq_len = 73
    tensors = make_inputs(
        (seq_len,),
        query_len=seq_len,
        force_outliers=True,
        permute_pages=True,
    )
    byte_cache = tensors["byte_cache"]
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    block_table = tensors["block_table"]
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)

    logical_tokens = torch.arange(seq_len, device=byte_cache.device)
    physical_pages = block_table[0].index_select(0, logical_tokens // 16).long()
    slot_mapping = physical_pages * 16 + logical_tokens % 16
    key = raw_key.view(-1, 8, 128).index_select(0, slot_mapping)
    value = raw_value.view(-1, 8, 128).index_select(0, slot_mapping)
    num_pages = byte_cache.shape[0]
    workspace_spec = byte_v2_attn_module.ByteV2RawStagingWorkspaceSpec(
        num_blocks=num_pages,
        num_staging_slots=num_pages + 1,
        slot_size_bytes=ByteV2RawStagingLayout().slot_size_bytes,
        device=byte_cache.device,
    )
    manager = byte_v2_attn_module.ByteV2RawStagingManager(
        tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
        num_kv_heads=8,
        raw_fallback_store=object(),
    )
    manager.bind_shared_workspace(workspace_spec.allocate())
    metadata = SimpleNamespace(
        num_actual_tokens=seq_len,
        query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
    )

    staged = manager.stage_initial_prefill(
        key=key,
        value=value,
        kv_cache=byte_cache,
        slot_mapping=slot_mapping,
        attn_metadata=metadata,
    )
    assert staged is not None
    raw_staging, page_map, staging_to_page, valid_rows = staged
    byte_cache.zero_()
    tensors["raw_staging"] = raw_staging
    tensors["page_to_raw_slot"] = page_map

    result = compare_hybrid_byte_v2_and_raw(
        tensors,
        query_len=seq_len,
        iterations=1,
        num_splits=0,
    )
    manager.release_initial_prefill(staging_to_page, valid_rows)
    torch.accelerator.synchronize()

    assert result["out_mismatch"] == 0
    assert result["lse_mismatch"] == 0
    assert result["out_max_abs"] == 0.0
    assert result["lse_max_abs"] == 0.0
    assert bool((page_map == -1).all())


def _run_byte_v2_hybrid_fa2_subprocess(probe: str):
    probe_source = r"""
import sys

import torch
import vllm.vllm_flash_attn  # noqa: F401

from scripts.byte_v2_fa2_oracle import (
    add_hybrid_raw_pages,
    compare_hybrid_byte_v2_and_raw,
    make_inputs,
)
from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayoutV5

probe = sys.argv[1]
tensors = make_inputs((16,), query_len=1, force_outliers=True)
block_table = tensors["block_table"]
byte_cache = tensors["byte_cache"]
assert isinstance(block_table, torch.Tensor)
assert isinstance(byte_cache, torch.Tensor)
raw_page = int(block_table[0, 0].item())
add_hybrid_raw_pages(tensors, [raw_page])

if probe == "invalid-high-slot":
    baseline = compare_hybrid_byte_v2_and_raw(
        tensors,
        query_len=1,
        iterations=1,
        num_splits=1,
    )
    assert baseline["out_mismatch"] == 0
    assert baseline["lse_mismatch"] == 0
    raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    page_to_raw_slot[raw_page] = raw_staging.size(0)
    print({"hybrid_invalid_map_probe": probe}, flush=True)
    compare_hybrid_byte_v2_and_raw(
        tensors,
        query_len=1,
        iterations=1,
        num_splits=1,
    )
    print({"hybrid_invalid_map_unexpected_success": probe}, flush=True)
    raise SystemExit(0)

layout = ByteV2PageLayoutV5()
if probe == "overflow":
    byte_cache[raw_page, layout.outlier_pool_overflow_offset] = 1
elif probe == "fallback":
    for kv_head in range(layout.num_kv_heads):
        byte_cache[
            raw_page,
            layout.k_fallback_mask_offset(kv_head=kv_head),
        ] = 1
        byte_cache[
            raw_page,
            layout.v_fallback_mask_offset(kv_head=kv_head),
        ] = 1
elif probe == "poison":
    byte_cache[raw_page].zero_()
else:
    raise ValueError(f"unknown hybrid probe: {probe}")

result = compare_hybrid_byte_v2_and_raw(
    tensors,
    query_len=1,
    iterations=1,
    num_splits=1,
)
assert result["out_mismatch"] == 0, result
assert result["lse_mismatch"] == 0, result
assert result["out_max_abs"] == 0.0, result
assert result["lse_max_abs"] == 0.0, result
print({"hybrid_recovery_ok": probe}, flush=True)
"""
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [sys.executable, "-c", probe_source, probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("probe", ["overflow", "fallback", "poison"])
def test_byte_v2_hybrid_fa2_raw_page_recovers_in_subprocess(probe):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")

    from scripts.byte_v2_fa2_oracle import hybrid_fa2_oracle_ops_are_available

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")

    completed = _run_byte_v2_hybrid_fa2_subprocess(probe)
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined_output
    assert f"'hybrid_recovery_ok': '{probe}'" in combined_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_fa2_invalid_raw_slot_fails_closed_in_subprocess():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")

    from scripts.byte_v2_fa2_oracle import hybrid_fa2_oracle_ops_are_available

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")

    probe = "invalid-high-slot"
    completed = _run_byte_v2_hybrid_fa2_subprocess(probe)
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "hybrid_invalid_map_probe" in combined_output
    assert "hybrid_invalid_map_unexpected_success" not in combined_output


_BYTE_V2_HYBRID_WRITER_OPS = (
    "byte_v2_prepare_raw_staging",
    "byte_v2_hydrate_raw_staging_from_hybrid_cache",
    "byte_v2_append_raw_staging",
    "byte_v2_commit_raw_staging_to_hybrid_cache",
    "byte_v2_reset_raw_fallback_pages",
    "byte_v2_release_raw_staging",
)


def _byte_v2_hybrid_writer_ops_are_available() -> bool:
    return all(
        _has_torch_op("_C_cache_ops", op_name) for op_name in _BYTE_V2_HYBRID_WRITER_OPS
    )


def _byte_v2_fused_hybrid_writer_q1_op_is_available() -> bool:
    return _byte_v2_hybrid_writer_ops_are_available() and _has_torch_op(
        "_C_cache_ops", "byte_v2_update_hybrid_cache_raw_staging_q1"
    )


def _byte_v2_fused_hybrid_writer_multi_token_op_is_available() -> bool:
    return _byte_v2_hybrid_writer_ops_are_available() and _has_torch_op(
        "_C_cache_ops", "byte_v2_update_hybrid_cache_raw_staging_multi_token"
    )


def _byte_v2_retained_hybrid_writer_multi_token_op_is_available() -> bool:
    return _byte_v2_hybrid_writer_ops_are_available() and _has_torch_op(
        "_C_cache_ops",
        "byte_v2_update_hybrid_cache_raw_staging_multi_token_retained",
    )


def _byte_v2_test_forced_raw_promotion_op_is_available() -> bool:
    return _byte_v2_fused_hybrid_writer_q1_op_is_available() and _has_torch_op(
        "_C_cache_ops", "byte_v2_test_force_promote_raw_staging_q1"
    )


def _make_byte_v2_hybrid_writer_case(
    *,
    seq_len: int = 16,
    query_len: int = 1,
    force_outliers: bool = False,
    force_pool_overflow: bool = False,
    raw_pool_slots: int = 1,
):
    from scripts.byte_v2_fa2_oracle import make_inputs

    tensors = make_inputs(
        (seq_len,), query_len=query_len, force_outliers=force_outliers
    )
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)

    if force_pool_overflow:
        bit_indices = torch.arange(
            raw_key.numel(),
            dtype=torch.int32,
            device=raw_key.device,
        )
        # Sixteen adjacent BF16 high-byte bins overflow the shared outlier
        # pool while keeping every value finite and at most roughly 2.0.
        high7 = (bit_indices % 16) + 48
        lows = bit_indices % 251
        forced_key = ((high7 << 8) | lows).to(torch.int16).view(torch.bfloat16)
        forced_value = (
            (((((bit_indices + 5) % 16) + 48) << 8) | ((lows + 17) % 251))
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        raw_key.copy_(forced_key.reshape_as(raw_key))
        raw_value.copy_(forced_value.reshape_as(raw_value))

    byte_cache.zero_()
    layout = ByteV2RawStagingLayout()
    num_pages = byte_cache.size(0)
    persistent_raw_staging = torch.zeros(
        (raw_pool_slots, layout.slot_size_bytes),
        dtype=torch.uint8,
        device=byte_cache.device,
    )
    page_to_raw_slot = torch.full(
        (num_pages,),
        -1,
        dtype=torch.int32,
        device=byte_cache.device,
    )
    tensors["raw_staging"] = persistent_raw_staging
    tensors["page_to_raw_slot"] = page_to_raw_slot
    state = SimpleNamespace(
        transient_raw_staging=torch.zeros(
            (num_pages, layout.slot_size_bytes),
            dtype=torch.uint8,
            device=byte_cache.device,
        ),
        block_to_staging_slot=torch.full(
            (num_pages,), -1, dtype=torch.int32, device=byte_cache.device
        ),
        staging_to_physical_block=torch.full(
            (num_pages,), -1, dtype=torch.int32, device=byte_cache.device
        ),
        valid_rows=torch.zeros(
            (num_pages,), dtype=torch.int32, device=byte_cache.device
        ),
        next_staging_slot=torch.zeros(
            (1,), dtype=torch.int32, device=byte_cache.device
        ),
        staging_overflow=torch.zeros((1,), dtype=torch.int32, device=byte_cache.device),
        free_raw_slots=torch.arange(
            raw_pool_slots - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=byte_cache.device,
        ),
        free_raw_slot_count=torch.full(
            (1,), raw_pool_slots, dtype=torch.int32, device=byte_cache.device
        ),
        raw_pool_overflow=torch.zeros(
            (1,), dtype=torch.int32, device=byte_cache.device
        ),
    )
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=byte_cache.device)
    return tensors, state, slot_mapping


def _byte_v2_hybrid_writer_update(
    tensors,
    state,
    slot_mapping: torch.Tensor,
    *,
    before_commit=None,
    active_capacity: int | None = None,
    page_unsafe_flags: torch.Tensor | None = None,
) -> None:
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    persistent_raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(persistent_raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)

    flat_key = raw_key.view(-1, 8, 128)
    flat_value = raw_value.view(-1, 8, 128)
    safe_slots = slot_mapping.clamp(min=0, max=flat_key.size(0) - 1)
    key = flat_key.index_select(0, safe_slots)
    value = flat_value.index_select(0, safe_slots)
    if active_capacity is None:
        active_capacity = state.transient_raw_staging.size(0)
    assert 0 < active_capacity <= state.transient_raw_staging.size(0)
    raw_staging = state.transient_raw_staging[:active_capacity]
    staging_to_physical_block = state.staging_to_physical_block[:active_capacity]
    valid_rows = state.valid_rows[:active_capacity]
    ops = torch.ops._C_cache_ops
    ops.byte_v2_prepare_raw_staging(
        slot_mapping,
        state.block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        state.next_staging_slot,
        state.staging_overflow,
        16,
    )
    ops.byte_v2_hydrate_raw_staging_from_hybrid_cache(
        raw_staging,
        byte_cache,
        persistent_raw_staging,
        page_to_raw_slot,
        staging_to_physical_block,
        valid_rows,
        16,
        16,
        16,
    )
    ops.byte_v2_append_raw_staging(
        key,
        value,
        raw_staging,
        slot_mapping,
        state.block_to_staging_slot,
        16,
        16,
        16,
    )
    if before_commit is not None:
        before_commit()
    ops.byte_v2_commit_raw_staging_to_hybrid_cache(
        raw_staging,
        byte_cache,
        persistent_raw_staging,
        page_to_raw_slot,
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        staging_to_physical_block,
        valid_rows,
        16,
        16,
        16,
    )
    if page_unsafe_flags is None:
        ops.byte_v2_release_raw_staging(
            state.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            state.next_staging_slot,
            state.staging_overflow,
        )
    else:
        byte_v2_release_raw_staging_and_update_flags(
            state.block_to_staging_slot,
            staging_to_physical_block,
            valid_rows,
            state.next_staging_slot,
            state.staging_overflow,
            page_unsafe_flags,
            byte_cache,
            tile_policy=(16, 16, 16, 64, 128, 128),
        )


def _byte_v2_fused_hybrid_writer_multi_token_update(
    tensors,
    state,
    slot_mapping: torch.Tensor,
    *,
    active_capacity: int | None = None,
    page_unsafe_flags: torch.Tensor | None = None,
    retain: bool = False,
) -> None:
    assert slot_mapping.numel() > 1
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    persistent_raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(persistent_raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)

    flat_key = raw_key.view(-1, 8, 128)
    flat_value = raw_value.view(-1, 8, 128)
    safe_slots = slot_mapping.clamp(min=0, max=flat_key.size(0) - 1)
    key = flat_key.index_select(0, safe_slots)
    value = flat_value.index_select(0, safe_slots)
    if active_capacity is None:
        active_capacity = state.transient_raw_staging.size(0)
    assert 0 < active_capacity <= state.transient_raw_staging.size(0)
    update_op = (
        byte_v2_update_hybrid_cache_raw_staging_multi_token_retained
        if retain
        else byte_v2_update_hybrid_cache_raw_staging_multi_token
    )
    update_op(
        key,
        value,
        state.transient_raw_staging[:active_capacity],
        byte_cache,
        persistent_raw_staging,
        slot_mapping,
        state.block_to_staging_slot,
        state.staging_to_physical_block[:active_capacity],
        state.valid_rows[:active_capacity],
        state.next_staging_slot,
        state.staging_overflow,
        page_to_raw_slot,
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        tile_policy=(16, 16, 16, 64, 128, 128),
        page_unsafe_flags=page_unsafe_flags,
    )


def _clone_byte_v2_hybrid_writer_case(tensors, state):
    cloned_tensors = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in tensors.items()
    }
    cloned_state = SimpleNamespace(
        **{
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in vars(state).items()
        }
    )
    return cloned_tensors, cloned_state


def _byte_v2_fused_hybrid_writer_q1_update(
    tensors,
    state,
    slot_mapping: torch.Tensor,
    *,
    page_unsafe_flags: torch.Tensor | None = None,
) -> None:
    assert slot_mapping.shape == (1,)
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    persistent_raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(persistent_raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)

    flat_key = raw_key.view(-1, 8, 128)
    flat_value = raw_value.view(-1, 8, 128)
    if int(slot_mapping[0].cpu().item()) < 0:
        key = torch.zeros_like(flat_key[:1])
        value = torch.zeros_like(flat_value[:1])
    else:
        key = flat_key.index_select(0, slot_mapping)
        value = flat_value.index_select(0, slot_mapping)
    byte_v2_update_hybrid_cache_raw_staging_q1(
        key,
        value,
        state.transient_raw_staging[:1],
        byte_cache,
        persistent_raw_staging,
        slot_mapping,
        state.block_to_staging_slot,
        state.staging_to_physical_block[:1],
        state.valid_rows[:1],
        state.next_staging_slot,
        state.staging_overflow,
        page_to_raw_slot,
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        tile_policy=(16, 16, 16, 64, 128, 128),
        page_unsafe_flags=page_unsafe_flags,
    )


def _byte_v2_test_force_promote_hybrid_q1(
    tensors,
    state,
    slot_mapping: torch.Tensor,
    diagnostic: torch.Tensor,
) -> None:
    byte_v2_test_force_promote_raw_staging_q1(
        state.transient_raw_staging[:1],
        tensors["raw_staging"],
        slot_mapping,
        tensors["page_to_raw_slot"],
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        diagnostic,
    )


def _byte_v2_hybrid_writer_reset(tensors, state, physical_block_ids) -> None:
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(page_to_raw_slot, torch.Tensor)
    torch.ops._C_cache_ops.byte_v2_reset_raw_fallback_pages(
        page_to_raw_slot,
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        physical_block_ids,
    )


def _assert_byte_v2_hybrid_writer_matches_raw(tensors) -> None:
    from scripts.byte_v2_fa2_oracle import compare_hybrid_byte_v2_and_raw

    result = compare_hybrid_byte_v2_and_raw(
        tensors,
        query_len=1,
        iterations=1,
        num_splits=1,
    )
    assert result["out_mismatch"] == 0, result
    assert result["lse_mismatch"] == 0, result
    assert result["out_max_abs"] == 0.0, result
    assert result["lse_max_abs"] == 0.0, result


def _assert_byte_v2_hybrid_writer_transient_state_is_clean(state) -> None:
    torch.testing.assert_close(
        state.block_to_staging_slot,
        torch.full_like(state.block_to_staging_slot, -1),
    )
    torch.testing.assert_close(
        state.staging_to_physical_block,
        torch.full_like(state.staging_to_physical_block, -1),
    )
    torch.testing.assert_close(state.valid_rows, torch.zeros_like(state.valid_rows))
    assert int(state.next_staging_slot.item()) == 0
    assert int(state.staging_overflow.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0


def _byte_v2_hybrid_writer_allocator_partition(tensors, state):
    page_to_raw_slot = tensors["page_to_raw_slot"].cpu().tolist()
    raw_pool_slots = tensors["raw_staging"].size(0)
    free_count = int(state.free_raw_slot_count.item())
    assert 0 <= free_count <= raw_pool_slots
    free_slots = state.free_raw_slots[:free_count].cpu().tolist()
    mapped_slots = [raw_slot for raw_slot in page_to_raw_slot if raw_slot >= 0]
    assert sorted(free_slots + mapped_slots) == list(range(raw_pool_slots))
    return page_to_raw_slot, free_slots


def _assert_byte_v2_hybrid_writers_are_canonically_equal(
    reference_tensors,
    reference_state,
    candidate_tensors,
    candidate_state,
    reference_flags: torch.Tensor,
    candidate_flags: torch.Tensor,
) -> None:
    reference_map, reference_free = _byte_v2_hybrid_writer_allocator_partition(
        reference_tensors, reference_state
    )
    candidate_map, candidate_free = _byte_v2_hybrid_writer_allocator_partition(
        candidate_tensors, candidate_state
    )
    assert [raw_slot >= 0 for raw_slot in reference_map] == [
        raw_slot >= 0 for raw_slot in candidate_map
    ]
    assert len(reference_free) == len(candidate_free)
    assert int(reference_state.free_raw_slot_count.item()) == int(
        candidate_state.free_raw_slot_count.item()
    )
    reference_raw = reference_tensors["raw_staging"]
    candidate_raw = candidate_tensors["raw_staging"]
    for physical_block, reference_raw_slot in enumerate(reference_map):
        if reference_raw_slot < 0:
            continue
        candidate_raw_slot = candidate_map[physical_block]
        torch.testing.assert_close(
            reference_raw[reference_raw_slot],
            candidate_raw[candidate_raw_slot],
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(reference_flags, candidate_flags, atol=0, rtol=0)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(reference_state)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(candidate_state)


def _byte_v2_q34_partial_two_page_mapping(device: torch.device) -> torch.Tensor:
    slot_mapping = torch.full((34,), -1, dtype=torch.int64, device=device)
    # The two physical pages recur in separate 16-token owner chunks, and each
    # page receives non-contiguous rows in a deliberately permuted order.
    slot_mapping[0] = 31
    slot_mapping[7] = 3
    slot_mapping[18] = 16
    slot_mapping[33] = 15
    return slot_mapping


def _byte_v2_q34_full_two_page_mapping(device: torch.device) -> torch.Tensor:
    page_rows = torch.arange(16, dtype=torch.int64, device=device)
    interleaved = torch.stack((page_rows + 16, page_rows), dim=1).flatten()
    return torch.cat(
        (interleaved[:9], interleaved.new_tensor([-1, -1]), interleaved[9:])
    )


def _require_byte_v2_hybrid_writer_reader_ops() -> None:
    if not _byte_v2_hybrid_writer_ops_are_available():
        pytest.skip("ByteV2 hybrid writer custom ops are not registered")

    from scripts.byte_v2_fa2_oracle import hybrid_fa2_oracle_ops_are_available

    if not hybrid_fa2_oracle_ops_are_available():
        pytest.skip("Hybrid ByteV2 and raw FA2 extension ops are not registered")


def _require_byte_v2_fused_hybrid_writer_reader_ops() -> None:
    if not _byte_v2_fused_hybrid_writer_q1_op_is_available():
        pytest.skip("ByteV2 fused hybrid Q1 writer op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()


def _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops() -> None:
    if not _byte_v2_fused_hybrid_writer_multi_token_op_is_available():
        pytest.skip("ByteV2 fused hybrid multi-token writer op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [73, 2925])
def test_byte_v2_retained_initial_prefill_matches_raw_fa2_bitwise(seq_len):
    if not _byte_v2_retained_hybrid_writer_multi_token_op_is_available():
        pytest.skip("ByteV2 retained hybrid writer op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=seq_len,
        query_len=seq_len,
        force_outliers=True,
        raw_pool_slots=max(1, (seq_len + 15) // 16),
    )
    active_capacity = (seq_len + 15) // 16
    _byte_v2_fused_hybrid_writer_multi_token_update(
        tensors,
        state,
        slot_mapping,
        active_capacity=active_capacity,
        retain=True,
    )

    query = tensors["query"]
    staged_output = torch.empty_like(query)
    raw_output = torch.empty_like(query)
    byte_v2_ops_module.byte_v2_fa2_raw_staging_prefill_attention(
        staged_output,
        query,
        state.transient_raw_staging[:active_capacity],
        state.block_to_staging_slot,
        tensors["cu_seqlens_q"],
        tensors["block_table"],
        tensors["seq_lens"],
        scale=float(tensors["scale"]),
        num_kv_heads=8,
        block_size=16,
        head_dim=128,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        causal=True,
    )
    torch.ops._vllm_fa2_C.varlen_fwd(
        query,
        tensors["key"],
        tensors["value"],
        raw_output,
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        seq_len,
        seq_len,
        0.0,
        float(tensors["scale"]),
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )
    byte_v2_release_raw_staging(
        state.block_to_staging_slot,
        state.staging_to_physical_block[:active_capacity],
        state.valid_rows[:active_capacity],
        state.next_staging_slot,
        state.staging_overflow,
    )
    torch.accelerator.synchronize()

    assert (
        int((staged_output.view(torch.int16) != raw_output.view(torch.int16)).sum())
        == 0
    )
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_retained_initial_prefill_cuda_graph_replay():
    if not _byte_v2_retained_hybrid_writer_multi_token_op_is_available():
        pytest.skip("ByteV2 retained hybrid writer op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()
    seq_len = 73
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=seq_len,
        query_len=seq_len,
        force_outliers=True,
        raw_pool_slots=(seq_len + 15) // 16,
    )
    active_capacity = (seq_len + 15) // 16
    flat_key = tensors["key"].view(-1, 8, 128)
    flat_value = tensors["value"].view(-1, 8, 128)
    key = flat_key.index_select(0, slot_mapping)
    value = flat_value.index_select(0, slot_mapping)
    query = tensors["query"]
    staged_output = torch.empty_like(query)
    raw_output = torch.empty_like(query)

    def run_retained_prefill():
        byte_v2_update_hybrid_cache_raw_staging_multi_token_retained(
            key,
            value,
            state.transient_raw_staging[:active_capacity],
            tensors["byte_cache"],
            tensors["raw_staging"],
            slot_mapping,
            state.block_to_staging_slot,
            state.staging_to_physical_block[:active_capacity],
            state.valid_rows[:active_capacity],
            state.next_staging_slot,
            state.staging_overflow,
            tensors["page_to_raw_slot"],
            state.free_raw_slots,
            state.free_raw_slot_count,
            state.raw_pool_overflow,
            tile_policy=(16, 16, 16, 64, 128, 128),
        )
        byte_v2_ops_module.byte_v2_fa2_raw_staging_prefill_attention(
            staged_output,
            query,
            state.transient_raw_staging[:active_capacity],
            state.block_to_staging_slot,
            tensors["cu_seqlens_q"],
            tensors["block_table"],
            tensors["seq_lens"],
            scale=float(tensors["scale"]),
            num_kv_heads=8,
            block_size=16,
            head_dim=128,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            causal=True,
        )
        byte_v2_release_raw_staging(
            state.block_to_staging_slot,
            state.staging_to_physical_block[:active_capacity],
            state.valid_rows[:active_capacity],
            state.next_staging_slot,
            state.staging_overflow,
        )

    run_retained_prefill()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_retained_prefill()
    for _ in range(2):
        graph.replay()
    torch.ops._vllm_fa2_C.varlen_fwd(
        query,
        tensors["key"],
        tensors["value"],
        raw_output,
        tensors["cu_seqlens_q"],
        tensors["dummy_cu_seqlens_k"],
        tensors["seq_lens"],
        None,
        tensors["block_table"],
        None,
        seq_len,
        seq_len,
        0.0,
        float(tensors["scale"]),
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )
    torch.accelerator.synchronize()

    assert (
        int((staged_output.view(torch.int16) != raw_output.view(torch.int16)).sum())
        == 0
    )
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source", ["compact", "existing-raw"])
def test_byte_v2_fused_hybrid_multi_token_q34_matches_generic_bitwise(source):
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    tensors, state, full_slots = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        force_outliers=source == "compact",
        force_pool_overflow=source == "existing-raw",
        raw_pool_slots=2,
    )
    _byte_v2_hybrid_writer_update(
        tensors,
        state,
        full_slots,
        active_capacity=2,
    )
    torch.accelerator.synchronize()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    if source == "compact":
        assert bool((page_to_raw_slot == -1).all())
    else:
        assert bool((page_to_raw_slot >= 0).all())
        # Only the persistent raw pages remain authoritative for hydration.
        tensors["byte_cache"].zero_()

    slot_mapping = _byte_v2_q34_partial_two_page_mapping(tensors["byte_cache"].device)
    updated_slots = slot_mapping[slot_mapping >= 0]
    for tensor_name in ("key", "value"):
        flat = tensors[tensor_name].view(-1, 8, 128)
        flat.index_copy_(0, updated_slots, -flat.index_select(0, updated_slots))

    reference_tensors, reference_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    candidate_tensors, candidate_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    reference_flags = torch.zeros(
        (tensors["byte_cache"].size(0),),
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    candidate_flags = torch.zeros_like(reference_flags)

    _byte_v2_hybrid_writer_update(
        reference_tensors,
        reference_state,
        slot_mapping,
        active_capacity=2,
        page_unsafe_flags=reference_flags,
    )
    _byte_v2_fused_hybrid_writer_multi_token_update(
        candidate_tensors,
        candidate_state,
        slot_mapping,
        active_capacity=2,
        page_unsafe_flags=candidate_flags,
    )
    torch.accelerator.synchronize()

    _assert_byte_v2_hybrid_writer_matches_raw(reference_tensors)
    _assert_byte_v2_hybrid_writer_matches_raw(candidate_tensors)
    _assert_byte_v2_hybrid_writers_are_canonically_equal(
        reference_tensors,
        reference_state,
        candidate_tensors,
        candidate_state,
        reference_flags,
        candidate_flags,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source", ["compact", "existing-raw"])
def test_byte_v2_fused_hybrid_small_batch_matches_generic_bitwise(source):
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    tensors, state, full_slots = _make_byte_v2_hybrid_writer_case(
        seq_len=64,
        force_outliers=source == "compact",
        force_pool_overflow=source == "existing-raw",
        raw_pool_slots=4,
    )
    _byte_v2_hybrid_writer_update(
        tensors,
        state,
        full_slots,
        active_capacity=4,
    )
    torch.accelerator.synchronize()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    if source == "compact":
        assert bool((page_to_raw_slot == -1).all())
    else:
        assert bool((page_to_raw_slot >= 0).all())
        tensors["byte_cache"].zero_()

    # Eight entries touch four pages in a non-monotonic order. Pages 0 and 1
    # each appear twice, which verifies that the page-parallel fast path still
    # elects one owner and merges every updated row. Each touched page includes
    # row 15 so the update satisfies the writer's append-prefix contract.
    slot_mapping = torch.tensor(
        [3, 31, -1, 15, 47, 20, 63, -1],
        dtype=torch.int64,
        device=tensors["byte_cache"].device,
    )
    updated_slots = slot_mapping[slot_mapping >= 0]
    for tensor_name in ("key", "value"):
        flat = tensors[tensor_name].view(-1, 8, 128)
        flat.index_copy_(0, updated_slots, -flat.index_select(0, updated_slots))

    reference_tensors, reference_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    candidate_tensors, candidate_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    reference_flags = torch.zeros(
        (tensors["byte_cache"].size(0),),
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    candidate_flags = torch.zeros_like(reference_flags)

    _byte_v2_hybrid_writer_update(
        reference_tensors,
        reference_state,
        slot_mapping,
        active_capacity=4,
        page_unsafe_flags=reference_flags,
    )
    _byte_v2_fused_hybrid_writer_multi_token_update(
        candidate_tensors,
        candidate_state,
        slot_mapping,
        active_capacity=4,
        page_unsafe_flags=candidate_flags,
    )
    torch.accelerator.synchronize()

    _assert_byte_v2_hybrid_writer_matches_raw(reference_tensors)
    _assert_byte_v2_hybrid_writer_matches_raw(candidate_tensors)
    _assert_byte_v2_hybrid_writers_are_canonically_equal(
        reference_tensors,
        reference_state,
        candidate_tensors,
        candidate_state,
        reference_flags,
        candidate_flags,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_pages", [2, 3, 4])
def test_byte_v2_fused_hybrid_multi_token_adaptive_ctas_are_bitwise(
    num_pages,
):
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    tensors, state, full_slots = _make_byte_v2_hybrid_writer_case(
        seq_len=64,
        force_outliers=True,
        raw_pool_slots=4,
    )
    _byte_v2_hybrid_writer_update(
        tensors,
        state,
        full_slots,
        active_capacity=4,
    )
    torch.accelerator.synchronize()
    assert bool((tensors["page_to_raw_slot"] == -1).all())

    slot_mapping = (
        torch.arange(num_pages, dtype=torch.int64, device=full_slots.device) * 16 + 15
    )
    for tensor_name in ("key", "value"):
        flat = tensors[tensor_name].view(-1, 8, 128)
        flat.index_copy_(
            0,
            slot_mapping,
            -flat.index_select(0, slot_mapping),
        )

    reference_tensors, reference_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    candidate_tensors, candidate_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    reference_flags = torch.zeros(
        (tensors["byte_cache"].size(0),),
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    candidate_flags = torch.zeros_like(reference_flags)

    _byte_v2_hybrid_writer_update(
        reference_tensors,
        reference_state,
        slot_mapping,
        active_capacity=num_pages,
        page_unsafe_flags=reference_flags,
    )
    _byte_v2_fused_hybrid_writer_multi_token_update(
        candidate_tensors,
        candidate_state,
        slot_mapping,
        active_capacity=num_pages,
        page_unsafe_flags=candidate_flags,
    )
    torch.accelerator.synchronize()

    _assert_byte_v2_hybrid_writer_matches_raw(reference_tensors)
    _assert_byte_v2_hybrid_writer_matches_raw(candidate_tensors)
    _assert_byte_v2_hybrid_writers_are_canonically_equal(
        reference_tensors,
        reference_state,
        candidate_tensors,
        candidate_state,
        reference_flags,
        candidate_flags,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [16, 17])
def test_byte_v2_fused_hybrid_multi_token_dispatch_boundary_is_bitwise(
    num_tokens,
):
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    tensors, state, full_slots = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        force_outliers=True,
        raw_pool_slots=2,
    )
    _byte_v2_hybrid_writer_update(
        tensors,
        state,
        full_slots,
        active_capacity=2,
    )
    torch.accelerator.synchronize()
    assert bool((tensors["page_to_raw_slot"] == -1).all())

    slot_mapping = torch.arange(16, dtype=torch.int64, device=full_slots.device)
    if num_tokens == 17:
        slot_mapping = torch.cat((slot_mapping, slot_mapping.new_tensor([31])))
    updated_slots = slot_mapping[slot_mapping >= 0]
    for tensor_name in ("key", "value"):
        flat = tensors[tensor_name].view(-1, 8, 128)
        flat.index_copy_(0, updated_slots, -flat.index_select(0, updated_slots))

    reference_tensors, reference_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    candidate_tensors, candidate_state = _clone_byte_v2_hybrid_writer_case(
        tensors, state
    )
    reference_flags = torch.zeros(
        (tensors["byte_cache"].size(0),),
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    candidate_flags = torch.zeros_like(reference_flags)

    _byte_v2_hybrid_writer_update(
        reference_tensors,
        reference_state,
        slot_mapping,
        active_capacity=2,
        page_unsafe_flags=reference_flags,
    )
    _byte_v2_fused_hybrid_writer_multi_token_update(
        candidate_tensors,
        candidate_state,
        slot_mapping,
        active_capacity=2,
        page_unsafe_flags=candidate_flags,
    )
    torch.accelerator.synchronize()

    _assert_byte_v2_hybrid_writer_matches_raw(reference_tensors)
    _assert_byte_v2_hybrid_writer_matches_raw(candidate_tensors)
    _assert_byte_v2_hybrid_writers_are_canonically_equal(
        reference_tensors,
        reference_state,
        candidate_tensors,
        candidate_state,
        reference_flags,
        candidate_flags,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_multi_token_all_dummy_q34_is_noop():
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    tensors, state, _ = _make_byte_v2_hybrid_writer_case(seq_len=32, raw_pool_slots=2)
    slot_mapping = torch.full(
        (34,), -1, dtype=torch.int64, device=tensors["byte_cache"].device
    )
    page_flags = torch.full(
        (tensors["byte_cache"].size(0),),
        7,
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    watched = (
        tensors["byte_cache"],
        tensors["raw_staging"],
        tensors["page_to_raw_slot"],
        state.free_raw_slots,
        state.free_raw_slot_count,
        page_flags,
    )
    snapshots = [tensor.clone() for tensor in watched]

    _byte_v2_fused_hybrid_writer_multi_token_update(
        tensors,
        state,
        slot_mapping,
        active_capacity=2,
        page_unsafe_flags=page_flags,
    )
    torch.accelerator.synchronize()

    for tensor, snapshot in zip(watched, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [16, 34])
def test_byte_v2_fused_hybrid_multi_token_cuda_graph_reuses_workspace(
    num_tokens,
):
    _require_byte_v2_fused_multi_token_hybrid_writer_reader_ops()
    seq_len = 16 if num_tokens == 16 else 32
    active_capacity = 1 if num_tokens == 16 else 2
    tensors, state, _ = _make_byte_v2_hybrid_writer_case(
        seq_len=seq_len,
        force_pool_overflow=True,
        raw_pool_slots=active_capacity,
    )
    flat_key = tensors["key"].view(-1, 8, 128)
    flat_value = tensors["value"].view(-1, 8, 128)
    graph_key = torch.zeros(
        (num_tokens, 8, 128), dtype=torch.bfloat16, device=flat_key.device
    )
    graph_value = torch.zeros_like(graph_key)
    graph_slot = torch.full(
        (num_tokens,), -1, dtype=torch.int64, device=flat_key.device
    )
    page_flags = torch.zeros(
        (tensors["byte_cache"].size(0),),
        dtype=torch.int32,
        device=flat_key.device,
    )

    def update() -> None:
        byte_v2_update_hybrid_cache_raw_staging_multi_token(
            graph_key,
            graph_value,
            state.transient_raw_staging[:active_capacity],
            tensors["byte_cache"],
            tensors["raw_staging"],
            graph_slot,
            state.block_to_staging_slot,
            state.staging_to_physical_block[:active_capacity],
            state.valid_rows[:active_capacity],
            state.next_staging_slot,
            state.staging_overflow,
            tensors["page_to_raw_slot"],
            state.free_raw_slots,
            state.free_raw_slot_count,
            state.raw_pool_overflow,
            tile_policy=(16, 16, 16, 64, 128, 128),
            page_unsafe_flags=page_flags,
        )

    update()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()

    actual_slots = (
        torch.arange(16, dtype=torch.int64, device=flat_key.device)
        if num_tokens == 16
        else _byte_v2_q34_full_two_page_mapping(flat_key.device)
    )
    safe_slots = actual_slots.clamp_min(0)
    graph_key.copy_(flat_key.index_select(0, safe_slots))
    graph_value.copy_(flat_value.index_select(0, safe_slots))
    graph_slot.copy_(actual_slots)
    graph.replay()
    torch.accelerator.synchronize()
    assert bool((tensors["page_to_raw_slot"] >= 0).all())
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)

    _byte_v2_hybrid_writer_reset(
        tensors,
        state,
        torch.arange(active_capacity, dtype=torch.int32, device=flat_key.device),
    )
    tensors["byte_cache"].zero_()
    page_flags.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert bool((tensors["page_to_raw_slot"] >= 0).all())
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)

    watched = (
        tensors["byte_cache"],
        tensors["raw_staging"],
        tensors["page_to_raw_slot"],
        state.free_raw_slots,
        state.free_raw_slot_count,
        page_flags,
    )
    snapshots = [tensor.clone() for tensor in watched]
    graph_key.zero_()
    graph_value.zero_()
    graph_slot.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()
    for tensor, snapshot in zip(watched, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
    _assert_byte_v2_hybrid_writer_transient_state_is_clean(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_q1_dummy_slot_is_noop():
    if not _byte_v2_fused_hybrid_writer_q1_op_is_available():
        pytest.skip("ByteV2 fused hybrid Q1 writer op is not registered")
    tensors, state, _ = _make_byte_v2_hybrid_writer_case()
    page_flags = torch.full(
        (tensors["byte_cache"].shape[0],),
        7,
        dtype=torch.int32,
        device=tensors["byte_cache"].device,
    )
    watched = (
        tensors["byte_cache"],
        tensors["raw_staging"],
        tensors["page_to_raw_slot"],
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        page_flags,
    )
    snapshots = [tensor.clone() for tensor in watched]

    _byte_v2_fused_hybrid_writer_q1_update(
        tensors,
        state,
        torch.tensor([-1], dtype=torch.int64, device=tensors["byte_cache"].device),
        page_unsafe_flags=page_flags,
    )
    torch.accelerator.synchronize()

    for tensor, snapshot in zip(watched, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
    torch.testing.assert_close(
        state.block_to_staging_slot,
        torch.full_like(state.block_to_staging_slot, -1),
    )
    torch.testing.assert_close(
        state.staging_to_physical_block,
        torch.full_like(state.staging_to_physical_block, -1),
    )
    torch.testing.assert_close(state.valid_rows, torch.zeros_like(state.valid_rows))
    assert int(state.next_staging_slot.item()) == 0
    assert int(state.staging_overflow.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_q1_ordinary_outliers_remain_compact_bitwise():
    _require_byte_v2_fused_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(force_outliers=True)

    for slot in slot_mapping.split(1):
        _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
    torch.accelerator.synchronize()

    cache = tensors["byte_cache"].cpu()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    layout = ByteV2PageLayoutV5()
    stats = collect_byte_v2_fallback_stats(cache, layout=layout)
    assert stats.overlay_tiles > 0
    assert stats.fallback_tiles == 0
    assert _load_u32_bytes(cache, 0, layout.outlier_pool_overflow_offset) == 0
    assert bool((page_to_raw_slot == -1).all())
    assert int(state.free_raw_slot_count.item()) == 1
    assert int(state.raw_pool_overflow.item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_q1_pool_overflow_publishes_raw_bitwise():
    _require_byte_v2_fused_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        force_pool_overflow=True
    )

    for slot in slot_mapping.split(1):
        _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
    torch.accelerator.synchronize()

    cache = tensors["byte_cache"].cpu()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    layout = ByteV2PageLayoutV5()
    assert _load_u32_bytes(cache, 0, layout.outlier_pool_overflow_offset) != 0
    assert int(page_to_raw_slot[0].item()) >= 0
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_q1_updates_existing_raw_page_bitwise():
    _require_byte_v2_fused_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        force_pool_overflow=True
    )

    for slot in slot_mapping[:15].split(1):
        _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
    torch.accelerator.synchronize()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    raw_slot = int(page_to_raw_slot[0].item())
    assert raw_slot >= 0

    # The final update must hydrate from the authoritative raw page, not the
    # deliberately corrupted compact representation.
    tensors["byte_cache"][0].zero_()
    _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[15:])
    torch.accelerator.synchronize()

    assert int(page_to_raw_slot[0].item()) == raw_slot
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    persistent_bf16 = (
        tensors["raw_staging"].view(torch.bfloat16).reshape(-1, 2, 8, 16, 128)
    )
    torch.testing.assert_close(
        persistent_bf16[raw_slot, 0].permute(1, 0, 2),
        tensors["key"][0],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        persistent_bf16[raw_slot, 1].permute(1, 0, 2),
        tensors["value"][0],
        atol=0,
        rtol=0,
    )
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_hybrid_q1_cuda_graph_reset_reuses_raw_slot():
    _require_byte_v2_fused_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        force_pool_overflow=True,
        raw_pool_slots=2,
    )
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    persistent_raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(persistent_raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)

    flat_key = raw_key.view(-1, 8, 128)
    flat_value = raw_value.view(-1, 8, 128)
    graph_key = torch.zeros_like(flat_key[:1])
    graph_value = torch.zeros_like(flat_value[:1])
    graph_slot = torch.full((1,), -1, dtype=torch.int64, device=byte_cache.device)
    page_flags = torch.zeros(
        (byte_cache.size(0),), dtype=torch.int32, device=byte_cache.device
    )

    def update() -> None:
        byte_v2_update_hybrid_cache_raw_staging_q1(
            graph_key,
            graph_value,
            state.transient_raw_staging[:1],
            byte_cache,
            persistent_raw_staging,
            graph_slot,
            state.block_to_staging_slot,
            state.staging_to_physical_block[:1],
            state.valid_rows[:1],
            state.next_staging_slot,
            state.staging_overflow,
            page_to_raw_slot,
            state.free_raw_slots,
            state.free_raw_slot_count,
            state.raw_pool_overflow,
            tile_policy=(16, 16, 16, 64, 128, 128),
            page_unsafe_flags=page_flags,
        )

    # Warm up and capture with a dummy slot so capture itself cannot mutate a
    # physical page or consume a raw-pool slot.
    update()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()

    def replay_slots(slots: torch.Tensor) -> None:
        for slot in slots:
            token_idx = int(slot.cpu().item())
            graph_key.copy_(flat_key[token_idx : token_idx + 1])
            graph_value.copy_(flat_value[token_idx : token_idx + 1])
            graph_slot.fill_(token_idx)
            graph.replay()

    replay_slots(slot_mapping)
    torch.accelerator.synchronize()

    initial_raw_slots = page_to_raw_slot.cpu().tolist()
    assert sorted(initial_raw_slots) == [0, 1]
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)

    released_raw_slot = initial_raw_slots[0]
    retained_raw_slot = initial_raw_slots[1]
    _byte_v2_hybrid_writer_reset(
        tensors,
        state,
        torch.tensor([0], dtype=torch.int32, device=byte_cache.device),
    )
    byte_cache[0].zero_()
    torch.accelerator.synchronize()
    assert page_to_raw_slot.cpu().tolist() == [-1, retained_raw_slot]
    assert int(state.free_raw_slot_count.item()) == 1

    # Replay the already captured graph to rebuild page zero. Its overflow must
    # reuse exactly the slot released by reset while page one remains mapped.
    replay_slots(slot_mapping[:16])
    torch.accelerator.synchronize()
    assert page_to_raw_slot.cpu().tolist() == [
        released_raw_slot,
        retained_raw_slot,
    ]
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)

    watched = (
        byte_cache,
        persistent_raw_staging,
        page_to_raw_slot,
        state.free_raw_slots,
        state.free_raw_slot_count,
        state.raw_pool_overflow,
        page_flags,
    )
    snapshots = [tensor.clone() for tensor in watched]
    graph_key.zero_()
    graph_value.zero_()
    graph_slot.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()

    for tensor, snapshot in zip(watched, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
    torch.testing.assert_close(
        state.block_to_staging_slot,
        torch.full_like(state.block_to_staging_slot, -1),
    )
    torch.testing.assert_close(
        state.staging_to_physical_block,
        torch.full_like(state.staging_to_physical_block, -1),
    )
    torch.testing.assert_close(state.valid_rows, torch.zeros_like(state.valid_rows))
    assert int(state.next_staging_slot.item()) == 0
    assert int(state.staging_overflow.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_test_forced_raw_q1_eager_bitwise_reset_reuses_slot():
    if not _byte_v2_test_forced_raw_promotion_op_is_available():
        pytest.skip("ByteV2 test forced-raw promotion op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=2,
        raw_pool_slots=1,
    )
    byte_cache = tensors["byte_cache"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    diagnostic = torch.tensor([1, 0, 0], dtype=torch.int32, device=byte_cache.device)

    _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[:1])
    _byte_v2_test_force_promote_hybrid_q1(tensors, state, slot_mapping[:1], diagnostic)
    torch.accelerator.synchronize()
    raw_slot = int(page_to_raw_slot[0].item())
    assert raw_slot == 0
    assert diagnostic.cpu().tolist() == [0, 1, 0]
    assert int(state.free_raw_slot_count.item()) == 0

    # Make compact unusable so the next fused update must hydrate from the
    # forced raw mapping. The diagnostic op then observes the mapped visit.
    byte_cache[0].zero_()
    _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[1:])
    _byte_v2_test_force_promote_hybrid_q1(tensors, state, slot_mapping[1:], diagnostic)
    torch.accelerator.synchronize()
    assert diagnostic.cpu().tolist() == [0, 1, 1]
    assert int(page_to_raw_slot[0].item()) == raw_slot
    assert int(state.raw_pool_overflow.item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)

    _byte_v2_hybrid_writer_reset(
        tensors,
        state,
        torch.tensor([0], dtype=torch.int32, device=byte_cache.device),
    )
    byte_cache[0].zero_()
    diagnostic[0].fill_(1)
    torch.accelerator.synchronize()
    assert int(page_to_raw_slot[0].item()) == -1
    assert int(state.free_raw_slot_count.item()) == 1
    assert diagnostic.cpu().tolist() == [1, 1, 1]

    for slot in slot_mapping.split(1):
        _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
        _byte_v2_test_force_promote_hybrid_q1(tensors, state, slot, diagnostic)
    torch.accelerator.synchronize()
    assert int(page_to_raw_slot[0].item()) == raw_slot
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    assert diagnostic.cpu().tolist() == [0, 2, 2]
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_test_forced_raw_q1_cuda_graph_latch_and_reset():
    if not _byte_v2_test_forced_raw_promotion_op_is_available():
        pytest.skip("ByteV2 test forced-raw promotion op is not registered")
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=2,
        raw_pool_slots=1,
    )
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    byte_cache = tensors["byte_cache"]
    persistent_raw_staging = tensors["raw_staging"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(persistent_raw_staging, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    flat_key = raw_key.view(-1, 8, 128)
    flat_value = raw_value.view(-1, 8, 128)
    graph_key = torch.zeros_like(flat_key[:1])
    graph_value = torch.zeros_like(flat_value[:1])
    graph_slot = torch.full((1,), -1, dtype=torch.int64, device=byte_cache.device)
    diagnostic = torch.zeros((3,), dtype=torch.int32, device=byte_cache.device)

    def update() -> None:
        byte_v2_update_hybrid_cache_raw_staging_q1(
            graph_key,
            graph_value,
            state.transient_raw_staging[:1],
            byte_cache,
            persistent_raw_staging,
            graph_slot,
            state.block_to_staging_slot,
            state.staging_to_physical_block[:1],
            state.valid_rows[:1],
            state.next_staging_slot,
            state.staging_overflow,
            page_to_raw_slot,
            state.free_raw_slots,
            state.free_raw_slot_count,
            state.raw_pool_overflow,
            tile_policy=(16, 16, 16, 64, 128, 128),
        )
        byte_v2_test_force_promote_raw_staging_q1(
            state.transient_raw_staging[:1],
            persistent_raw_staging,
            graph_slot,
            page_to_raw_slot,
            state.free_raw_slots,
            state.free_raw_slot_count,
            state.raw_pool_overflow,
            diagnostic,
        )

    update()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()

    # Capture and a subsequent dummy replay leave the armed one-shot intact.
    diagnostic[0].fill_(1)
    graph.replay()
    torch.accelerator.synchronize()
    assert diagnostic.cpu().tolist() == [1, 0, 0]
    assert int(page_to_raw_slot[0].item()) == -1
    assert int(state.free_raw_slot_count.item()) == 1

    def replay(token_idx: int) -> None:
        graph_key.copy_(flat_key[token_idx : token_idx + 1])
        graph_value.copy_(flat_value[token_idx : token_idx + 1])
        graph_slot.fill_(token_idx)
        graph.replay()

    replay(0)
    torch.accelerator.synchronize()
    raw_slot = int(page_to_raw_slot[0].item())
    assert raw_slot == 0
    assert diagnostic.cpu().tolist() == [0, 1, 0]

    byte_cache[0].zero_()
    replay(1)
    torch.accelerator.synchronize()
    assert diagnostic.cpu().tolist() == [0, 1, 1]
    assert int(page_to_raw_slot[0].item()) == raw_slot
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)

    _byte_v2_hybrid_writer_reset(
        tensors,
        state,
        torch.tensor([0], dtype=torch.int32, device=byte_cache.device),
    )
    byte_cache[0].zero_()
    diagnostic[0].fill_(1)
    replay(0)
    replay(1)
    torch.accelerator.synchronize()
    assert int(page_to_raw_slot[0].item()) == raw_slot
    assert int(state.free_raw_slot_count.item()) == 0
    assert int(state.raw_pool_overflow.item()) == 0
    assert diagnostic.cpu().tolist() == [0, 2, 2]
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)

    _byte_v2_hybrid_writer_reset(
        tensors,
        state,
        torch.tensor([0], dtype=torch.int32, device=byte_cache.device),
    )
    torch.accelerator.synchronize()
    assert int(page_to_raw_slot[0].item()) == -1
    assert int(state.free_raw_slot_count.item()) == 1
    assert int(state.raw_pool_overflow.item()) == 0
    assert diagnostic.cpu().tolist() == [0, 2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_writer_ordinary_outliers_remain_compact_bitwise():
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(force_outliers=True)

    _byte_v2_hybrid_writer_update(tensors, state, slot_mapping)
    torch.accelerator.synchronize()

    byte_cache = tensors["byte_cache"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    cache = byte_cache.cpu()
    layout = ByteV2PageLayoutV5()
    stats = collect_byte_v2_fallback_stats(cache, layout=layout)
    assert stats.overlay_tiles > 0
    assert stats.fallback_tiles == 0
    assert _load_u32_bytes(cache, 0, layout.outlier_pool_overflow_offset) == 0
    assert bool((page_to_raw_slot == -1).all())
    assert int(state.free_raw_slot_count.cpu().item()) == 1
    assert int(state.raw_pool_overflow.cpu().item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_writer_pool_overflow_publishes_raw_bitwise():
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        force_pool_overflow=True
    )

    _byte_v2_hybrid_writer_update(tensors, state, slot_mapping)
    torch.accelerator.synchronize()

    byte_cache = tensors["byte_cache"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)
    cache = byte_cache.cpu()
    layout = ByteV2PageLayoutV5()
    assert _load_u32_bytes(cache, 0, layout.outlier_pool_overflow_offset) != 0
    assert int(page_to_raw_slot[0].cpu().item()) >= 0
    assert int(state.free_raw_slot_count.cpu().item()) == 0
    assert int(state.raw_pool_overflow.cpu().item()) == 0
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_writer_updates_existing_raw_page_bitwise():
    _require_byte_v2_hybrid_writer_reader_ops()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        force_pool_overflow=True
    )

    _byte_v2_hybrid_writer_update(tensors, state, slot_mapping[:15])
    torch.accelerator.synchronize()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    byte_cache = tensors["byte_cache"]
    assert isinstance(page_to_raw_slot, torch.Tensor)
    assert isinstance(byte_cache, torch.Tensor)
    raw_slot = int(page_to_raw_slot[0].cpu().item())
    assert raw_slot >= 0

    # Make the compact page unusable so the next hydrate must follow the raw map.
    byte_cache[0].zero_()
    _byte_v2_hybrid_writer_update(tensors, state, slot_mapping[15:])
    torch.accelerator.synchronize()

    assert int(page_to_raw_slot[0].cpu().item()) == raw_slot
    assert int(state.free_raw_slot_count.cpu().item()) == 0
    assert int(state.raw_pool_overflow.cpu().item()) == 0
    persistent_raw = tensors["raw_staging"]
    raw_key = tensors["key"]
    raw_value = tensors["value"]
    assert isinstance(persistent_raw, torch.Tensor)
    assert isinstance(raw_key, torch.Tensor)
    assert isinstance(raw_value, torch.Tensor)
    persistent_bf16 = persistent_raw.view(torch.bfloat16).reshape(-1, 2, 8, 16, 128)
    torch.testing.assert_close(
        persistent_bf16[raw_slot, 0].permute(1, 0, 2),
        raw_key[0],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        persistent_bf16[raw_slot, 1].permute(1, 0, 2),
        raw_value[0],
        atol=0,
        rtol=0,
    )
    _assert_byte_v2_hybrid_writer_matches_raw(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_writer_duplicate_reset_reuses_slot_without_leak():
    if not _byte_v2_hybrid_writer_ops_are_available():
        pytest.skip("ByteV2 hybrid writer custom ops are not registered")

    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        force_pool_overflow=True,
        raw_pool_slots=1,
    )
    byte_cache = tensors["byte_cache"]
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(byte_cache, torch.Tensor)
    assert isinstance(page_to_raw_slot, torch.Tensor)

    for physical_page in (0, 1, 0, 1):
        byte_cache[physical_page].zero_()
        page_slots = slot_mapping[physical_page * 16 : (physical_page + 1) * 16]
        _byte_v2_hybrid_writer_update(tensors, state, page_slots)
        torch.accelerator.synchronize()
        assert int(page_to_raw_slot[physical_page].cpu().item()) == 0
        assert int(state.free_raw_slot_count.cpu().item()) == 0
        assert int(state.raw_pool_overflow.cpu().item()) == 0

        duplicate_ids = torch.tensor(
            [physical_page, physical_page],
            dtype=torch.int32,
            device=byte_cache.device,
        )
        _byte_v2_hybrid_writer_reset(tensors, state, duplicate_ids)
        torch.accelerator.synchronize()
        assert bool((page_to_raw_slot == -1).all())
        assert int(state.free_raw_slot_count.cpu().item()) == 1
        assert int(state.free_raw_slots[0].cpu().item()) == 0
        assert int(state.raw_pool_overflow.cpu().item()) == 0


def _byte_v2_hybrid_writer_pool_exhaustion_probe() -> None:
    assert _byte_v2_hybrid_writer_ops_are_available()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        force_pool_overflow=True,
        raw_pool_slots=1,
    )
    _byte_v2_hybrid_writer_update(tensors, state, slot_mapping[:16])
    torch.accelerator.synchronize()
    page_to_raw_slot = tensors["page_to_raw_slot"]
    assert isinstance(page_to_raw_slot, torch.Tensor)
    assert int(page_to_raw_slot[0].cpu().item()) == 0
    assert int(state.free_raw_slot_count.cpu().item()) == 0

    def announce_commit() -> None:
        torch.accelerator.synchronize()
        print({"hybrid_raw_pool_exhaustion_probe": "commit"}, flush=True)

    _byte_v2_hybrid_writer_update(
        tensors,
        state,
        slot_mapping[16:],
        before_commit=announce_commit,
    )
    torch.accelerator.synchronize()
    print({"hybrid_raw_pool_exhaustion_unexpected_success": True}, flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_hybrid_writer_raw_pool_exhaustion_fails_closed_subprocess():
    if not _byte_v2_hybrid_writer_ops_are_available():
        pytest.skip("ByteV2 hybrid writer custom ops are not registered")

    probe_source = r"""
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
namespace["_byte_v2_hybrid_writer_pool_exhaustion_probe"]()
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_source, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "hybrid_raw_pool_exhaustion_probe" in combined_output
    assert "hybrid_raw_pool_exhaustion_unexpected_success" not in combined_output


def _byte_v2_fused_hybrid_q1_fatal_probe(probe: str) -> None:
    assert _byte_v2_fused_hybrid_writer_q1_op_is_available()
    if probe == "invalid-map":
        tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case()
        tensors["page_to_raw_slot"][0] = -2
        torch.accelerator.synchronize()
        print({"fused_hybrid_q1_fatal_probe": probe}, flush=True)
        _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[:1])
    elif probe == "pool-exhaustion":
        tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
            seq_len=32,
            force_pool_overflow=True,
            raw_pool_slots=1,
        )
        for slot in slot_mapping[:16].split(1):
            _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
        torch.accelerator.synchronize()
        assert int(tensors["page_to_raw_slot"][0].item()) == 0
        assert int(state.free_raw_slot_count.item()) == 0
        print({"fused_hybrid_q1_fatal_probe": probe}, flush=True)
        for slot in slot_mapping[16:].split(1):
            _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
    elif probe == "corrupt-free-count":
        tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
            force_pool_overflow=True,
            raw_pool_slots=1,
        )
        state.free_raw_slot_count.fill_(2)
        torch.accelerator.synchronize()
        print({"fused_hybrid_q1_fatal_probe": probe}, flush=True)
        for slot in slot_mapping.split(1):
            _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot)
    else:
        raise ValueError(f"unknown probe: {probe}")
    torch.accelerator.synchronize()
    print({"fused_hybrid_q1_unexpected_success": probe}, flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "probe", ["invalid-map", "pool-exhaustion", "corrupt-free-count"]
)
def test_byte_v2_fused_hybrid_q1_fatal_state_fails_closed_subprocess(probe):
    if not _byte_v2_fused_hybrid_writer_q1_op_is_available():
        pytest.skip("ByteV2 fused hybrid Q1 writer op is not registered")

    probe_source = r"""
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
namespace["_byte_v2_fused_hybrid_q1_fatal_probe"](sys.argv[2])
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe_source,
            str(Path(__file__).resolve()),
            probe,
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "fused_hybrid_q1_fatal_probe" in combined_output
    assert "fused_hybrid_q1_unexpected_success" not in combined_output


def _byte_v2_fused_hybrid_multi_token_fatal_probe(probe: str) -> None:
    assert _byte_v2_fused_hybrid_writer_multi_token_op_is_available()
    if probe == "invalid-map":
        tensors, state, _ = _make_byte_v2_hybrid_writer_case(seq_len=2)
        tensors["page_to_raw_slot"][0] = -2
        slot_mapping = torch.tensor(
            [0, 1], dtype=torch.int64, device=tensors["byte_cache"].device
        )
        active_capacity = 1
    elif probe == "duplicate-slot":
        tensors, state, _ = _make_byte_v2_hybrid_writer_case(seq_len=2)
        slot_mapping = torch.tensor(
            [0, 0], dtype=torch.int64, device=tensors["byte_cache"].device
        )
        active_capacity = 1
    elif probe == "capacity-overflow":
        tensors, state, _ = _make_byte_v2_hybrid_writer_case(
            seq_len=32, raw_pool_slots=2
        )
        slot_mapping = torch.tensor(
            [0, 16], dtype=torch.int64, device=tensors["byte_cache"].device
        )
        active_capacity = 1
    elif probe == "pool-exhaustion":
        tensors, state, full_slots = _make_byte_v2_hybrid_writer_case(
            seq_len=32,
            force_pool_overflow=True,
            raw_pool_slots=1,
        )
        _byte_v2_fused_hybrid_writer_multi_token_update(
            tensors,
            state,
            full_slots[:16],
            active_capacity=1,
        )
        torch.accelerator.synchronize()
        assert int(tensors["page_to_raw_slot"][0].item()) == 0
        assert int(state.free_raw_slot_count.item()) == 0
        slot_mapping = full_slots[16:]
        active_capacity = 1
    else:
        raise ValueError(f"unknown probe: {probe}")

    torch.accelerator.synchronize()
    print({"fused_hybrid_multi_token_fatal_probe": probe}, flush=True)
    _byte_v2_fused_hybrid_writer_multi_token_update(
        tensors,
        state,
        slot_mapping,
        active_capacity=active_capacity,
    )
    torch.accelerator.synchronize()
    print({"fused_hybrid_multi_token_unexpected_success": probe}, flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "probe", ["invalid-map", "duplicate-slot", "capacity-overflow", "pool-exhaustion"]
)
def test_byte_v2_fused_hybrid_multi_token_fatal_state_fails_closed_subprocess(
    probe,
):
    if not _byte_v2_fused_hybrid_writer_multi_token_op_is_available():
        pytest.skip("ByteV2 fused hybrid multi-token writer op is not registered")

    probe_source = r"""
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
namespace["_byte_v2_fused_hybrid_multi_token_fatal_probe"](sys.argv[2])
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe_source,
            str(Path(__file__).resolve()),
            probe,
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "fused_hybrid_multi_token_fatal_probe" in combined_output
    assert "fused_hybrid_multi_token_unexpected_success" not in combined_output


def _byte_v2_test_forced_raw_pool_exhaustion_probe() -> None:
    assert _byte_v2_test_forced_raw_promotion_op_is_available()
    tensors, state, slot_mapping = _make_byte_v2_hybrid_writer_case(
        seq_len=32,
        raw_pool_slots=1,
    )
    byte_cache = tensors["byte_cache"]
    assert isinstance(byte_cache, torch.Tensor)
    diagnostic = torch.tensor([1, 0, 0], dtype=torch.int32, device=byte_cache.device)

    _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[:1])
    _byte_v2_test_force_promote_hybrid_q1(tensors, state, slot_mapping[:1], diagnostic)
    torch.accelerator.synchronize()
    assert int(tensors["page_to_raw_slot"][0].item()) == 0
    assert int(state.free_raw_slot_count.item()) == 0

    diagnostic[0].fill_(1)
    torch.accelerator.synchronize()
    print({"test_forced_raw_pool_exhaustion_probe": True}, flush=True)
    _byte_v2_fused_hybrid_writer_q1_update(tensors, state, slot_mapping[16:17])
    _byte_v2_test_force_promote_hybrid_q1(
        tensors, state, slot_mapping[16:17], diagnostic
    )
    torch.accelerator.synchronize()
    print({"test_forced_raw_pool_exhaustion_unexpected_success": True}, flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_test_forced_raw_pool_exhaustion_fails_closed_subprocess():
    if not _byte_v2_test_forced_raw_promotion_op_is_available():
        pytest.skip("ByteV2 test forced-raw promotion op is not registered")

    probe_source = r"""
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
namespace["_byte_v2_test_forced_raw_pool_exhaustion_probe"]()
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_source, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "test_forced_raw_pool_exhaustion_probe" in combined_output
    assert "test_forced_raw_pool_exhaustion_unexpected_success" not in (combined_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fatal_option",
    [
        "--inject-overflow",
        "--inject-fallback",
        "--stress-pool-overflow",
    ],
)
def test_byte_v2_fa2_fatal_v5_page_fails_closed_in_subprocess(fatal_option):
    from scripts.byte_v2_fa2_oracle import fa2_oracle_ops_are_available

    if not fa2_oracle_ops_are_available():
        pytest.skip("ByteV2 and raw FA2 extension ops are not registered")

    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/byte_v2_fa2_oracle.py",
            "--seq-len",
            "16",
            "--byte-only",
            fatal_option,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert "fatal_test" in combined_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_collect_cache_stats_cuda_detects_masks():
    if not _has_torch_op("_C_cache_ops", "byte_v2_collect_cache_stats"):
        pytest.skip("ByteV2 cache stats custom op is not registered")

    layout = ByteV2PageLayoutV5()
    kv_cache = torch.zeros(
        (2, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([32], dtype=torch.int32, device="cuda")
    stats = torch.empty((4,), dtype=torch.int32, device="cuda")

    byte_v2_collect_cache_stats(
        stats,
        kv_cache,
        block_tables,
        seq_lens,
        max_seq_len=32,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    assert stats.cpu().tolist() == [0, 0, 0, 2]

    kv_cache[
        0,
        layout.k_fallback_mask_offset(kv_head=0),
    ] = 0b00000001
    kv_cache[
        1,
        layout.v_outlier_mask_offset(kv_head=1),
    ] = 0b00000011

    byte_v2_collect_cache_stats(
        stats,
        kv_cache,
        block_tables,
        seq_lens,
        max_seq_len=32,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    assert stats.cpu().tolist() == [1, 1, 2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_update_cache_unsafe_flags_cuda_detects_touched_masks():
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"):
        pytest.skip("ByteV2 unsafe flags custom op is not registered")

    layout = ByteV2PageLayoutV5()
    kv_cache = torch.zeros(
        (3, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    kv_cache[
        0,
        layout.k_fallback_mask_offset(kv_head=0),
    ] = 0b00000001
    kv_cache[
        2,
        layout.v_outlier_mask_offset(kv_head=1),
    ] = 0b00000001
    page_unsafe_flags = torch.full((3,), -7, dtype=torch.int32, device="cuda")
    slot_mapping = torch.tensor([0, 16, -1], dtype=torch.int64, device="cuda")

    byte_v2_update_cache_unsafe_flags(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    assert page_unsafe_flags.cpu().tolist() == [3, 0, -7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_staging_release_updates_unique_page_flags():
    op_name = "byte_v2_release_raw_staging_and_update_flags"
    if not _has_torch_op("_C_cache_ops", op_name):
        pytest.skip("ByteV2 fused staging release custom op is not registered")

    layout = ByteV2PageLayoutV5()
    kv_cache = torch.zeros(
        (3, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    kv_cache[0, layout.k_fallback_mask_offset(kv_head=0)] = 1
    kv_cache[2, layout.v_outlier_mask_offset(kv_head=1)] = 1
    block_to_staging_slot = torch.tensor([0, -1, 1], dtype=torch.int32, device="cuda")
    staging_to_physical_block = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    valid_rows = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    next_staging_slot = torch.tensor([2], dtype=torch.int32, device="cuda")
    overflow = torch.tensor([1], dtype=torch.int32, device="cuda")
    page_unsafe_flags = torch.full((3,), -7, dtype=torch.int32, device="cuda")

    byte_v2_release_raw_staging_and_update_flags(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        page_unsafe_flags,
        kv_cache,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    assert page_unsafe_flags.cpu().tolist() == [3, -7, 5]
    assert block_to_staging_slot.cpu().tolist() == [-1, -1, -1]
    assert staging_to_physical_block.cpu().tolist() == [-1, -1]
    assert valid_rows.cpu().tolist() == [0, 0]
    assert next_staging_slot.cpu().tolist() == [0]
    assert overflow.cpu().tolist() == [0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_reshape_and_cache_cuda_writes_v5_payload():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    key = (
        torch.arange(2 * 8 * 128, dtype=torch.float32, device="cuda")
        .reshape(2, 8, 128)
        .to(torch.bfloat16)
    )
    value = (key + 17).to(torch.bfloat16)
    kv_cache = torch.full(
        (1, layout.page_size_bytes),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    pool_entries_used = _load_u32_bytes(cache, 0, layout.outlier_pool_used_offset)
    assert 0 <= pool_entries_used <= layout.outlier_pool_entries
    assert _load_u32_bytes(cache, 0, layout.outlier_pool_overflow_offset) == 0
    assert torch.all(cache[0, 8 : layout.page_header_bytes] == 0)

    k_bits0 = _bf16_bits(key[0, 0, 0])
    k_bits1 = _bf16_bits(key[0, 0, 1])
    k_offset = layout.k_payload_offset(kv_head=0, dim_tile=0)
    assert int(cache[0, k_offset]) == k_bits0 & 0xFF
    assert int(cache[0, k_offset + 1]) == k_bits1 & 0xFF
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=0,
        dim=0,
        is_value=False,
    ) == pytest.approx(float(key[0, 0, 0].float().cpu()))

    v_bits0 = _bf16_bits(value[1, 0, 0])
    v_bits1 = _bf16_bits(value[1, 0, 1])
    v_offset = layout.v_payload_offset(kv_head=0, dim_tile=0)
    row_offset = layout.tile_policy.codec_dim_block
    assert int(cache[0, v_offset + row_offset]) == v_bits0 & 0xFF
    assert int(cache[0, v_offset + row_offset + 1]) == v_bits1 & 0xFF
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=1,
        dim=0,
        is_value=True,
    ) == pytest.approx(float(value[1, 0, 0].float().cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_reshape_and_cache_cuda_writes_outlier_overlay():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    key_bits = torch.zeros((16, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    key_bits[0, 0, 0] = (32 << 8) | 0x55
    value_bits[1, 0, 0] = (48 << 8) | 0x66
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)
    kv_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.arange(16, dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    k_fallback_mask = _load_u32_bytes(
        cache,
        0,
        layout.k_fallback_mask_offset(kv_head=0),
    )
    k_outlier_mask = _load_u32_bytes(
        cache,
        0,
        layout.k_outlier_mask_offset(kv_head=0),
    )
    assert not (k_fallback_mask & 1)
    assert k_outlier_mask & 1
    assert int(cache[0, layout.k_outlier_count_offset(kv_head=0, dim_tile=0)]) == 1
    k_entry = _load_u16_bytes(
        cache,
        0,
        _v5_outlier_payload_offset(
            cache,
            layout,
            physical_block=0,
            kv_head=0,
            dim_tile=0,
            is_value=False,
        ),
    )
    assert layout.outlier_entry_policy.decode(k_entry) == (0, 32)
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=0,
        dim=0,
        is_value=False,
    ) == pytest.approx(float(key[0, 0, 0].float().cpu()))

    v_fallback_mask = _load_u32_bytes(
        cache,
        0,
        layout.v_fallback_mask_offset(kv_head=0),
    )
    v_outlier_mask = _load_u32_bytes(
        cache,
        0,
        layout.v_outlier_mask_offset(kv_head=0),
    )
    assert not (v_fallback_mask & 1)
    assert v_outlier_mask & 1
    assert int(cache[0, layout.v_outlier_count_offset(kv_head=0, dim_tile=0)]) == 1
    v_entry = _load_u16_bytes(
        cache,
        0,
        _v5_outlier_payload_offset(
            cache,
            layout,
            physical_block=0,
            kv_head=0,
            dim_tile=0,
            is_value=True,
        ),
    )
    assert layout.outlier_entry_policy.decode(v_entry) == (16, 48)
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=1,
        dim=0,
        is_value=True,
    ) == pytest.approx(float(value[1, 0, 0].float().cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_reshape_and_cache_sideband_high_cuda_writes_outlier_sideband():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache_sideband_high"):
        pytest.skip("ByteV2 sideband-high custom op is not registered")

    layout = ByteV2PageLayoutV5(
        codec_payload_policy=ByteV2CodecPayloadPolicy(
            outlier_high_sideband=True,
        ),
    )
    key_bits = torch.zeros((16, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    key_bits[0, 0, 0] = (32 << 8) | 0x55
    value_bits[1, 0, 0] = (48 << 8) | 0x66
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)
    kv_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.arange(16, dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache_sideband_high(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    k_outlier_mask = _load_u32_bytes(
        cache,
        0,
        layout.k_outlier_mask_offset(kv_head=0),
    )
    assert k_outlier_mask & 1
    assert int(cache[0, layout.k_outlier_count_offset(kv_head=0, dim_tile=0)]) == 1
    k_sideband_offset = _v5_outlier_payload_offset(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        dim_tile=0,
        is_value=False,
    )
    assert int(cache[0, k_sideband_offset]) == 32
    assert int(cache[0, k_sideband_offset + 1]) == 0
    assert _decode_sideband_high_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=0,
        dim=0,
        is_value=False,
    ) == pytest.approx(float(key[0, 0, 0].float().cpu()))

    v_outlier_mask = _load_u32_bytes(
        cache,
        0,
        layout.v_outlier_mask_offset(kv_head=0),
    )
    assert v_outlier_mask & 1
    assert int(cache[0, layout.v_outlier_count_offset(kv_head=0, dim_tile=0)]) == 1
    v_sideband_offset = _v5_outlier_payload_offset(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        dim_tile=0,
        is_value=True,
    )
    assert int(cache[0, v_sideband_offset + 16]) == 48
    assert _decode_sideband_high_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=1,
        dim=0,
        is_value=True,
    ) == pytest.approx(float(value[1, 0, 0].float().cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_reshape_and_cache_cuda_full_tile_overlay_avoids_raw_fallback():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    key_bits = torch.zeros((16, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    tile_pattern = torch.empty((16, 16), dtype=torch.int16, device="cuda")
    flat_tile = tile_pattern.view(-1)
    flat_tile[:86] = 0 << 8
    flat_tile[86:171] = 32 << 8
    flat_tile[171:] = 64 << 8
    key_bits[:, 0, :16] = tile_pattern
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)
    kv_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.arange(16, dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    k_fallback_mask = _load_u32_bytes(
        cache,
        0,
        layout.k_fallback_mask_offset(kv_head=0),
    )
    k_outlier_mask = _load_u32_bytes(
        cache,
        0,
        layout.k_outlier_mask_offset(kv_head=0),
    )
    assert not (k_fallback_mask & 1)
    assert k_outlier_mask & 1
    assert int(cache[0, layout.k_outlier_count_offset(kv_head=0, dim_tile=0)]) == 170
    k_entry = _load_u16_bytes(
        cache,
        0,
        _v5_outlier_payload_offset(
            cache,
            layout,
            physical_block=0,
            kv_head=0,
            dim_tile=0,
            is_value=False,
            entry_idx=169,
        ),
    )
    assert layout.outlier_entry_policy.decode(k_entry) == (255, 64)
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=0,
        row=15,
        dim=15,
        is_value=False,
    ) == pytest.approx(float(key[15, 0, 15].float().cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_reshape_and_cache_cuda_block_direct_writes_v5_payload():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    key = (
        torch.arange(17 * 8 * 128, dtype=torch.float32, device="cuda")
        .reshape(17, 8, 128)
        .to(torch.bfloat16)
    )
    value = (key + 17).to(torch.bfloat16)
    kv_cache = torch.full(
        (2, layout.page_size_bytes),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.arange(17, dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    assert torch.all(cache[:, 8 : layout.page_header_bytes] == 0)
    for physical_block in range(cache.shape[0]):
        assert (
            _load_u32_bytes(cache, physical_block, layout.outlier_pool_overflow_offset)
            == 0
        )
        assert (
            _load_u32_bytes(cache, physical_block, layout.outlier_pool_used_offset)
            <= layout.outlier_pool_entries
        )

    k_bits0 = _bf16_bits(key[16, 7, 126])
    k_bits1 = _bf16_bits(key[16, 7, 127])
    k_offset = layout.k_payload_offset(kv_head=7, dim_tile=7)
    elem_offset = 14
    assert int(cache[1, k_offset + elem_offset]) == k_bits0 & 0xFF
    assert int(cache[1, k_offset + elem_offset + 1]) == k_bits1 & 0xFF
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=1,
        kv_head=7,
        row=0,
        dim=126,
        is_value=False,
    ) == pytest.approx(float(key[16, 7, 126].float().cpu()))

    v_bits0 = _bf16_bits(value[15, 3, 0])
    v_bits1 = _bf16_bits(value[15, 3, 1])
    v_offset = layout.v_payload_offset(kv_head=3, dim_tile=0)
    row_offset = 15 * layout.tile_policy.codec_dim_block
    assert int(cache[0, v_offset + row_offset]) == v_bits0 & 0xFF
    assert int(cache[0, v_offset + row_offset + 1]) == v_bits1 & 0xFF
    assert _decode_current_byte_v2_payload(
        cache,
        layout,
        physical_block=0,
        kv_head=3,
        row=15,
        dim=0,
        is_value=True,
    ) == pytest.approx(float(value[15, 3, 0].float().cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fallback_stats_counts_cache_writer_tiles():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    key = (
        torch.arange(17 * 8 * 128, dtype=torch.float32, device="cuda")
        .reshape(17, 8, 128)
        .to(torch.bfloat16)
    )
    value = (key + 17).to(torch.bfloat16)
    kv_cache = torch.zeros(
        (2, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    slot_mapping = torch.arange(17, dtype=torch.int64, device="cuda")

    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    cache = kv_cache.cpu()

    expected_fallback_tiles = 0
    expected_overlay_tiles = 0
    expected_overlay_entries = 0
    for physical_block in range(cache.shape[0]):
        for kv_head in range(layout.num_kv_heads):
            expected_fallback_tiles += _load_u32_bytes(
                cache,
                physical_block,
                layout.k_fallback_mask_offset(kv_head=kv_head),
            ).bit_count()
            expected_fallback_tiles += _load_u32_bytes(
                cache,
                physical_block,
                layout.v_fallback_mask_offset(kv_head=kv_head),
            ).bit_count()
            k_outlier_mask = _load_u32_bytes(
                cache,
                physical_block,
                layout.k_outlier_mask_offset(kv_head=kv_head),
            )
            v_outlier_mask = _load_u32_bytes(
                cache,
                physical_block,
                layout.v_outlier_mask_offset(kv_head=kv_head),
            )
            expected_overlay_tiles += k_outlier_mask.bit_count()
            expected_overlay_tiles += v_outlier_mask.bit_count()
            for tile_index in range(layout.tile_policy.codec_tiles_per_k_page):
                if k_outlier_mask & (1 << tile_index):
                    dim_tile = (
                        tile_index
                        // layout.tile_policy.codec_token_tiles_per_alloc_block
                    )
                    token_tile = (
                        tile_index
                        % layout.tile_policy.codec_token_tiles_per_alloc_block
                    )
                    expected_overlay_entries += int(
                        cache[
                            physical_block,
                            layout.k_outlier_count_offset(
                                kv_head=kv_head,
                                dim_tile=dim_tile,
                                token_tile=token_tile,
                            ),
                        ]
                    )
            for tile_index in range(layout.tile_policy.codec_tiles_per_v_page):
                if v_outlier_mask & (1 << tile_index):
                    token_tile = tile_index // layout.tile_policy.v_dim_tiles
                    dim_tile = tile_index % layout.tile_policy.v_dim_tiles
                    expected_overlay_entries += int(
                        cache[
                            physical_block,
                            layout.v_outlier_count_offset(
                                kv_head=kv_head,
                                dim_tile=dim_tile,
                                token_tile=token_tile,
                            ),
                        ]
                    )

    stats = collect_byte_v2_fallback_stats(cache, layout=layout)

    assert stats.total_tiles == 2 * 8 * 16
    assert stats.fallback_tiles == expected_fallback_tiles
    assert stats.overlay_tiles == expected_overlay_tiles
    assert stats.overlay_entries == expected_overlay_entries
    assert stats.fallback_tiles + stats.overlay_tiles > 0
    assert stats.raw_tile_bytes == expected_fallback_tiles * 512
    assert stats.estimated_overlay_bytes == stats.outlier_entries * 2
    assert stats.overlay_bytes == expected_overlay_entries * 2
    assert 0.0 <= stats.overlay_to_raw_tile_bytes_ratio <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [3, 16])
def test_byte_v2_raw_staging_commit_matches_direct_cache(num_tokens):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_append_raw_staging"):
        pytest.skip("ByteV2 raw staging append custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_prepare_raw_staging"):
        pytest.skip("ByteV2 raw staging prepare custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_release_raw_staging"):
        pytest.skip("ByteV2 raw staging release custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_commit_raw_staging_to_cache"):
        pytest.skip("ByteV2 raw staging commit custom op is not registered")

    compressed_layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    num_blocks = (
        num_tokens + compressed_layout.tile_policy.alloc_block_tokens - 1
    ) // (compressed_layout.tile_policy.alloc_block_tokens)
    key = (
        torch.arange(num_tokens * 8 * 128, dtype=torch.float32, device="cuda")
        .reshape(num_tokens, 8, 128)
        .to(torch.bfloat16)
    )
    value = (key + 17).to(torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    block_to_staging_slot = torch.full(
        (num_blocks,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    staging_to_physical_block = torch.full(
        (num_blocks,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    valid_rows = torch.zeros((num_blocks,), dtype=torch.int32, device="cuda")
    next_staging_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    raw_staging = torch.empty(
        (num_blocks, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    staged_cache = torch.zeros(
        (num_blocks, compressed_layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    direct_cache = torch.zeros_like(staged_cache)

    byte_v2_prepare_raw_staging(
        slot_mapping,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        alloc_block_tokens=16,
    )
    byte_v2_append_raw_staging(
        key,
        value,
        raw_staging,
        slot_mapping,
        block_to_staging_slot,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_commit_raw_staging_to_cache(
        raw_staging,
        staged_cache,
        staging_to_physical_block,
        valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        direct_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    assert int(overflow.cpu().item()) == 0
    assert int(next_staging_slot.cpu().item()) == num_blocks
    _assert_byte_v2_caches_decode_equal(
        staged_cache,
        direct_cache,
        compressed_layout,
        num_tokens=num_tokens,
    )

    byte_v2_release_raw_staging(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
    )

    assert torch.all(block_to_staging_slot.cpu() == -1)
    assert torch.all(staging_to_physical_block.cpu() == -1)
    assert torch.all(valid_rows.cpu() == 0)
    assert int(next_staging_slot.cpu().item()) == 0
    assert int(overflow.cpu().item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_direct_cache_update_handles_unaligned_request_boundaries():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")

    layout = ByteV2PageLayoutV5()
    sequence_lengths = (1, 15, 8, 9)
    num_tokens = sum(sequence_lengths)
    torch.manual_seed(20260717)
    key = torch.randn(num_tokens, 8, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.tensor(
        [
            physical_block * layout.tile_policy.alloc_block_tokens + row
            for physical_block, sequence_length in enumerate(sequence_lengths)
            for row in range(sequence_length)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    cache = torch.zeros(
        (len(sequence_lengths), layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )

    byte_v2_reshape_and_cache(
        key,
        value,
        cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    _assert_byte_v2_cache_decodes_tokens(
        cache,
        key,
        value,
        slot_mapping,
        layout=layout,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_raw_staging_cross_page_update_matches_direct_cache():
    required_ops = (
        "byte_v2_reshape_and_cache",
        "byte_v2_append_raw_staging",
        "byte_v2_prepare_raw_staging",
        "byte_v2_hydrate_raw_staging_from_cache",
        "byte_v2_release_raw_staging",
        "byte_v2_commit_raw_staging_to_cache",
    )
    if not all(_has_torch_op("_C_cache_ops", op) for op in required_ops):
        pytest.skip("ByteV2 raw staging custom ops are not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    total_tokens = 20
    initial_tokens = 12
    update_tokens = total_tokens - initial_tokens
    num_blocks = 2
    torch.manual_seed(7)
    key = torch.randn(total_tokens, 8, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device="cuda")
    update_slots = slot_mapping[initial_tokens:]

    staged_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.uint8, device="cuda"
    )
    direct_cache = torch.zeros_like(staged_cache)
    byte_v2_reshape_and_cache(
        key[:initial_tokens],
        value[:initial_tokens],
        staged_cache,
        slot_mapping[:initial_tokens],
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        direct_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    block_to_staging_slot = torch.full(
        (num_blocks,), -1, dtype=torch.int32, device="cuda"
    )
    staging_to_physical_block = torch.full(
        (update_tokens,), -1, dtype=torch.int32, device="cuda"
    )
    valid_rows = torch.zeros((update_tokens,), dtype=torch.int32, device="cuda")
    next_staging_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    raw_staging = torch.empty(
        (update_tokens, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )

    byte_v2_prepare_raw_staging(
        update_slots,
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
        alloc_block_tokens=16,
    )
    byte_v2_hydrate_raw_staging_from_cache(
        raw_staging,
        staged_cache,
        staging_to_physical_block,
        valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_append_raw_staging(
        key[initial_tokens:],
        value[initial_tokens:],
        raw_staging,
        update_slots,
        block_to_staging_slot,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_commit_raw_staging_to_cache(
        raw_staging,
        staged_cache,
        staging_to_physical_block,
        valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    assert int(overflow.cpu().item()) == 0
    assert int(next_staging_slot.cpu().item()) == num_blocks
    staged_cache_cpu = staged_cache.cpu()
    direct_cache_cpu = direct_cache.cpu()
    for token_idx in range(total_tokens):
        physical_block, row = divmod(token_idx, 16)
        for kv_head in range(8):
            for dim in range(128):
                for is_value in (False, True):
                    staged = _decode_current_byte_v2_payload(
                        staged_cache_cpu,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=is_value,
                    )
                    direct = _decode_current_byte_v2_payload(
                        direct_cache_cpu,
                        layout,
                        physical_block=physical_block,
                        kv_head=kv_head,
                        row=row,
                        dim=dim,
                        is_value=is_value,
                    )
                    assert staged == direct

    byte_v2_release_raw_staging(
        block_to_staging_slot,
        staging_to_physical_block,
        valid_rows,
        next_staging_slot,
        overflow,
    )

    assert torch.all(block_to_staging_slot.cpu() == -1)
    assert torch.all(staging_to_physical_block.cpu() == -1)
    assert torch.all(valid_rows.cpu() == 0)
    assert int(next_staging_slot.cpu().item()) == 0
    assert int(overflow.cpu().item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("start_row", [0, 12])
@pytest.mark.parametrize(
    ("fuse_metadata_clear", "warp_parallel_histogram"),
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("update_pattern", ["random", "safe"])
def test_byte_v2_native_raw_staging_update_matches_chained_ops(
    start_row, fuse_metadata_clear, warp_parallel_histogram, update_pattern
):
    required_ops = (
        "byte_v2_reshape_and_cache",
        "byte_v2_append_raw_staging",
        "byte_v2_prepare_raw_staging",
        "byte_v2_hydrate_raw_staging_from_cache",
        "byte_v2_commit_raw_staging_to_cache",
        "byte_v2_release_raw_staging_and_update_flags",
        "byte_v2_update_cache_raw_staging",
    )
    if not all(_has_torch_op("_C_cache_ops", op) for op in required_ops):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    num_tokens = 8
    num_blocks = (start_row + num_tokens + 15) // 16
    initial_tokens = num_blocks * 16
    torch.manual_seed(20260714)
    initial_key = torch.randn(
        initial_tokens, 8, 128, dtype=torch.bfloat16, device="cuda"
    )
    initial_value = torch.randn_like(initial_key)
    initial_slots = torch.arange(initial_tokens, dtype=torch.int64, device="cuda")
    if update_pattern == "random":
        key = torch.randn(num_tokens, 8, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.randn_like(key)
    else:
        key = torch.ones(num_tokens, 8, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.full_like(key, 0.5)
    slot_mapping = torch.arange(
        start_row,
        start_row + num_tokens,
        dtype=torch.int64,
        device="cuda",
    )
    chained_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.uint8, device="cuda"
    )
    byte_v2_reshape_and_cache(
        initial_key,
        initial_value,
        chained_cache,
        initial_slots,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    native_cache = chained_cache.clone()

    def make_state():
        return (
            torch.empty(
                (num_tokens, staging_layout.slot_size_bytes),
                dtype=torch.uint8,
                device="cuda",
            ),
            torch.full((num_blocks,), -1, dtype=torch.int32, device="cuda"),
            torch.full((num_tokens,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros((num_tokens,), dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            torch.zeros((num_blocks,), dtype=torch.int32, device="cuda"),
        )

    chained_state = make_state()
    native_state = make_state()
    (
        chained_staging,
        chained_block_to_slot,
        chained_slot_to_block,
        chained_valid_rows,
        chained_next_slot,
        chained_overflow,
        chained_flags,
    ) = chained_state
    byte_v2_prepare_raw_staging(
        slot_mapping,
        chained_block_to_slot,
        chained_slot_to_block,
        chained_valid_rows,
        chained_next_slot,
        chained_overflow,
        alloc_block_tokens=16,
    )
    byte_v2_hydrate_raw_staging_from_cache(
        chained_staging,
        chained_cache,
        chained_slot_to_block,
        chained_valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_append_raw_staging(
        key,
        value,
        chained_staging,
        slot_mapping,
        chained_block_to_slot,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_commit_raw_staging_to_cache(
        chained_staging,
        chained_cache,
        chained_slot_to_block,
        chained_valid_rows,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    byte_v2_release_raw_staging_and_update_flags(
        chained_block_to_slot,
        chained_slot_to_block,
        chained_valid_rows,
        chained_next_slot,
        chained_overflow,
        chained_flags,
        chained_cache,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    (
        native_staging,
        native_block_to_slot,
        native_slot_to_block,
        native_valid_rows,
        native_next_slot,
        native_overflow,
        native_flags,
    ) = native_state
    byte_v2_update_cache_raw_staging(
        key,
        value,
        native_staging,
        native_cache,
        slot_mapping,
        native_block_to_slot,
        native_slot_to_block,
        native_valid_rows,
        native_next_slot,
        native_overflow,
        native_flags,
        tile_policy=(16, 16, 16, 64, 128, 128),
        fuse_metadata_clear=fuse_metadata_clear,
        warp_parallel_histogram=warp_parallel_histogram,
    )
    torch.accelerator.synchronize()

    _assert_byte_v2_caches_decode_equal(
        native_cache,
        chained_cache,
        layout,
        num_tokens=initial_tokens,
    )
    for native_tensor, chained_tensor in zip(native_state[1:], chained_state[1:]):
        torch.testing.assert_close(native_tensor, chained_tensor, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("row", [0, 1, 8, 15])
@pytest.mark.parametrize("update_pattern", ["random", "safe", "strided"])
@pytest.mark.parametrize("stage_metadata_clear", [False, True])
def test_byte_v2_fused_single_token_staging_matches_safe_update(
    row,
    update_pattern,
    stage_metadata_clear,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_raw_staging"):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    torch.manual_seed(20260719 + row)
    baseline_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    if row:
        initial_key = torch.randn(
            row,
            8,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        initial_value = torch.randn_like(initial_key)
        initial_slots = torch.arange(row, dtype=torch.int64, device="cuda")
        byte_v2_reshape_and_cache(
            initial_key,
            initial_value,
            baseline_cache,
            initial_slots,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )
    candidate_cache = baseline_cache.clone()

    if update_pattern == "random":
        key = torch.randn(1, 8, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.randn_like(key)
    elif update_pattern == "safe":
        key = torch.ones(1, 8, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.full_like(key, 0.5)
    else:
        key = torch.randn(1, 8, 256, dtype=torch.bfloat16, device="cuda")[..., ::2]
        value = torch.randn(1, 8, 256, dtype=torch.bfloat16, device="cuda")[..., 1::2]
        assert key.stride(2) == value.stride(2) == 2
    slot_mapping = torch.tensor([row], dtype=torch.int64, device="cuda")

    def make_state():
        return (
            torch.empty(
                (1, staging_layout.slot_size_bytes),
                dtype=torch.uint8,
                device="cuda",
            ),
            torch.full((1,), -1, dtype=torch.int32, device="cuda"),
            torch.full((1,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
        )

    baseline_state = make_state()
    candidate_state = make_state()

    def update(cache, state, *, fused_stage):
        (
            raw_staging,
            block_to_slot,
            slot_to_block,
            valid_rows,
            next_slot,
            overflow,
            flags,
        ) = state
        byte_v2_update_cache_raw_staging(
            key,
            value,
            raw_staging,
            cache,
            slot_mapping,
            block_to_slot,
            slot_to_block,
            valid_rows,
            next_slot,
            overflow,
            flags,
            tile_policy=(16, 16, 16, 64, 128, 128),
            fuse_metadata_clear=True,
            warp_parallel_histogram=True,
            fuse_single_token_staging=fused_stage,
            fuse_single_token_commit_release=fused_stage,
            fuse_single_token_stage_metadata_clear=(
                fused_stage and stage_metadata_clear
            ),
        )

    update(baseline_cache, baseline_state, fused_stage=False)
    update(candidate_cache, candidate_state, fused_stage=True)
    torch.accelerator.synchronize()

    _assert_byte_v2_caches_decode_equal(
        candidate_cache,
        baseline_cache,
        layout,
        num_tokens=row + 1,
    )
    for candidate_tensor, baseline_tensor in zip(
        candidate_state[1:], baseline_state[1:]
    ):
        torch.testing.assert_close(candidate_tensor, baseline_tensor, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("unsafe_sides", "expected_flag"),
    [
        ("none", 0),
        ("k", 3),
        ("v", 5),
        ("kv", 7),
    ],
)
def test_byte_v2_fused_single_token_commit_release_updates_exact_flags(
    unsafe_sides,
    expected_flag,
):
    required_ops = (
        "byte_v2_update_cache_raw_staging",
        "byte_v2_update_cache_unsafe_flags",
    )
    if not all(_has_torch_op("_C_cache_ops", name) for name in required_ops):
        pytest.skip("ByteV2 native update and flag ops are not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    clean_bits = 0x3F80
    outlier_bits = 0x5F80
    key_bits = torch.full((1, 8, 128), clean_bits, dtype=torch.int16, device="cuda")
    value_bits = torch.full_like(key_bits, clean_bits)
    if "k" in unsafe_sides:
        key_bits[0, 0, 0] = outlier_bits
    if "v" in unsafe_sides:
        value_bits[0, 0, 0] = outlier_bits
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)

    kv_cache = torch.zeros(
        (1, layout.page_size_bytes), dtype=torch.uint8, device="cuda"
    )
    raw_staging = torch.empty(
        (1, staging_layout.slot_size_bytes), dtype=torch.uint8, device="cuda"
    )
    slot_mapping = torch.zeros((1,), dtype=torch.int64, device="cuda")
    block_to_slot = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    slot_to_block = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    valid_rows = torch.zeros((1,), dtype=torch.int32, device="cuda")
    next_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    flags = torch.full((1,), 7, dtype=torch.int32, device="cuda")

    byte_v2_update_cache_raw_staging(
        key,
        value,
        raw_staging,
        kv_cache,
        slot_mapping,
        block_to_slot,
        slot_to_block,
        valid_rows,
        next_slot,
        overflow,
        flags,
        tile_policy=(16, 16, 16, 64, 128, 128),
        fuse_metadata_clear=True,
        warp_parallel_histogram=True,
        fuse_single_token_staging=True,
        fuse_single_token_commit_release=True,
        fuse_single_token_stage_metadata_clear=True,
    )
    oracle_flags = torch.full_like(flags, -1)
    byte_v2_update_cache_unsafe_flags(
        oracle_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )
    torch.accelerator.synchronize()

    assert flags.item() == expected_flag
    torch.testing.assert_close(flags, oracle_flags, atol=0, rtol=0)
    torch.testing.assert_close(block_to_slot, torch.full_like(block_to_slot, -1))
    torch.testing.assert_close(slot_to_block, torch.full_like(slot_to_block, -1))
    torch.testing.assert_close(valid_rows, torch.zeros_like(valid_rows))
    torch.testing.assert_close(next_slot, torch.zeros_like(next_slot))
    torch.testing.assert_close(overflow, torch.zeros_like(overflow))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slot", [-1, 16])
def test_byte_v2_fused_single_token_staging_ignores_invalid_slot(slot):
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_raw_staging"):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    key = torch.randn(1, 8, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    kv_cache = torch.randint(
        0,
        256,
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    original_cache = kv_cache.clone()
    raw_staging = torch.empty(
        (1, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_to_slot = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    slot_to_block = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    valid_rows = torch.zeros((1,), dtype=torch.int32, device="cuda")
    next_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    flags = torch.full((1,), 7, dtype=torch.int32, device="cuda")

    byte_v2_update_cache_raw_staging(
        key,
        value,
        raw_staging,
        kv_cache,
        torch.tensor([slot], dtype=torch.int64, device="cuda"),
        block_to_slot,
        slot_to_block,
        valid_rows,
        next_slot,
        overflow,
        flags,
        tile_policy=(16, 16, 16, 64, 128, 128),
        fuse_metadata_clear=True,
        warp_parallel_histogram=True,
        fuse_single_token_staging=True,
        fuse_single_token_commit_release=True,
        fuse_single_token_stage_metadata_clear=True,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(kv_cache, original_cache, atol=0, rtol=0)
    torch.testing.assert_close(block_to_slot, torch.full_like(block_to_slot, -1))
    torch.testing.assert_close(slot_to_block, torch.full_like(slot_to_block, -1))
    torch.testing.assert_close(valid_rows, torch.zeros_like(valid_rows))
    torch.testing.assert_close(next_slot, torch.zeros_like(next_slot))
    torch.testing.assert_close(overflow, torch.zeros_like(overflow))
    torch.testing.assert_close(flags, torch.full_like(flags, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_raw_staging_schema_keeps_old_call_signature():
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_raw_staging"):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    key = torch.zeros(1, 8, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros_like(key)
    kv_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    raw_staging = torch.empty(
        (1, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_to_slot = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    slot_to_block = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    valid_rows = torch.zeros((1,), dtype=torch.int32, device="cuda")
    next_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    flags = torch.zeros((1,), dtype=torch.int32, device="cuda")

    torch.ops._C_cache_ops.byte_v2_update_cache_raw_staging(
        key,
        value,
        raw_staging,
        kv_cache,
        torch.tensor([-1], dtype=torch.int64, device="cuda"),
        block_to_slot,
        slot_to_block,
        valid_rows,
        next_slot,
        overflow,
        flags,
        [16, 16, 16, 64, 128, 128],
        True,
        False,
        True,
    )
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "key_tokens",
        "tile_policy",
        "fuse_metadata_clear",
        "warp_parallel_histogram",
    ),
    [
        (1, (16, 16, 32, 64, 128, 128), True, True),
        (1, (16, 16, 16, 64, 128, 128), False, True),
        (2, (16, 16, 16, 64, 128, 128), True, True),
    ],
)
def test_byte_v2_fused_single_token_staging_preflight_does_not_mutate(
    key_tokens,
    tile_policy,
    fuse_metadata_clear,
    warp_parallel_histogram,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_raw_staging"):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    tensors = [
        torch.zeros(key_tokens, 8, 128, dtype=torch.bfloat16, device="cuda"),
        torch.zeros(key_tokens, 8, 128, dtype=torch.bfloat16, device="cuda"),
        torch.zeros(
            (1, staging_layout.slot_size_bytes),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.zeros(
            (1, layout.page_size_bytes),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.zeros((1,), dtype=torch.int64, device="cuda"),
        torch.full((1,), -1, dtype=torch.int32, device="cuda"),
        torch.full((1,), -1, dtype=torch.int32, device="cuda"),
        torch.zeros((1,), dtype=torch.int32, device="cuda"),
        torch.zeros((1,), dtype=torch.int32, device="cuda"),
        torch.zeros((1,), dtype=torch.int32, device="cuda"),
        torch.zeros((1,), dtype=torch.int32, device="cuda"),
    ]
    snapshots = [tensor.clone() for tensor in tensors]

    with pytest.raises(RuntimeError):
        byte_v2_update_cache_raw_staging(
            *tensors,
            tile_policy=tile_policy,
            fuse_metadata_clear=fuse_metadata_clear,
            warp_parallel_histogram=warp_parallel_histogram,
            fuse_single_token_staging=True,
            fuse_single_token_commit_release=True,
            fuse_single_token_stage_metadata_clear=True,
        )
    torch.accelerator.synchronize()

    for tensor, snapshot in zip(tensors, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_fused_single_token_staging_cuda_graph_cross_page():
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_raw_staging"):
        pytest.skip("ByteV2 native raw staging update op is not registered")

    layout = ByteV2PageLayoutV5()
    staging_layout = ByteV2RawStagingLayout()
    torch.manual_seed(20260719)
    all_key = torch.randn(32, 8, 128, dtype=torch.bfloat16, device="cuda")
    all_value = torch.randn_like(all_key)
    all_slots = torch.arange(32, dtype=torch.int64, device="cuda")
    reference_cache = torch.zeros(
        (2, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        all_key,
        all_value,
        reference_cache,
        all_slots,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    candidate_cache = torch.zeros_like(reference_cache)
    key = torch.empty_like(all_key[:1])
    value = torch.empty_like(all_value[:1])
    slot_mapping = torch.full((1,), -1, dtype=torch.int64, device="cuda")
    raw_staging = torch.empty(
        (1, staging_layout.slot_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_to_slot = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    slot_to_block = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    valid_rows = torch.zeros((1,), dtype=torch.int32, device="cuda")
    next_slot = torch.zeros((1,), dtype=torch.int32, device="cuda")
    overflow = torch.zeros((1,), dtype=torch.int32, device="cuda")
    flags = torch.zeros((2,), dtype=torch.int32, device="cuda")

    def update():
        byte_v2_update_cache_raw_staging(
            key,
            value,
            raw_staging,
            candidate_cache,
            slot_mapping,
            block_to_slot,
            slot_to_block,
            valid_rows,
            next_slot,
            overflow,
            flags,
            tile_policy=(16, 16, 16, 64, 128, 128),
            fuse_metadata_clear=True,
            warp_parallel_histogram=True,
            fuse_single_token_staging=True,
            fuse_single_token_commit_release=True,
            fuse_single_token_stage_metadata_clear=True,
        )

    update()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()

    for token_idx in range(32):
        key.copy_(all_key[token_idx : token_idx + 1])
        value.copy_(all_value[token_idx : token_idx + 1])
        slot_mapping.fill_(token_idx)
        graph.replay()
    torch.accelerator.synchronize()

    _assert_byte_v2_caches_decode_equal(
        candidate_cache,
        reference_cache,
        layout,
        num_tokens=32,
    )
    torch.testing.assert_close(block_to_slot, torch.full_like(block_to_slot, -1))
    torch.testing.assert_close(slot_to_block, torch.full_like(slot_to_block, -1))
    torch.testing.assert_close(valid_rows, torch.zeros_like(valid_rows))
    torch.testing.assert_close(next_slot, torch.zeros_like(next_slot))
    torch.testing.assert_close(overflow, torch.zeros_like(overflow))

    completed_cache = candidate_cache.clone()
    slot_mapping.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(candidate_cache, completed_cache, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_raw_staging_incremental_partial_block_matches_direct_cache():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_prepare_raw_staging"):
        pytest.skip("ByteV2 raw staging prepare custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_hydrate_raw_staging_from_cache"):
        pytest.skip("ByteV2 raw staging hydrate custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_append_raw_staging"):
        pytest.skip("ByteV2 raw staging append custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_commit_raw_staging_to_cache"):
        pytest.skip("ByteV2 raw staging commit custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_release_raw_staging"):
        pytest.skip("ByteV2 raw staging release custom op is not registered")

    layout = ByteV2PageLayoutV5()
    total_tokens = 56
    initial_tokens = 52
    num_blocks = (
        total_tokens + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(
        total_tokens * 8 * 128,
        dtype=torch.float32,
        device="cuda",
    )
    key = ((token_base % 257) / 1024).reshape(total_tokens, 8, 128).to(torch.bfloat16)
    value = (
        (((token_base + 17) % 263) / 1024)
        .reshape(
            total_tokens,
            8,
            128,
        )
        .to(torch.bfloat16)
    )
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device="cuda")

    staged_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )

    impl = byte_v2_attn_module.ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )
    impl.do_kv_cache_update(
        None,
        key[:initial_tokens],
        value[:initial_tokens],
        staged_cache,
        slot_mapping[:initial_tokens],
    )
    for token_idx in range(initial_tokens, total_tokens):
        impl.do_kv_cache_update(
            None,
            key[token_idx : token_idx + 1],
            value[token_idx : token_idx + 1],
            staged_cache,
            slot_mapping[token_idx : token_idx + 1],
        )

    _assert_byte_v2_cache_decodes_tokens(
        staged_cache,
        key,
        value,
        slot_mapping,
        layout=layout,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("row", [0, 7, 15])
def test_byte_v2_single_token_cache_update_cuda_matches_direct_cache(row):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_single_token_fused"):
        pytest.skip("ByteV2 single-token cache update custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = row + 1
    token_base = torch.arange(
        seq_len * 8 * 128,
        dtype=torch.float32,
        device="cuda",
    )
    key = ((token_base % 257) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = (
        (((token_base + 17) % 263) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    )
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    fused_cache = torch.zeros(
        (1, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    if row > 0:
        byte_v2_reshape_and_cache(
            key[:row],
            value[:row],
            fused_cache,
            slot_mapping[:row],
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )
    page_unsafe_flags = torch.zeros((1,), dtype=torch.int32, device="cuda")
    byte_v2_update_cache_single_token(
        key[row : row + 1],
        value[row : row + 1],
        fused_cache,
        slot_mapping[row : row + 1],
        page_unsafe_flags=page_unsafe_flags,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    _assert_byte_v2_cache_decodes_tokens(
        fused_cache,
        key,
        value,
        slot_mapping,
        layout=layout,
    )
    expected_flags = torch.zeros_like(page_unsafe_flags)
    byte_v2_update_cache_unsafe_flags(
        expected_flags,
        fused_cache,
        slot_mapping[row : row + 1],
        tile_policy=(16, 16, 16, 64, 128, 128),
    )
    torch.testing.assert_close(page_unsafe_flags, expected_flags)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("compute_block_n", [64, 128])
def test_byte_v2_prefill_attention_cuda_matches_reference(causal, compute_block_n):
    if not _has_torch_op("_C", "byte_v2_prefill_attention"):
        pytest.skip("ByteV2 prefill custom op is not registered")

    torch.manual_seed(0)
    query = (torch.randn((7, 4, 128), device="cuda") / 8).to(torch.bfloat16)
    key = (torch.randn((7, 2, 128), device="cuda") / 8).to(torch.bfloat16)
    value = (torch.randn((7, 2, 128), device="cuda") / 8).to(torch.bfloat16)
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)

    byte_v2_prefill_attention(
        output,
        query,
        key,
        value,
        query_start_loc,
        max_query_len=4,
        scale=0.125,
        num_kv_heads=2,
        causal=causal,
        tile_policy=(
            DEFAULT_BYTE_V2_TILE_POLICY.codec_token_block,
            DEFAULT_BYTE_V2_TILE_POLICY.codec_dim_block,
            DEFAULT_BYTE_V2_TILE_POLICY.alloc_block_tokens,
            compute_block_n,
            DEFAULT_BYTE_V2_TILE_POLICY.head_dim,
            DEFAULT_BYTE_V2_TILE_POLICY.head_dim_v,
        ),
    )

    expected = _reference_byte_v2_prefill(
        query,
        key,
        value,
        query_start_loc,
        scale=0.125,
        num_kv_heads=2,
        causal=causal,
    )
    torch.testing.assert_close(output.float().cpu(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compute_block_n", [64, 128])
def test_byte_v2_paged_decode_attention_cuda_matches_current_codec_reference(
    compute_block_n,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention"):
        pytest.skip("ByteV2 decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_lens_cpu = torch.tensor([5, 19], dtype=torch.int32)
    block_tables_cpu = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)
    total_tokens = int(seq_lens_cpu.sum())

    base = torch.arange(total_tokens * 8 * 128, dtype=torch.float32, device="cuda")
    key = ((base % 257) / 1024).reshape(total_tokens, 8, 128).to(torch.bfloat16)
    value = (
        (((base + 17) % 263) / 1024).reshape(total_tokens, 8, 128).to(torch.bfloat16)
    )

    slot_mapping = []
    for seq_idx, seq_len in enumerate(seq_lens_cpu.tolist()):
        for token_idx in range(seq_len):
            block_idx = token_idx // layout.tile_policy.alloc_block_tokens
            row = token_idx % layout.tile_policy.alloc_block_tokens
            slot_mapping.append(int(block_tables_cpu[seq_idx, block_idx]) * 16 + row)
    slot_mapping_gpu = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")

    kv_cache = torch.zeros(
        (3, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping_gpu,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(2 * 32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(2, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = block_tables_cpu.to(device="cuda")
    seq_lens = seq_lens_cpu.to(device="cuda")
    scale = 0.125

    byte_v2_paged_decode_attention(
        output,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=19,
        tile_policy=(16, 16, 16, compute_block_n, 128, 128),
    )

    expected = _reference_current_byte_v2_decode(
        query,
        kv_cache.cpu(),
        block_tables_cpu,
        seq_lens_cpu,
        scale=scale,
        num_kv_heads=8,
        layout=layout,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compute_block_n", [64, 128])
@pytest.mark.parametrize("seq_len", [1, 5, 16, 17, 64, 128])
@pytest.mark.parametrize("assume_no_outlier", [False, True])
def test_byte_v2_paged_decode_attention_cuda_matches_raw_reference(
    seq_len,
    compute_block_n,
    assume_no_outlier,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention"):
        pytest.skip("ByteV2 decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(seq_len * 8 * 128, dtype=torch.float32, device="cuda")
    key_values = (
        ((token_base % 255) + 1) / 1024
        if assume_no_outlier
        else (token_base % 257) / 1024
    )
    value_values = (
        (((token_base + 17) % 255) + 1) / 1024
        if assume_no_outlier
        else ((token_base + 17) % 263) / 1024
    )
    key = key_values.reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = value_values.reshape(seq_len, 8, 128).to(torch.bfloat16)

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    scale = 0.125

    byte_v2_paged_decode_attention(
        output,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        tile_policy=(
            (16, 16, 16, compute_block_n, 128, 128, 0, 1)
            if assume_no_outlier
            else (16, 16, 16, compute_block_n, 128, 128)
        ),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("partition_size", [16, 32])
@pytest.mark.parametrize("seq_len", [32, 64, 96])
@pytest.mark.parametrize("assume_no_outlier", [False, True])
def test_byte_v2_paged_decode_attention_split_k_cuda_matches_raw_reference(
    seq_len,
    partition_size,
    assume_no_outlier,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k"):
        pytest.skip("ByteV2 split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(seq_len * 8 * 128, dtype=torch.float32, device="cuda")
    key_values = (
        ((token_base % 255) + 1) / 1024
        if assume_no_outlier
        else (token_base % 257) / 1024
    )
    value_values = (
        (((token_base + 17) % 255) + 1) / 1024
        if assume_no_outlier
        else ((token_base + 17) % 263) / 1024
    )
    key = key_values.reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = value_values.reshape(seq_len, 8, 128).to(torch.bfloat16)

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, 32, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, 32, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_paged_decode_attention_split_k(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(
            (16, 16, 16, 64, 128, 128, 0, 1)
            if assume_no_outlier
            else (16, 16, 16, 64, 128, 128)
        ),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_gqa_packed", [False, True])
def test_byte_v2_paged_decode_attention_split_k_guarded_cuda_matches_raw_reference(
    use_gqa_packed,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"):
        pytest.skip("ByteV2 unsafe flags custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k_guarded"):
        pytest.skip("ByteV2 guarded split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 64
    partition_size = 32
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    key_bits = torch.zeros((seq_len, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    token_high_bits = (32 + torch.arange(seq_len, device="cuda") % 8).to(torch.int16)
    key_bits[:, :, 0] = (token_high_bits.view(seq_len, 1) << 8) | 0x55
    value_bits[:, :, 1] = ((token_high_bits + 16).view(seq_len, 1) << 8) | 0x66
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    page_unsafe_flags = torch.empty((num_blocks,), dtype=torch.int32, device="cuda")
    byte_v2_update_cache_unsafe_flags(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )
    assert page_unsafe_flags.cpu().tolist() == [7] * num_blocks

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, 32, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, 32, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_paged_decode_attention_split_k_guarded(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, 64, 128, 128, 0, 1, int(use_gqa_packed)),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_byte_v2_guarded_fa2_direct_unsafe_pages_cuda_matches_raw_reference():
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"):
        pytest.skip("ByteV2 unsafe flags custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k_guarded"):
        pytest.skip("ByteV2 guarded split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 64
    partition_size = 64
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    key_bits = torch.zeros((seq_len, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    token_high_bits = (32 + torch.arange(seq_len, device="cuda") % 8).to(torch.int16)
    key_bits[:, :, 0] = (token_high_bits.view(seq_len, 1) << 8) | 0x55
    value_bits[:, :, 1] = ((token_high_bits + 16).view(seq_len, 1) << 8) | 0x66
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    page_unsafe_flags = torch.empty((num_blocks,), dtype=torch.int32, device="cuda")
    byte_v2_update_cache_unsafe_flags(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )
    assert page_unsafe_flags.cpu().tolist() == [7] * num_blocks

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, 32, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, 32, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_paged_decode_attention_split_k_guarded(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, 64, 128, 128, 0, 1, 1, 1, 1, 0, 0, 1),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_len", [2, 4, 8, 16])
def test_byte_v2_speculative_verify_gqa_cuda_matches_causal_raw_reference(
    query_len,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"):
        pytest.skip("ByteV2 unsafe flags custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_speculative_verify_gqa"):
        pytest.skip("ByteV2 GQA speculative verify custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 64
    partition_size = 32
    num_blocks = seq_len // layout.tile_policy.alloc_block_tokens
    key_bits = torch.zeros((seq_len, 8, 128), dtype=torch.int16, device="cuda")
    value_bits = torch.zeros_like(key_bits)
    token_high_bits = (32 + torch.arange(seq_len, device="cuda") % 8).to(torch.int16)
    key_bits[:, :, 0] = (token_high_bits.view(seq_len, 1) << 8) | 0x55
    value_bits[:, :, 1] = ((token_high_bits + 16).view(seq_len, 1) << 8) | 0x66
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    page_unsafe_flags = torch.empty((num_blocks,), dtype=torch.int32, device="cuda")
    byte_v2_update_cache_unsafe_flags(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )
    assert page_unsafe_flags.cpu().tolist() == [7] * num_blocks

    query_base = torch.arange(query_len * 32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(query_len, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1, num_blocks
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, query_len * 32, num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, query_len * 32, num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_speculative_verify_gqa(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        page_unsafe_flags,
        block_tables,
        seq_lens,
        speculative_query_len=query_len,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    query_float = query.float()
    key_float = key.float()
    value_float = value.float()
    expected = torch.empty_like(query_float)
    for token_idx in range(query_len):
        token_seq_len = seq_len - (query_len - 1) + token_idx
        for head_idx in range(32):
            kv_head_idx = head_idx // 4
            scores = torch.matmul(
                key_float[:token_seq_len, kv_head_idx],
                query_float[token_idx, head_idx],
            )
            probs = torch.softmax(scores * scale, dim=0)
            expected[token_idx, head_idx] = torch.matmul(
                probs,
                value_float[:token_seq_len, kv_head_idx],
            )
    torch.testing.assert_close(
        output.cpu().float(),
        expected.cpu().to(torch.bfloat16).float(),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_lens", [(1, 4, 0, 1), (0, 1, 2, 3, 4)])
def test_byte_v2_speculative_verify_ragged_q4_cuda_matches_raw_reference(
    query_lens,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C_cache_ops", "byte_v2_update_cache_unsafe_flags"):
        pytest.skip("ByteV2 unsafe flags custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_speculative_verify_ragged_q4"):
        pytest.skip("ByteV2 ragged Q4 custom op is not registered")

    layout = ByteV2PageLayoutV5()
    num_requests = len(query_lens)
    seq_len = 64
    partition_size = 32
    blocks_per_request = seq_len // layout.tile_policy.alloc_block_tokens
    num_blocks = num_requests * blocks_per_request
    num_cache_tokens = num_requests * seq_len

    key_bits = torch.zeros(
        (num_cache_tokens, 8, 128),
        dtype=torch.int16,
        device="cuda",
    )
    value_bits = torch.zeros_like(key_bits)
    local_token = torch.arange(num_cache_tokens, device="cuda") % seq_len
    request_idx = torch.arange(num_cache_tokens, device="cuda") // seq_len
    token_high_bits = (32 + local_token % 8).to(torch.int16)
    key_low_bits = (0x55 + request_idx).to(torch.int16)
    value_low_bits = (0x66 + request_idx).to(torch.int16)
    key_bits[:, :, 0] = (token_high_bits.view(-1, 1) << 8) | key_low_bits.view(-1, 1)
    value_bits[:, :, 1] = (
        (token_high_bits + 16).view(-1, 1) << 8
    ) | value_low_bits.view(-1, 1)
    key = key_bits.view(torch.bfloat16)
    value = value_bits.view(torch.bfloat16)

    slot_mapping = torch.arange(
        num_cache_tokens,
        dtype=torch.int64,
        device="cuda",
    )
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )
    page_unsafe_flags = torch.empty(
        (num_blocks,),
        dtype=torch.int32,
        device="cuda",
    )
    byte_v2_update_cache_unsafe_flags(
        page_unsafe_flags,
        kv_cache,
        slot_mapping,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    query_start_locs_list = [0]
    for query_len in query_lens:
        query_start_locs_list.append(query_start_locs_list[-1] + query_len)
    num_query_tokens = query_start_locs_list[-1]
    padded_query_tokens = num_query_tokens + 1
    query_base = torch.arange(
        padded_query_tokens * 32 * 128,
        dtype=torch.float32,
        device="cuda",
    )
    query = (
        ((query_base % 127) / 31)
        .reshape(padded_query_tokens, 32, 128)
        .to(torch.bfloat16)
    )
    output = torch.full_like(query, 17.0)
    block_tables = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device="cuda",
    ).reshape(num_requests, blocks_per_request)
    seq_lens = torch.full(
        (num_requests,),
        seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    query_start_locs = torch.tensor(
        query_start_locs_list,
        dtype=torch.int32,
        device="cuda",
    )
    num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (num_requests, 128, num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (num_requests, 128, num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_speculative_verify_ragged_q4(
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
        num_actual_tokens=num_query_tokens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, 64, 128, 128),
    )

    query_float = query[:num_query_tokens].float()
    key_float = key.float().reshape(num_requests, seq_len, 8, 128)
    value_float = value.float().reshape(num_requests, seq_len, 8, 128)
    expected = torch.empty_like(query_float)
    for request_idx, query_len in enumerate(query_lens):
        query_start = query_start_locs_list[request_idx]
        for token_idx in range(query_len):
            token_seq_len = seq_len - (query_len - 1) + token_idx
            output_token_idx = query_start + token_idx
            for head_idx in range(32):
                kv_head_idx = head_idx // 4
                scores = torch.matmul(
                    key_float[request_idx, :token_seq_len, kv_head_idx],
                    query_float[output_token_idx, head_idx],
                )
                probs = torch.softmax(scores * scale, dim=0)
                expected[output_token_idx, head_idx] = torch.matmul(
                    probs,
                    value_float[request_idx, :token_seq_len, kv_head_idx],
                )
    torch.testing.assert_close(
        output[:num_query_tokens].cpu().float(),
        expected.cpu().to(torch.bfloat16).float(),
        atol=5e-3,
        rtol=5e-3,
    )
    torch.testing.assert_close(
        output[num_query_tokens:],
        torch.full_like(output[num_query_tokens:], 17.0),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compute_block_n", [64, 128])
def test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference(
    compute_block_n,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k"):
        pytest.skip("ByteV2 split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 96
    partition_size = 32
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(seq_len * 8 * 128, dtype=torch.float32, device="cuda")
    key = (((token_base % 255) + 1) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = (
        ((((token_base + 17) % 255) + 1) / 1024)
        .reshape(seq_len, 8, 128)
        .to(torch.bfloat16)
    )

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, 32, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, 32, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_paged_decode_attention_split_k(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, compute_block_n, 128, 128, 0, 1, 1),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    (
        "use_qk_mma",
        "use_fa2_mainloop",
        "use_fa2_multiwarp",
        "use_fa2_direct",
        "seq_len",
        "partition_size",
        "direct_diagnostic_mode",
    ),
    [
        (False, False, False, False, 96, 64, 0),
        (True, False, False, False, 96, 64, 0),
        (True, True, False, False, 96, 64, 0),
        (True, False, True, False, 96, 64, 0),
        (True, False, False, True, 96, 64, 0),
        (True, False, False, True, 192, 128, 0),
        (True, False, False, True, 320, 256, 0),
        (True, False, False, True, 320, 256, 27),
    ],
)
def test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference(
    use_qk_mma,
    use_fa2_mainloop,
    use_fa2_multiwarp,
    use_fa2_direct,
    seq_len,
    partition_size,
    direct_diagnostic_mode,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k"):
        pytest.skip("ByteV2 split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(seq_len * 8 * 128, dtype=torch.float32, device="cuda")
    key = (((token_base % 255) + 1) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = (
        ((((token_base + 17) % 255) + 1) / 1024)
        .reshape(seq_len, 8, 128)
        .to(torch.bfloat16)
    )

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, 32, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = (
        torch.empty_like(exp_sums)
        if direct_diagnostic_mode == 27
        else torch.empty((0,), dtype=torch.float32, device="cuda")
    )
    tmp_out = torch.empty(
        (1, 32, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125
    tile_policy: tuple[int, ...] = (
        16,
        16,
        16,
        64,
        128,
        128,
        0,
        1,
        1,
        1,
        int(use_qk_mma),
    )
    if use_fa2_multiwarp:
        tile_policy += (0, 1)
    elif use_fa2_mainloop:
        tile_policy += (1,)
    elif use_fa2_direct:
        tile_policy += (0, 0, 1)
        if direct_diagnostic_mode:
            tile_policy += (direct_diagnostic_mode,)

    byte_v2_paged_decode_attention_split_k(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=tile_policy,
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=8,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("q_heads_per_kv", [1, 2, 4, 8, 16, 32])
def test_byte_v2_split_k_fa2_direct_grouped_q_cuda_matches_raw_reference(
    q_heads_per_kv,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention_split_k"):
        pytest.skip("ByteV2 split-k decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 48
    partition_size = 64
    num_kv_heads = 8
    num_heads = num_kv_heads * q_heads_per_kv
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(
        seq_len * num_kv_heads * 128,
        dtype=torch.float32,
        device="cuda",
    )
    key = (
        (((token_base % 255) + 1) / 1024)
        .reshape(
            seq_len,
            num_kv_heads,
            128,
        )
        .to(torch.bfloat16)
    )
    value = (
        ((((token_base + 17) % 255) + 1) / 1024)
        .reshape(seq_len, num_kv_heads, 128)
        .to(torch.bfloat16)
    )

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(num_heads * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, num_heads, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    max_num_partitions = (seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(
        (1, num_heads, max_num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.empty((0,), dtype=torch.float32, device="cuda")
    tmp_out = torch.empty(
        (1, num_heads, max_num_partitions, 128),
        dtype=torch.float32,
        device="cuda",
    )
    scale = 0.125

    byte_v2_paged_decode_attention_split_k(
        output,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        block_size=16,
        max_seq_len=seq_len,
        partition_size=partition_size,
        tile_policy=(16, 16, 16, 64, 128, 128, 0, 1, 1, 1, 1, 0, 0, 1),
    )

    expected = _reference_raw_paged_decode(
        query,
        key,
        value,
        seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compute_block_n", [64, 128])
def test_byte_v2_paged_decode_attention_cuda_rejects_raw_fallback_without_payload(
    compute_block_n,
):
    if not _has_torch_op("_C_cache_ops", "byte_v2_reshape_and_cache"):
        pytest.skip("ByteV2 cache custom op is not registered")
    if not _has_torch_op("_C", "byte_v2_paged_decode_attention"):
        pytest.skip("ByteV2 decode custom op is not registered")

    layout = ByteV2PageLayoutV5()
    seq_len = 1
    num_blocks = (
        seq_len + layout.tile_policy.alloc_block_tokens - 1
    ) // layout.tile_policy.alloc_block_tokens
    token_base = torch.arange(seq_len * 8 * 128, dtype=torch.float32, device="cuda")
    key = ((token_base % 257) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    value = (
        (((token_base + 17) % 263) / 1024).reshape(seq_len, 8, 128).to(torch.bfloat16)
    )

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (num_blocks, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        codec_token_block=16,
        codec_dim_block=16,
        alloc_block_tokens=16,
    )

    query_base = torch.arange(32 * 128, dtype=torch.float32, device="cuda")
    query = ((query_base % 127) / 31).reshape(1, 32, 128).to(torch.bfloat16)
    output = torch.empty_like(query)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        1,
        num_blocks,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    scale = 0.125

    with pytest.raises(RuntimeError, match="raw decode fallback requires"):
        byte_v2_paged_decode_attention(
            output,
            query,
            kv_cache,
            block_tables,
            seq_lens,
            scale=scale,
            num_kv_heads=8,
            block_size=16,
            max_seq_len=seq_len,
            tile_policy=(16, 16, 16, compute_block_n, 128, 128, 1),
        )
