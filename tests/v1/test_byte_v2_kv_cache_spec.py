# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    ByteV2FullAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


def _round_up4(value: int) -> int:
    return (value + 3) // 4 * 4


def _make_attention_for_cache_spec(
    *,
    kv_cache_dtype: str = "byte_v2",
    sliding_window: int | None = None,
) -> Attention:
    attention = object.__new__(Attention)
    attention.attn_type = AttentionType.DECODER
    attention.kv_cache_dtype = kv_cache_dtype
    attention.kv_cache_torch_dtype = torch.uint8
    attention.num_kv_heads = 8
    attention.head_size = 128
    attention.head_size_v = 128
    attention.sliding_window = sliding_window
    return attention


def test_attention_get_kv_cache_spec_uses_byte_v2_spec():
    attention = _make_attention_for_cache_spec()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(use_mla=False),
    )

    spec = Attention.get_kv_cache_spec(attention, vllm_config)

    assert isinstance(spec, ByteV2FullAttentionSpec)
    assert spec.block_size == 16
    assert spec.num_kv_heads == 8
    assert spec.head_size == 128
    assert spec.head_size_v == 128
    assert spec.dtype is torch.uint8
    assert spec.raw_tail_bytes is not None


def test_attention_get_kv_cache_spec_uses_default_sparse_fallback_pool_ratio(
    monkeypatch,
):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL", "1")
    monkeypatch.delenv("VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO", raising=False)

    attention = _make_attention_for_cache_spec()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(use_mla=False),
    )

    try:
        spec = Attention.get_kv_cache_spec(attention, vllm_config)
    finally:
        envs.disable_envs_cache()

    assert isinstance(spec, ByteV2FullAttentionSpec)
    assert spec.raw_tail_bytes == 0
    assert spec.sparse_fallback_pool_ratio == pytest.approx(0.03)
    assert spec.sparse_fallback_pool_min_blocks == 512


def test_attention_get_kv_cache_spec_uses_outlier_arena_env(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK", "3.5")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES", "256")

    attention = _make_attention_for_cache_spec()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(use_mla=False),
    )

    try:
        spec = Attention.get_kv_cache_spec(attention, vllm_config)
    finally:
        envs.disable_envs_cache()

    assert isinstance(spec, ByteV2FullAttentionSpec)
    assert spec.outlier_arena_entries_per_block == pytest.approx(3.5)
    assert spec.outlier_arena_min_entries == 256


def test_attention_get_kv_cache_spec_rejects_byte_v2_sliding_window():
    attention = _make_attention_for_cache_spec(sliding_window=128)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(use_mla=False),
    )

    with pytest.raises(ValueError, match="sliding window"):
        Attention.get_kv_cache_spec(attention, vllm_config)


def test_byte_v2_full_attention_spec_page_size_bytes():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        raw_tail_bytes=4096,
        fallback_pool_bytes=512,
    )

    # block=16 gives one token tile. K and V each have 128 / 16 dim tiles.
    num_tiles = 1 * 8 * (8 + 8)
    expected = 16 + num_tiles * 386 + 4096 + 512

    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected


def test_byte_v2_full_attention_spec_v3_page_size_bytes():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        payload_layout="v3",
        raw_tail_bytes=0,
    )

    expected_meta_bytes = _round_up4(8 * 32)
    expected = 128 + expected_meta_bytes + 8 * (8 + 8) * 384

    assert spec.payload_layout == "v3"
    assert spec.page_header_bytes == 128
    assert spec.fast_tile_payload_bytes == 384
    assert spec.raw_tail_bytes == 0
    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected


def test_byte_v2_full_attention_spec_defaults_raw_tail_overlay():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
    )

    compressed_payload = 2 * (2 + 1) * 386
    raw_block_bytes = 16 * 2 * (32 + 16) * 2

    assert spec.raw_tail_bytes == raw_block_bytes - compressed_payload
    assert spec.page_size_bytes == 16 + raw_block_bytes


def test_byte_v2_full_attention_spec_supports_compressed_only_pages():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
        raw_tail_bytes=0,
    )

    compressed_payload = 2 * (2 + 1) * 386

    assert spec.raw_tail_bytes == 0
    assert spec.page_size_bytes == 16 + compressed_payload


def test_byte_v2_full_attention_spec_sparse_fallback_allocation_size():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
        raw_tail_bytes=0,
        sparse_fallback_pool_ratio=0.25,
    )
    num_blocks = 8
    compressed_payload = 1 * (2 + 1) * 386
    main_page_size = 16 + compressed_payload
    raw_block_bytes = 16 * 1 * (32 + 16) * 2
    pool_blocks = 2
    metadata_start = _round_up4(
        num_blocks * main_page_size + pool_blocks * raw_block_bytes
    )
    tile_metadata_bytes = (
        num_blocks * spec.tile_fallback_tiles_per_block * 4
    )
    expected = metadata_start + num_blocks * 4 + 4 + tile_metadata_bytes + 4

    assert spec.page_size_bytes == main_page_size
    assert spec.sparse_fallback_pool_blocks(num_blocks) == pool_blocks
    assert spec.allocation_size_bytes(num_blocks) == expected


def test_byte_v2_full_attention_spec_outlier_arena_allocation_size():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
        raw_tail_bytes=0,
        sparse_fallback_pool_ratio=0.25,
        outlier_arena_entries_per_block=1.5,
        outlier_arena_min_entries=16,
    )
    num_blocks = 8
    main_cache_bytes = num_blocks * spec.main_page_size_bytes
    fallback_pool_bytes = 2 * spec.raw_block_bytes
    outlier_arena_entries = 16
    outlier_arena_start = _round_up4(main_cache_bytes + fallback_pool_bytes)
    metadata_start = _round_up4(outlier_arena_start + outlier_arena_entries * 4)
    sparse_metadata_bytes = (
        num_blocks * 4
        + 4
        + num_blocks * spec.tile_fallback_tiles_per_block * 4
        + 4
    )
    outlier_metadata_bytes = (
        num_blocks * 4
        + num_blocks * spec.outlier_tile_bitmap_words_per_block * 4
        + num_blocks * spec.tile_fallback_tiles_per_block * 4
        + 4
    )
    expected = metadata_start + sparse_metadata_bytes + outlier_metadata_bytes

    assert spec.outlier_arena_entries(num_blocks) == outlier_arena_entries
    assert spec.outlier_arena_start_bytes(num_blocks) == outlier_arena_start
    assert spec.metadata_start_bytes(num_blocks) == metadata_start
    assert spec.allocation_size_bytes(num_blocks) == expected


def test_byte_v2_full_attention_spec_rejects_outlier_arena_without_pool():
    with pytest.raises(ValueError, match="requires sparse fallback pool"):
        ByteV2FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=32,
            head_size_v=16,
            dtype=torch.uint8,
            raw_tail_bytes=0,
            outlier_arena_entries_per_block=1.0,
        )


def test_byte_v2_full_attention_spec_sparse_fallback_min_blocks():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
        raw_tail_bytes=0,
        sparse_fallback_pool_ratio=0.03,
        sparse_fallback_pool_min_blocks=512,
    )

    assert spec.sparse_fallback_pool_blocks(1000) == 512
    assert spec.sparse_fallback_pool_blocks(128) == 128


def test_byte_v2_kv_cache_config_accounts_sparse_fallback_pool():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=16,
        dtype=torch.uint8,
        raw_tail_bytes=0,
        sparse_fallback_pool_ratio=0.25,
    )
    target_blocks = 8
    groups = [
        KVCacheGroupSpec(
            layer_names=["layer.0", "layer.1"],
            kv_cache_spec=spec,
        )
    ]
    available_memory = 2 * spec.allocation_size_bytes(target_blocks)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )

    config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory)

    assert config.num_blocks == target_blocks
    assert [tensor.size for tensor in config.kv_cache_tensors] == [
        spec.allocation_size_bytes(target_blocks),
        spec.allocation_size_bytes(target_blocks),
    ]
    assert [tensor.shared_by for tensor in config.kv_cache_tensors] == [
        ["layer.0"],
        ["layer.1"],
    ]


def test_byte_v2_full_attention_spec_validates_tile_shape():
    with pytest.raises(ValueError, match="block_size"):
        ByteV2FullAttentionSpec(
            block_size=8,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )

    with pytest.raises(ValueError, match="head_size"):
        ByteV2FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=96 + 1,
            dtype=torch.uint8,
        )


def test_byte_v2_full_attention_spec_uses_full_attention_manager():
    register_all_kvcache_specs(None)
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
    )

    assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec
