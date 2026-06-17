# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.byte_v2_attn import (
    ByteV2AttentionBackend,
    ByteV2AttentionImpl,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    BYTE_V2_PAGE_VALID_ROWS_OFFSET,
    ByteV2PageLayout,
    unpack_byte_v2_kv_block_from_page,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec


def _assert_bf16_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def _bf16_from_u16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def test_byte_v2_backend_registry_resolves():
    assert AttentionBackendEnum.BYTE_V2.get_class() is ByteV2AttentionBackend
    assert ByteV2AttentionBackend.get_name() == "BYTE_V2"


def test_byte_v2_backend_kv_cache_shape_matches_spec():
    shape = ByteV2AttentionBackend.get_kv_cache_shape(
        num_blocks=4,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str="byte_v2",
    )
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
    )

    assert shape == (4, spec.page_size_bytes)


def test_byte_v2_backend_kv_cache_shape_supports_compressed_only_env(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE", "1")

    shape = ByteV2AttentionBackend.get_kv_cache_shape(
        num_blocks=4,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str="byte_v2",
    )
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        raw_tail_bytes=0,
    )

    assert shape == (4, spec.page_size_bytes)


def test_byte_v2_backend_head_size_support():
    assert ByteV2AttentionBackend.supports_head_size(16)
    assert ByteV2AttentionBackend.supports_head_size(128)
    assert not ByteV2AttentionBackend.supports_head_size(0)
    assert not ByteV2AttentionBackend.supports_head_size(129)


def test_byte_v2_backend_validate_configuration_allows_correctness_fallback():
    invalid_reasons = ByteV2AttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="byte_v2",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 0),
        attn_type=AttentionType.DECODER,
    )

    assert invalid_reasons == []


def test_byte_v2_backend_rejects_unsupported_configuration():
    invalid_reasons = ByteV2AttentionBackend.validate_configuration(
        head_size=96,
        dtype=torch.float16,
        kv_cache_dtype="bfloat16",
        block_size=8,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(7, 5),
        attn_type=AttentionType.DECODER,
    )

    assert "dtype not supported" in invalid_reasons
    assert "kv_cache_dtype not supported" in invalid_reasons
    assert "block_size not supported" in invalid_reasons
    assert "compute capability not supported" in invalid_reasons


def test_byte_v2_attention_impl_cache_update_reference_path():
    impl_cls = ByteV2AttentionBackend.get_impl_cls()
    impl = impl_cls(
        num_heads=2,
        head_size=16,
        scale=1.0,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    layout = ByteV2PageLayout(
        block_size=16, num_kv_heads=1, head_size=16, head_size_v=16
    )
    key = (1.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(16, 1, 16)).to(torch.bfloat16)
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)

    impl.do_kv_cache_update(
        layer=torch.nn.Module(),
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=torch.arange(16),
    )

    key_out, value_out = unpack_byte_v2_kv_block_from_page(kv_cache[0], layout)
    _assert_bf16_bits_equal(key_out, key)
    _assert_bf16_bits_equal(value_out, value)


def test_byte_v2_attention_impl_sparse_fallback_pool_min_blocks(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO", "0.03")
    monkeypatch.setenv("VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS", "512")

    impl_cls = ByteV2AttentionBackend.get_impl_cls()
    impl = impl_cls(
        num_heads=2,
        head_size=16,
        scale=1.0,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    kv_cache = torch.empty(1000, layout.page_size_bytes, dtype=torch.uint8)

    try:
        pool = impl._get_sparse_fallback_pool(kv_cache, layout)
    finally:
        envs.disable_envs_cache()

    assert pool is not None
    assert pool.fallback_pool.shape == (512, layout.raw_block_bytes)


def test_byte_v2_attention_impl_outlier_arena_pool(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO", "0.25")
    monkeypatch.setenv("VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS", "0")
    monkeypatch.setenv("VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK", "2.5")
    monkeypatch.setenv("VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES", "16")

    impl = ByteV2AttentionImpl(
        num_heads=2,
        head_size=16,
        scale=1.0,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    kv_cache = torch.empty(8, layout.page_size_bytes, dtype=torch.uint8)

    try:
        pool = impl._get_sparse_fallback_pool(kv_cache, layout)
        stats = impl.get_sparse_fallback_pool_stats()
    finally:
        envs.disable_envs_cache()

    assert pool is not None
    assert pool.fallback_pool.shape == (2, layout.raw_block_bytes)
    assert pool.outlier_arena is not None
    assert pool.outlier_arena.shape == (20,)
    assert pool.outlier_block_flags is not None
    assert pool.outlier_block_flags.shape == (8,)
    assert int(pool.outlier_block_flags.max().item()) == 0
    assert pool.outlier_tile_bitmap is not None
    assert pool.outlier_tile_bitmap.shape == (8, 1)
    assert int(pool.outlier_tile_bitmap.max().item()) == 0
    assert pool.outlier_tile_meta is not None
    assert pool.outlier_tile_meta.shape == (8, layout.total_tiles)
    assert int(pool.outlier_tile_meta.max().item()) == -1
    assert pool.outlier_next_entry is not None
    assert int(pool.outlier_next_entry.item()) == 0
    assert stats["outlier_capacity"] == 20
    assert stats["assigned_outlier_blocks"] == 0
    assert stats["assigned_outlier_bitmap_tiles"] == 0
    assert stats["outlier_next_entry"] == 0
    assert stats["assigned_outlier_tiles"] == 0
    assert stats["outlier_exhausted"] is False


def test_byte_v2_tile_fallback_stats_read_tile_pool():
    impl = ByteV2AttentionImpl(
        num_heads=1,
        head_size=16,
        scale=1.0,
        num_kv_heads=1,
        kv_cache_dtype="byte_v2",
    )
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8)
    kv_cache[0, BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_COMPRESSED
    kv_cache[0, BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size

    fallback_pool = torch.zeros(1, layout.raw_block_bytes, dtype=torch.uint8)
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32)
    fallback_next_slot = torch.zeros(1, dtype=torch.int32)
    fallback_tile_ids = torch.full((1, layout.total_tiles), -1, dtype=torch.int32)
    fallback_tile_next_slot = torch.ones(1, dtype=torch.int32)

    exp = torch.full((16, 16), 60, dtype=torch.int32)
    bits = exp << 7
    bits[0, 0] = 90 << 7
    raw_tile = _bf16_from_u16(bits, (16, 16))
    tile_slot = 1
    raw_tile_bytes = raw_tile.contiguous().view(torch.uint8).reshape(-1)
    fallback_pool.view(-1, 512)[tile_slot] = raw_tile_bytes
    fallback_tile_ids[0, 0] = tile_slot

    impl.register_sparse_fallback_pool(
        kv_cache,
        fallback_pool,
        fallback_block_ids,
        fallback_next_slot,
        fallback_tile_ids,
        fallback_tile_next_slot,
    )

    stats = impl.get_tile_fallback_stats(kv_cache)
    estimates = stats["outlier_storage_estimates"]
    scenario = estimates["scenarios"]["max_outliers_1"]

    assert stats["full_tile_fallback_tiles"] == 1
    assert stats["full_tile_pool_bad_tiles"] == 1
    assert stats["full_bad_tiles"] == 1
    assert stats["sum_bad_tile_misses"] == 1
    assert scenario["fit_bad_tiles"] == 1
    assert scenario["additional_bytes"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_decode_partial_workspace_reuses_buffer(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DECODE_SPLIT_K", "4")
    envs.disable_envs_cache()
    impl = ByteV2AttentionImpl(
        num_heads=8,
        head_size=32,
        scale=1.0,
        num_kv_heads=2,
        kv_cache_dtype="byte_v2",
    )
    query = torch.empty(1, 8, 32, dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty(1, 20, dtype=torch.int32, device="cuda")

    try:
        workspace = impl._get_decode_partial_workspace(
            query,
            num_decode_tokens=1,
            block_table=block_table,
        )
        reused = impl._get_decode_partial_workspace(
            query,
            num_decode_tokens=1,
            block_table=block_table,
        )
    finally:
        envs.disable_envs_cache()

    assert workspace is not None
    assert reused is not None
    assert workspace.numel() == 1 * 8 * 4 * (32 + 1)
    assert workspace.data_ptr() == reused.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_deferred_cache_update_records_sticky_error(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BYTE_V2_USE_NATIVE_KERNELS", "1")
    monkeypatch.setenv("VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK", "1")
    envs.disable_envs_cache()
    ByteV2AttentionImpl._deferred_cache_update_errors.clear()

    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.zeros(1, 1, 16, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros(1, 1, 16, dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8, device="cuda")
    slot_mapping = torch.tensor([0], dtype=torch.int64, device="cuda")
    fallback_pool = torch.empty(0, layout.raw_block_bytes, dtype=torch.uint8).cuda()
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    total_tiles = layout.num_kv_heads * (layout.k_dim_tiles + layout.v_dim_tiles)
    fallback_tile_ids = torch.full(
        (1, total_tiles), -1, dtype=torch.int32, device="cuda"
    )
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    deferred_error = ByteV2AttentionImpl._get_deferred_cache_update_error(
        torch.device("cuda")
    )
    ByteV2AttentionImpl.reset_deferred_cache_update_error(torch.device("cuda"))

    try:
        result = ops.byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            head_size_v=16,
            page_size_bytes=layout.page_size_bytes,
            fallback_pool=fallback_pool,
            fallback_block_ids=fallback_block_ids,
            fallback_next_slot=fallback_next_slot,
            fallback_tile_ids=fallback_tile_ids,
            fallback_tile_next_slot=fallback_tile_next_slot,
            deferred_error=deferred_error,
        )
        torch.cuda.synchronize()

        assert result.numel() == 0
        assert int(deferred_error[0].item()) == 7
        with pytest.raises(RuntimeError, match="sparse fallback pool exhausted"):
            ByteV2AttentionImpl.check_deferred_cache_update_error(
                torch.device("cuda")
            )
        assert int(deferred_error[0].item()) == 0
    finally:
        ByteV2AttentionImpl._deferred_cache_update_errors.clear()
        envs.disable_envs_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_deferred_batched_decode_append_fast_path_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.randn(2, 1, 16, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(2, 1, 16, dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.zeros(2, layout.page_size_bytes, dtype=torch.uint8,
                           device="cuda")
    fallback_pool = torch.empty(2, layout.raw_block_bytes, dtype=torch.uint8,
                                device="cuda")
    fallback_block_ids = torch.full((2,), -1, dtype=torch.int32,
                                    device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    total_tiles = layout.num_kv_heads * (layout.k_dim_tiles +
                                         layout.v_dim_tiles)
    fallback_tile_ids = torch.full((2, total_tiles), -1, dtype=torch.int32,
                                   device="cuda")
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32,
                                          device="cuda")
    deferred_error = torch.zeros(4, dtype=torch.int32, device="cuda")

    result = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.tensor([0, 16], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        deferred_error=deferred_error,
        decode_append_fast_path_safe=True,
    )
    torch.cuda.synchronize()

    assert result.numel() == 0
    assert deferred_error.cpu().tolist() == [0, 0, 0, 0]
    assert kv_cache[:, BYTE_V2_PAGE_STATUS_OFFSET].cpu().tolist() == [
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
        BYTE_V2_PAGE_STATUS_RAW_FALLBACK,
    ]
    assert kv_cache[:, BYTE_V2_PAGE_VALID_ROWS_OFFSET].cpu().tolist() == [1, 1]
    assert sorted(fallback_block_ids.cpu().tolist()) == [0, 1]
    assert int(fallback_next_slot.item()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_deferred_batched_duplicate_block_records_error_cuda():
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        raw_tail_bytes=0,
    )
    key = torch.zeros(2, 1, 16, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros(2, 1, 16, dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.zeros(1, layout.page_size_bytes, dtype=torch.uint8,
                           device="cuda")
    fallback_pool = torch.empty(1, layout.raw_block_bytes, dtype=torch.uint8,
                                device="cuda")
    fallback_block_ids = torch.full((1,), -1, dtype=torch.int32,
                                    device="cuda")
    fallback_next_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    total_tiles = layout.num_kv_heads * (layout.k_dim_tiles +
                                         layout.v_dim_tiles)
    fallback_tile_ids = torch.full((1, total_tiles), -1, dtype=torch.int32,
                                   device="cuda")
    fallback_tile_next_slot = torch.zeros(1, dtype=torch.int32,
                                          device="cuda")
    deferred_error = torch.zeros(4, dtype=torch.int32, device="cuda")

    result = ops.byte_v2_reshape_and_cache(
        key,
        value,
        kv_cache,
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        page_size_bytes=layout.page_size_bytes,
        fallback_pool=fallback_pool,
        fallback_block_ids=fallback_block_ids,
        fallback_next_slot=fallback_next_slot,
        fallback_tile_ids=fallback_tile_ids,
        fallback_tile_next_slot=fallback_tile_next_slot,
        deferred_error=deferred_error,
        decode_append_fast_path_safe=True,
    )
    torch.cuda.synchronize()

    assert result.numel() == 0
    assert deferred_error.cpu().tolist() == [3, 0, 0, 1]
    assert int(fallback_next_slot.item()) == 0
