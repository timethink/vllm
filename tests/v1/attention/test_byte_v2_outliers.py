# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.byte_v2_outliers import (
    BYTE_V2_OUTLIER_INDEX_BYTES,
    BYTE_V2_OUTLIER_LIST_HEADER_BYTES,
    BYTE_V2_OUTLIER_VALUE_BYTES,
    BYTE_V2_RAW_TILE_BYTES,
    byte_v2_tile_exponent_miss_counts,
    compress_byte_v2_tile_with_outliers,
    decompress_byte_v2_tile_with_outliers,
    estimate_byte_v2_outlier_storage_from_misses,
    summarize_byte_v2_tile_miss_counts,
)


def _assert_bf16_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def _bf16_from_u16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def test_byte_v2_tile_exponent_miss_counts():
    exp_tiles = torch.tensor(
        [
            [10, 11, 12, 13],
            [0, 1, 2, 20],
            [0, 8, 16, 24],
        ],
        dtype=torch.int32,
    )

    miss_counts = byte_v2_tile_exponent_miss_counts(exp_tiles, chunk_size=2)
    summary = summarize_byte_v2_tile_miss_counts(miss_counts)

    assert miss_counts.tolist() == [0, 1, 2]
    assert summary["bad_tiles"] == 2
    assert summary["sum_bad_tile_misses"] == 3
    assert summary["bad_tiles_misses_le_1"] == 1
    assert summary["bad_tiles_misses_le_2"] == 2


def test_byte_v2_outlier_tile_round_trip_without_raw_fallback():
    exp = torch.full((16, 16), 100, dtype=torch.int32)
    low = torch.arange(256, dtype=torch.int32).reshape(16, 16) & 0x7F
    bits = (exp << 7) | low
    bits[3, 7] = (130 << 7) | 5
    tile = _bf16_from_u16(bits, (16, 16))

    payload = compress_byte_v2_tile_with_outliers(
        tile, max_outliers_per_tile=1
    )
    decompressed = decompress_byte_v2_tile_with_outliers(payload)

    assert not payload.uses_raw_fallback
    assert payload.outlier_count == 1
    assert payload.logical_extra_bytes == (
        BYTE_V2_OUTLIER_LIST_HEADER_BYTES
        + BYTE_V2_OUTLIER_INDEX_BYTES
        + BYTE_V2_OUTLIER_VALUE_BYTES
    )
    _assert_bf16_bits_equal(decompressed, tile)


def test_byte_v2_outlier_tile_uses_raw_fallback_when_threshold_exceeded():
    exp = (torch.arange(256, dtype=torch.int32).reshape(16, 16) % 32) + 40
    bits = exp << 7
    tile = _bf16_from_u16(bits, (16, 16))

    payload = compress_byte_v2_tile_with_outliers(
        tile, max_outliers_per_tile=0
    )
    decompressed = decompress_byte_v2_tile_with_outliers(payload)

    assert payload.uses_raw_fallback
    assert payload.outlier_count == 0
    assert payload.logical_extra_bytes == BYTE_V2_RAW_TILE_BYTES
    _assert_bf16_bits_equal(decompressed, tile)


def test_byte_v2_outlier_storage_estimate():
    estimates = estimate_byte_v2_outlier_storage_from_misses(
        torch.tensor([0, 1, 3, 9], dtype=torch.int64),
        total_tiles=16,
        num_blocks=2,
        raw_block_bytes=2048,
        thresholds=(1, 4),
    )

    assert estimates["total_tiles"] == 16
    assert estimates["bad_tiles"] == 3
    assert estimates["sum_bad_tile_misses"] == 13
    assert estimates["raw_tile_fallback_bytes"] == 3 * BYTE_V2_RAW_TILE_BYTES

    scenarios = estimates["scenarios"]
    threshold_1 = scenarios["max_outliers_1"]
    assert threshold_1["fit_bad_tiles"] == 1
    assert threshold_1["overflow_bad_tiles"] == 2
    assert threshold_1["outlier_entries"] == 1

    threshold_4 = scenarios["max_outliers_4"]
    assert threshold_4["fit_bad_tiles"] == 2
    assert threshold_4["overflow_bad_tiles"] == 1
    assert threshold_4["outlier_entries"] == 4
    assert threshold_4["outlier_list_bytes"] == 2 + 4 * 3
    assert threshold_4["overflow_raw_tile_bytes"] == BYTE_V2_RAW_TILE_BYTES
    assert threshold_4["additional_bytes"] == 526
    assert threshold_4["equivalent_raw_block_slots"] == 1
    assert threshold_4["equivalent_pool_ratio"] == pytest.approx(0.5)


def test_byte_v2_outlier_codec_validates_inputs():
    with pytest.raises(ValueError, match="16x16 BF16"):
        compress_byte_v2_tile_with_outliers(
            torch.zeros(16, 15, dtype=torch.bfloat16),
            max_outliers_per_tile=1,
        )

    with pytest.raises(ValueError, match=">= 0"):
        compress_byte_v2_tile_with_outliers(
            torch.zeros(16, 16, dtype=torch.bfloat16),
            max_outliers_per_tile=-1,
        )
