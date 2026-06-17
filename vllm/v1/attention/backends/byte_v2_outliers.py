# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference helpers for Byte-v2 element-level outlier experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.byte_v2_codec import (
    BYTE_V2_FAST_TILE_PAYLOAD_BYTES,
    BYTE_V2_PACKED_TILE_ELEMS,
    BYTE_V2_TILE_ELEMS,
    BYTE_V2_TILE_SIZE,
)

BYTE_V2_RAW_TILE_BYTES = BYTE_V2_TILE_ELEMS * 2
BYTE_V2_OUTLIER_INDEX_BYTES = 1
BYTE_V2_OUTLIER_VALUE_BYTES = 2
BYTE_V2_OUTLIER_LIST_HEADER_BYTES = 1
BYTE_V2_OUTLIER_THRESHOLDS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class ByteV2OutlierTilePayload:
    """Reference payload for one Byte-v2 tile plus optional raw outliers."""

    base: int
    low_bytes: torch.Tensor
    code_packed: torch.Tensor
    outlier_indices: torch.Tensor
    outlier_raw: torch.Tensor
    fallback_raw: torch.Tensor | None
    max_outliers_per_tile: int

    @property
    def uses_raw_fallback(self) -> bool:
        return self.fallback_raw is not None

    @property
    def outlier_count(self) -> int:
        return int(self.outlier_indices.numel())

    @property
    def logical_extra_bytes(self) -> int:
        if self.uses_raw_fallback:
            return BYTE_V2_RAW_TILE_BYTES
        if self.outlier_count == 0:
            return 0
        return (
            BYTE_V2_OUTLIER_LIST_HEADER_BYTES
            + self.outlier_count
            * (BYTE_V2_OUTLIER_INDEX_BYTES + BYTE_V2_OUTLIER_VALUE_BYTES)
        )


def _bf16_to_u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _u16_to_bf16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def _validate_tile(tile: torch.Tensor) -> torch.Tensor:
    if tile.dtype != torch.bfloat16 or tuple(tile.shape) != (
        BYTE_V2_TILE_SIZE,
        BYTE_V2_TILE_SIZE,
    ):
        raise ValueError("Byte-v2 outlier codec expects one 16x16 BF16 tile")
    return tile.detach().cpu().contiguous()


def _best_window_base_and_mask(exp: torch.Tensor) -> tuple[int, torch.Tensor]:
    hist = torch.bincount(exp.to(torch.int64), minlength=256)
    window = int(hist[:16].sum().item())
    best = window
    best_start = 0
    for start in range(1, 241):
        window += int(hist[start + 15].item()) - int(hist[start - 1].item())
        if window > best:
            best = window
            best_start = start
    mask = (exp >= best_start) & (exp <= best_start + 15)
    return best_start, mask


def compress_byte_v2_tile_with_outliers(
    tile: torch.Tensor,
    *,
    max_outliers_per_tile: int,
) -> ByteV2OutlierTilePayload:
    """Compress one BF16 tile using Byte-v2 plus an element outlier list.

    If a tile has more outliers than ``max_outliers_per_tile``, the reference
    payload marks the tile as raw fallback. This is a correctness and storage
    simulation helper; it is not part of the production CUDA decode path.
    """
    if max_outliers_per_tile < 0:
        raise ValueError("max_outliers_per_tile must be >= 0")

    tile = _validate_tile(tile)
    vals = _bf16_to_u16(tile).reshape(-1)
    exp = (vals >> 7) & 0xFF
    base, covered = _best_window_base_and_mask(exp)
    outlier_indices = torch.nonzero(~covered, as_tuple=False).flatten()

    if int(outlier_indices.numel()) > max_outliers_per_tile:
        return ByteV2OutlierTilePayload(
            base=base,
            low_bytes=torch.zeros(BYTE_V2_TILE_ELEMS, dtype=torch.uint8),
            code_packed=torch.zeros(BYTE_V2_PACKED_TILE_ELEMS, dtype=torch.uint8),
            outlier_indices=torch.empty(0, dtype=torch.uint8),
            outlier_raw=torch.empty(0, dtype=torch.bfloat16),
            fallback_raw=tile.clone(),
            max_outliers_per_tile=max_outliers_per_tile,
        )

    low = (vals & 0xFF).to(torch.uint8)
    delta = torch.clamp(exp - base, min=0, max=15)
    sign = (vals >> 15) & 1
    code = ((sign << 3) | ((delta >> 1) & 0x07)).to(torch.uint8)
    code_packed = code[0::2] | (code[1::2] << 4)

    outlier_raw = vals[outlier_indices].to(torch.int32)
    outlier_raw_bf16 = _u16_to_bf16(outlier_raw, (outlier_indices.numel(),))
    return ByteV2OutlierTilePayload(
        base=base,
        low_bytes=low,
        code_packed=code_packed,
        outlier_indices=outlier_indices.to(torch.uint8),
        outlier_raw=outlier_raw_bf16,
        fallback_raw=None,
        max_outliers_per_tile=max_outliers_per_tile,
    )


def decompress_byte_v2_tile_with_outliers(
    payload: ByteV2OutlierTilePayload,
) -> torch.Tensor:
    """Decompress a payload produced by ``compress_byte_v2_tile_with_outliers``."""
    if payload.fallback_raw is not None:
        return payload.fallback_raw.clone()

    low = payload.low_bytes.to(torch.int32)
    packed = payload.code_packed.to(torch.int32)
    code = torch.empty(BYTE_V2_TILE_ELEMS, dtype=torch.int32)
    code[0::2] = packed & 0x0F
    code[1::2] = (packed >> 4) & 0x0F

    low_exp_lsb = low >> 7
    delta_hi = code & 0x07
    exp_hi = (
        (payload.base >> 1)
        + delta_hi
        + ((payload.base & 1) & (low_exp_lsb ^ 1))
    )
    high = ((code & 0x08) << 4) | exp_hi
    bits = (high << 8) | low

    if payload.outlier_count:
        outlier_bits = _bf16_to_u16(payload.outlier_raw)
        bits[payload.outlier_indices.to(torch.int64)] = outlier_bits
    return _u16_to_bf16(
        bits,
        (BYTE_V2_TILE_SIZE, BYTE_V2_TILE_SIZE),
    )


def byte_v2_tile_exponent_miss_counts(
    exp_tiles: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return per-tile misses outside the best 16-exponent window.

    Args:
        exp_tiles: Tensor shaped ``[num_tiles, elems_per_tile]`` containing
            BF16 exponent bytes.
        chunk_size: Diagnostic chunk size for temporary searchsorted tensors.

    Returns:
        CPU int64 tensor shaped ``[num_tiles]``. A value of zero means the tile
        is exactly representable by the current Byte-v2 fast path.
    """
    if exp_tiles.dim() != 2:
        raise ValueError("exp_tiles must have shape [num_tiles, elems_per_tile]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    num_tiles = int(exp_tiles.shape[0])
    if num_tiles == 0:
        return torch.empty(0, dtype=torch.int64)

    tile_elems = int(exp_tiles.shape[1])
    misses_parts: list[torch.Tensor] = []
    for offset in range(0, num_tiles, chunk_size):
        chunk = exp_tiles[offset : offset + chunk_size].to(torch.int16)
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
    return torch.cat(misses_parts)


def summarize_byte_v2_tile_miss_counts(
    miss_counts: torch.Tensor,
) -> dict[str, int | float]:
    """Summarize nonzero tile miss counts with fixed threshold buckets."""
    misses = miss_counts.detach().cpu().to(torch.int64).reshape(-1)
    bad_misses = misses[misses > 0]
    if bad_misses.numel() == 0:
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

    bad_tiles = int(bad_misses.numel())
    sum_misses = int(bad_misses.sum().item())
    return {
        "bad_tiles": bad_tiles,
        "sum_bad_tile_misses": sum_misses,
        "max_misses_per_bad_tile": int(bad_misses.max().item()),
        "mean_misses_per_bad_tile": float(sum_misses / bad_tiles),
        "bad_tiles_misses_le_1": int((bad_misses <= 1).sum().item()),
        "bad_tiles_misses_le_2": int((bad_misses <= 2).sum().item()),
        "bad_tiles_misses_le_4": int((bad_misses <= 4).sum().item()),
        "bad_tiles_misses_le_8": int((bad_misses <= 8).sum().item()),
        "bad_tiles_misses_gt_8": int((bad_misses > 8).sum().item()),
    }


def byte_v2_bad_tile_miss_summary(
    exp_tiles: torch.Tensor,
) -> dict[str, int | float]:
    """Return lossless Byte-v2 fallback stats for exponent tiles."""
    return summarize_byte_v2_tile_miss_counts(
        byte_v2_tile_exponent_miss_counts(exp_tiles)
    )


def estimate_byte_v2_outlier_storage_from_misses(
    miss_counts: torch.Tensor,
    *,
    total_tiles: int | None = None,
    num_blocks: int | None = None,
    raw_block_bytes: int | None = None,
    thresholds: Sequence[int] = BYTE_V2_OUTLIER_THRESHOLDS,
    raw_tile_bytes: int = BYTE_V2_RAW_TILE_BYTES,
    outlier_header_bytes: int = BYTE_V2_OUTLIER_LIST_HEADER_BYTES,
    outlier_index_bytes: int = BYTE_V2_OUTLIER_INDEX_BYTES,
    outlier_value_bytes: int = BYTE_V2_OUTLIER_VALUE_BYTES,
) -> dict[str, object]:
    """Estimate extra storage for element-level outlier lists.

    The estimate assumes every tile still has the normal Byte-v2 fast payload
    in the main page. It only accounts for extra storage needed by bad tiles:
    either a compact element outlier list, or raw tile fallback if the tile has
    more misses than the chosen threshold.
    """
    misses = miss_counts.detach().cpu().to(torch.int64).reshape(-1)
    if total_tiles is None:
        total_tiles = int(misses.numel())
    bad_misses = misses[misses > 0]
    bad_tiles = int(bad_misses.numel())
    sum_misses = int(bad_misses.sum().item()) if bad_tiles else 0
    raw_tile_fallback_bytes = bad_tiles * raw_tile_bytes
    raw_tile_slots = (
        math.ceil(raw_tile_fallback_bytes / raw_block_bytes)
        if raw_block_bytes and raw_tile_fallback_bytes
        else 0
    )

    scenarios: dict[str, dict[str, int | float]] = {}
    entry_bytes = outlier_index_bytes + outlier_value_bytes
    for threshold in thresholds:
        if threshold < 0:
            raise ValueError("outlier thresholds must be >= 0")
        fit_mask = bad_misses <= threshold
        fit_bad_tiles = int(fit_mask.sum().item())
        overflow_bad_tiles = bad_tiles - fit_bad_tiles
        outlier_entries = (
            int(bad_misses[fit_mask].sum().item()) if fit_bad_tiles else 0
        )
        outlier_list_bytes = (
            fit_bad_tiles * outlier_header_bytes + outlier_entries * entry_bytes
        )
        overflow_raw_tile_bytes = overflow_bad_tiles * raw_tile_bytes
        additional_bytes = outlier_list_bytes + overflow_raw_tile_bytes
        equivalent_slots = (
            math.ceil(additional_bytes / raw_block_bytes)
            if raw_block_bytes and additional_bytes
            else 0
        )
        scenarios[f"max_outliers_{threshold}"] = {
            "max_outliers_per_tile": int(threshold),
            "fit_bad_tiles": fit_bad_tiles,
            "overflow_bad_tiles": overflow_bad_tiles,
            "outlier_entries": outlier_entries,
            "outlier_list_bytes": outlier_list_bytes,
            "overflow_raw_tile_bytes": overflow_raw_tile_bytes,
            "additional_bytes": additional_bytes,
            "bytes_over_raw_tile_fallback": (
                float(additional_bytes / raw_tile_fallback_bytes)
                if raw_tile_fallback_bytes
                else 0.0
            ),
            "equivalent_raw_block_slots": equivalent_slots,
            "equivalent_pool_ratio": (
                float(equivalent_slots / num_blocks)
                if num_blocks and equivalent_slots
                else 0.0
            ),
        }

    return {
        "total_tiles": int(total_tiles),
        "bad_tiles": bad_tiles,
        "sum_bad_tile_misses": sum_misses,
        "fast_tile_payload_bytes": BYTE_V2_FAST_TILE_PAYLOAD_BYTES,
        "raw_tile_bytes": raw_tile_bytes,
        "raw_tile_fallback_bytes": raw_tile_fallback_bytes,
        "raw_tile_fallback_equivalent_raw_block_slots": raw_tile_slots,
        "raw_tile_fallback_equivalent_pool_ratio": (
            float(raw_tile_slots / num_blocks)
            if num_blocks and raw_tile_slots
            else 0.0
        ),
        "scenarios": scenarios,
    }
