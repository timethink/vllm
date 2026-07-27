# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, lossless SplitZip packing for the experimental FA2 reader.

The reader deliberately reuses ByteV2 V5's 52,096-byte fixed-page envelope:
128 bytes of page header, 768 bytes of per-head metadata, two 24,576-byte
dense payloads, and one 2,048-byte page-local escape pool.  Only the codec
meaning changes:

* dense byte 0 stores BF16 sign plus the seven mantissa bits;
* dense byte 1 stores two four-bit exponent-window codes, with the even
  element in the low nibble (the FA2 page-wire convention);
* an escape entry stores ``(true_exponent << 8) | tile_element``.

Packing is intentionally outside the timed attention path.  This module is an
experiment harness, not an online vLLM cache writer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayoutV5

FORMAT_MAGIC = b"SZP1"
DEFAULT_K_EXPONENT_BASE = 115
DEFAULT_V_EXPONENT_BASE = 110


@dataclass(slots=True)
class SplitZipPackStats:
    """Statistics for one fixed-page packing pass."""

    pages: int = 0
    values: int = 0
    escape_entries: int = 0
    max_escape_entries_per_page: int = 0
    max_escape_entries_per_tile: int = 0
    overflow_pages: int = 0

    @property
    def escape_fraction(self) -> float:
        return self.escape_entries / self.values if self.values else 0.0

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["escape_fraction"] = self.escape_fraction
        return result


def _require_page_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use BF16")
    if tensor.ndim != 4 or tensor.shape[1:] != (16, 8, 128):
        raise ValueError(f"{name} must have shape [pages, 16, 8, 128]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _tile_bits(page_tensor: torch.Tensor) -> torch.Tensor:
    """Return unsigned BF16 bits as [page, head, dim_tile, 256]."""
    pages = page_tensor.shape[0]
    bits = page_tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    return (
        bits.view(pages, 16, 8, 8, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(pages, 8, 8, 256)
    )


def _encode_tiles(
    bits: torch.Tensor,
    exponent_base: int,
    valid_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode tiles and return payload, escape mask, and true exponents."""
    exponents = (bits >> 7) & 0xFF
    sign_mantissas = ((bits >> 8) & 0x80) | (bits & 0x7F)
    valid_elements = torch.arange(256, dtype=torch.int32, device=bits.device).div(
        16, rounding_mode="floor"
    ).view(1, 1, 1, 256) < valid_rows.view(-1, 1, 1, 1)
    in_window = (~valid_elements) | (
        (exponents >= exponent_base) & (exponents < exponent_base + 16)
    )
    codes = (exponents - exponent_base).clamp(0, 15)
    packed_codes = codes[..., 0::2] | (codes[..., 1::2] << 4)
    payload = torch.cat(
        (sign_mantissas.to(torch.uint8), packed_codes.to(torch.uint8)),
        dim=-1,
    )

    # Validate the exact BF16 reconstruction algebra before publishing pages.
    reconstructed_exponents = torch.where(
        in_window,
        exponent_base + codes,
        exponents,
    )
    reconstructed_bits = (
        (sign_mantissas & 0x7F)
        | ((reconstructed_exponents & 1) << 7)
        | ((sign_mantissas & 0x80) << 8)
        | ((reconstructed_exponents >> 1) << 8)
    )
    if not torch.equal(
        reconstructed_bits[valid_elements.expand_as(bits)],
        bits[valid_elements.expand_as(bits)],
    ):
        raise RuntimeError("SplitZip dense/escape reconstruction is not bitwise")
    return payload, valid_elements & ~in_window, exponents


def _store_u32(
    destination: torch.Tensor,
    offset: int,
    values: torch.Tensor,
) -> None:
    for byte in range(4):
        destination[..., offset + byte].copy_(
            ((values >> (8 * byte)) & 0xFF).to(torch.uint8)
        )


def _store_u16_planes(
    destination: torch.Tensor,
    offset: int,
    values: torch.Tensor,
) -> None:
    destination[..., offset : offset + 2 * values.shape[-1] : 2].copy_(
        (values & 0xFF).to(torch.uint8)
    )
    destination[..., offset + 1 : offset + 2 * values.shape[-1] : 2].copy_(
        ((values >> 8) & 0xFF).to(torch.uint8)
    )


def _pack_chunk(
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    k_exponent_base: int,
    v_exponent_base: int,
    layout: ByteV2PageLayoutV5,
) -> SplitZipPackStats:
    pages = key.shape[0]
    k_bits = _tile_bits(key)
    v_bits = _tile_bits(value)
    k_payload, k_escape, k_exponents = _encode_tiles(
        k_bits, k_exponent_base, valid_rows
    )
    v_payload, v_escape, v_exponents = _encode_tiles(
        v_bits, v_exponent_base, valid_rows
    )

    output.zero_()
    output[:, 8:12] = torch.tensor(
        list(FORMAT_MAGIC),
        dtype=torch.uint8,
        device=output.device,
    )
    output[
        :,
        layout.k_payload_base_bytes : layout.v_payload_base_bytes,
    ].view(pages, 8, 8, 384).copy_(k_payload)
    output[
        :,
        layout.v_payload_base_bytes : layout.outlier_pool_base_bytes,
    ].view(pages, 8, 8, 384).copy_(v_payload)

    k_counts = k_escape.sum(dim=-1, dtype=torch.int32)
    v_counts = v_escape.sum(dim=-1, dtype=torch.int32)
    all_counts = torch.cat(
        (k_counts.view(pages, 64), v_counts.view(pages, 64)),
        dim=1,
    )
    all_offsets = all_counts.cumsum(dim=1) - all_counts
    pool_used = all_counts.sum(dim=1)
    overflow = pool_used > layout.outlier_pool_entries

    stats = SplitZipPackStats(
        pages=pages,
        values=int(valid_rows.sum().item()) * 2 * 8 * 128,
        escape_entries=int(pool_used.sum().item()),
        max_escape_entries_per_page=int(pool_used.max().item()),
        max_escape_entries_per_tile=int(all_counts.max().item()),
        overflow_pages=int(overflow.sum().item()),
    )
    if stats.overflow_pages:
        raise RuntimeError(
            "SplitZip fixed-page escape pool overflow: "
            f"{stats.overflow_pages}/{pages} pages, "
            f"maximum demand {stats.max_escape_entries_per_page} > "
            f"{layout.outlier_pool_entries}"
        )

    _store_u32(output, layout.outlier_pool_used_offset, pool_used)
    head_metadata = output[
        :,
        layout.page_header_bytes : (
            layout.page_header_bytes + layout.num_kv_heads * layout.kv_head_meta_bytes
        ),
    ].view(pages, 8, layout.kv_head_meta_bytes)
    head_metadata[:, :, 8:16] = k_exponent_base
    head_metadata[:, :, 16:24] = v_exponent_base

    tile_weights = (1 << torch.arange(8, dtype=torch.int64, device=output.device)).view(
        1, 1, 8
    )
    k_masks = ((k_counts > 0) * tile_weights).sum(dim=-1).to(torch.int32)
    v_masks = ((v_counts > 0) * tile_weights).sum(dim=-1).to(torch.int32)
    _store_u32(head_metadata, 24, k_masks)
    _store_u32(head_metadata, 28, v_masks)

    per_head_counts = torch.cat((k_counts, v_counts), dim=-1)
    k_offsets = all_offsets[:, :64].view(pages, 8, 8)
    v_offsets = all_offsets[:, 64:].view(pages, 8, 8)
    per_head_offsets = torch.cat((k_offsets, v_offsets), dim=-1)
    _store_u16_planes(head_metadata, 32, per_head_counts)
    _store_u16_planes(head_metadata, 64, per_head_offsets)

    all_escape = torch.cat(
        (k_escape.view(pages, 64, 256), v_escape.view(pages, 64, 256)),
        dim=1,
    )
    all_exponents = torch.cat(
        (
            k_exponents.view(pages, 64, 256),
            v_exponents.view(pages, 64, 256),
        ),
        dim=1,
    )
    ranks = all_escape.cumsum(dim=-1, dtype=torch.int32) - 1
    pool_positions = all_offsets.unsqueeze(-1) + ranks
    element_indices = torch.arange(
        256,
        dtype=torch.int32,
        device=output.device,
    ).view(1, 1, 256)
    entries = (all_exponents << 8) | element_indices
    page_indices = (
        torch.arange(
            pages,
            dtype=torch.int64,
            device=output.device,
        )
        .view(pages, 1, 1)
        .expand_as(all_escape)
    )
    pool = torch.zeros(
        (pages, layout.outlier_pool_entries),
        dtype=torch.int32,
        device=output.device,
    )
    pool[
        page_indices[all_escape],
        pool_positions[all_escape].to(torch.int64),
    ] = entries[all_escape]
    pool_bytes = output[:, layout.outlier_pool_base_bytes :].view(
        pages, layout.outlier_pool_entries, 2
    )
    pool_bytes[:, :, 0].copy_((pool & 0xFF).to(torch.uint8))
    pool_bytes[:, :, 1].copy_(((pool >> 8) & 0xFF).to(torch.uint8))
    return stats


def pack_splitzip_fixed_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
    k_exponent_base: int = DEFAULT_K_EXPONENT_BASE,
    v_exponent_base: int = DEFAULT_V_EXPONENT_BASE,
    chunk_pages: int = 64,
) -> tuple[torch.Tensor, SplitZipPackStats]:
    """Pack matching raw paged K/V into fixed-size SplitZip pages.

    Args:
        key: Contiguous BF16 tensor shaped ``[pages, 16, 8, 128]``.
        value: Tensor with the same shape, dtype, and device as ``key``.
        valid_rows: Optional integer tensor shaped ``[pages]``. Invalid tail
            rows are encoded densely but never consume escape-pool entries.
        k_exponent_base: First exponent represented by K's 16-code window.
        v_exponent_base: First exponent represented by V's 16-code window.
        chunk_pages: Number of pages vectorized by each offline packing step.

    Returns:
        The uint8 fixed-page cache and aggregate packing statistics.

    Raises:
        ValueError: If tensor/layout arguments are unsupported.
        RuntimeError: If any page exceeds the fixed escape pool.
    """
    _require_page_tensor("key", key)
    _require_page_tensor("value", value)
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("key and value must have matching shape and device")
    if not 0 <= k_exponent_base <= 240:
        raise ValueError("K exponent base must be in [0, 240]")
    if not 0 <= v_exponent_base <= 240:
        raise ValueError("V exponent base must be in [0, 240]")
    if chunk_pages <= 0:
        raise ValueError("chunk_pages must be positive")
    pages = key.shape[0]
    if valid_rows is None:
        valid_rows = torch.full(
            (pages,),
            16,
            dtype=torch.int32,
            device=key.device,
        )
    else:
        if (
            valid_rows.ndim != 1
            or valid_rows.shape[0] != pages
            or valid_rows.device != key.device
            or valid_rows.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "valid_rows must be an integer tensor [pages] on key.device"
            )
        if bool(((valid_rows < 0) | (valid_rows > 16)).any()):
            raise ValueError("valid_rows values must be in [0, 16]")
        valid_rows = valid_rows.to(torch.int32).contiguous()

    layout = ByteV2PageLayoutV5()
    if (
        layout.page_size_bytes != 52_096
        or layout.k_payload_base_bytes != 896
        or layout.v_payload_base_bytes != 25_472
        or layout.outlier_pool_base_bytes != 50_048
        or layout.outlier_entry_bytes != 2
    ):
        raise RuntimeError("compiled SplitZip prototype requires ByteV2 V5 ABI")

    cache = torch.empty(
        (pages, layout.page_size_bytes),
        dtype=torch.uint8,
        device=key.device,
    )
    total = SplitZipPackStats()
    for start in range(0, pages, chunk_pages):
        stop = min(start + chunk_pages, pages)
        chunk = _pack_chunk(
            key[start:stop],
            value[start:stop],
            cache[start:stop],
            valid_rows[start:stop],
            k_exponent_base=k_exponent_base,
            v_exponent_base=v_exponent_base,
            layout=layout,
        )
        total.pages += chunk.pages
        total.values += chunk.values
        total.escape_entries += chunk.escape_entries
        total.max_escape_entries_per_page = max(
            total.max_escape_entries_per_page,
            chunk.max_escape_entries_per_page,
        )
        total.max_escape_entries_per_tile = max(
            total.max_escape_entries_per_tile,
            chunk.max_escape_entries_per_tile,
        )
        total.overflow_pages += chunk.overflow_pages
    return cache, total
