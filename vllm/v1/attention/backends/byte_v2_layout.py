# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-v2 compressed KV cache page layout helpers.

The first Byte-v2 integration milestone stores each physical KV block as one
flat uint8 page. This module defines the page offsets used by the Python
reference tests and, later, the CUDA cache-update/decode kernels.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.v1.attention.backends.byte_v2_codec import (
    BYTE_V2_FAST_TILE_PAYLOAD_BYTES,
    BYTE_V2_PACKED_TILE_ELEMS,
    BYTE_V2_TILE_ELEMS,
    BYTE_V2_TILE_SIZE,
    ByteV2TensorPayload,
    compress_byte_v2_tensor,
    decompress_byte_v2_tensor,
)
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec

ByteV2KVKind = Literal["k", "v"]

BYTE_V2_PAGE_STATUS_EMPTY = 0
BYTE_V2_PAGE_STATUS_COMPRESSED = 1
BYTE_V2_PAGE_STATUS_RAW_FALLBACK = 2

BYTE_V2_PAGE_STATUS_OFFSET = 0
BYTE_V2_PAGE_VALID_ROWS_OFFSET = 1
BYTE_V2_BF16_BYTES = 2
BYTE_V2_PAYLOAD_LAYOUT_VERSION_V1 = 1
BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3 = 3
BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET = 2
BYTE_V2_PAGE_LAYOUT_FLAGS_OFFSET = 3
BYTE_V2_PAGE_HEADER_BYTES_V3 = 128
BYTE_V2_KV_HEAD_META_BYTES_V3 = 32
BYTE_V2_TILE_PAYLOAD_BYTES_V3 = 384
BYTE_V2_TILE_PAYLOAD_STRIPE_BYTES_V3 = 96
BYTE_V2_TILE_PAYLOAD_STRIPE_PAIRS_V3 = 32
BYTE_V2_PAGE_FLAG_HAS_TILE_FALLBACK_METADATA = 1 << 0
BYTE_V2_PAGE_FLAG_HAS_OUTLIER_METADATA = 1 << 1
BYTE_V2_PAGE_FLAG_COMPRESSED_ONLY = 1 << 2


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class ByteV2TileOffsets:
    base: int
    fallback: int
    low_bytes: int
    code_packed: int


@dataclass(frozen=True)
class ByteV2KvHeadMetaOffsetsV3:
    k_base: int
    v_base: int
    k_fallback_mask: int
    v_fallback_mask: int
    k_outlier_mask: int
    v_outlier_mask: int


@dataclass(frozen=True)
class ByteV2PageStatusCounts:
    empty: int = 0
    compressed: int = 0
    partial_compressed: int = 0
    partial_raw: int = 0
    full_raw_fallback: int = 0
    invalid: int = 0

    @property
    def raw_pages(self) -> int:
        return self.partial_raw + self.full_raw_fallback


@dataclass(frozen=True)
class ByteV2PageLayout:
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    page_header_bytes: int = 16
    fast_tile_payload_bytes: int = BYTE_V2_FAST_TILE_PAYLOAD_BYTES
    raw_tail_bytes: int | None = None

    def __post_init__(self):
        if self.block_size != BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 page layout currently requires block_size=16")
        if self.head_size % BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 page layout head_size must be a multiple of 16")
        if self.head_size_v % BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 page layout head_size_v must be a multiple of 16")
        if self.fast_tile_payload_bytes != BYTE_V2_FAST_TILE_PAYLOAD_BYTES:
            raise ValueError("Byte-v2 page layout expects the fast tile payload size")
        if self.raw_tail_bytes is None:
            object.__setattr__(
                self,
                "raw_tail_bytes",
                max(0, self.raw_block_bytes - self.compressed_payload_bytes),
            )

    @classmethod
    def from_spec(cls, spec: ByteV2FullAttentionSpec) -> "ByteV2PageLayout":
        return cls(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            page_header_bytes=spec.page_header_bytes,
            fast_tile_payload_bytes=spec.fast_tile_payload_bytes,
            raw_tail_bytes=spec.raw_tail_bytes,
        )

    @property
    def k_dim_tiles(self) -> int:
        return self.head_size // BYTE_V2_TILE_SIZE

    @property
    def v_dim_tiles(self) -> int:
        return self.head_size_v // BYTE_V2_TILE_SIZE

    @property
    def tiles_per_head(self) -> int:
        return self.k_dim_tiles + self.v_dim_tiles

    @property
    def total_tiles(self) -> int:
        return self.num_kv_heads * self.tiles_per_head

    @property
    def compressed_payload_bytes(self) -> int:
        return self.total_tiles * self.fast_tile_payload_bytes

    @property
    def raw_key_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * BYTE_V2_BF16_BYTES
        )

    @property
    def raw_value_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size_v
            * BYTE_V2_BF16_BYTES
        )

    @property
    def raw_block_bytes(self) -> int:
        return self.raw_key_bytes + self.raw_value_bytes

    @property
    def page_size_bytes(self) -> int:
        assert self.raw_tail_bytes is not None
        return (
            self.page_header_bytes
            + self.compressed_payload_bytes
            + self.raw_tail_bytes
        )

    def tile_index(self, kind: ByteV2KVKind, kv_head: int, dim_tile: int) -> int:
        if kv_head < 0 or kv_head >= self.num_kv_heads:
            raise ValueError("kv_head is out of range")
        max_dim_tiles = self.k_dim_tiles if kind == "k" else self.v_dim_tiles
        if dim_tile < 0 or dim_tile >= max_dim_tiles:
            raise ValueError("dim_tile is out of range")
        kind_offset = 0 if kind == "k" else self.k_dim_tiles
        return kv_head * self.tiles_per_head + kind_offset + dim_tile

    def tile_offsets(
        self, kind: ByteV2KVKind, kv_head: int, dim_tile: int
    ) -> ByteV2TileOffsets:
        tile_start = (
            self.page_header_bytes
            + self.tile_index(kind, kv_head, dim_tile) * self.fast_tile_payload_bytes
        )
        return ByteV2TileOffsets(
            base=tile_start,
            fallback=tile_start + 1,
            low_bytes=tile_start + 2,
            code_packed=tile_start + 2 + BYTE_V2_TILE_ELEMS,
        )

    def raw_offsets(self, kind: ByteV2KVKind) -> tuple[int, int]:
        if self.page_size_bytes - self.page_header_bytes < self.raw_block_bytes:
            raise NotImplementedError(
                "Byte-v2 page does not reserve enough raw tail storage"
            )
        if kind == "k":
            return self.page_header_bytes, self.raw_key_bytes
        return self.page_header_bytes + self.raw_key_bytes, self.raw_value_bytes


@dataclass(frozen=True)
class ByteV2PageLayoutV3:
    """Experimental Byte-v2 V3 compressed page layout.

    V3 keeps the same 16x16 Byte-v2 codec tile, but moves per-tile metadata into
    a per-KV-head metadata block and stores each tile payload as four
    lane-striped 96B stripes. The default page is compressed-only; callers can
    reserve an optional raw tail for CPU/reference fallback tests.
    """

    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    page_header_bytes: int = BYTE_V2_PAGE_HEADER_BYTES_V3
    kv_head_meta_bytes: int = BYTE_V2_KV_HEAD_META_BYTES_V3
    tile_payload_bytes: int = BYTE_V2_TILE_PAYLOAD_BYTES_V3
    raw_tail_bytes: int = 0

    def __post_init__(self):
        if self.block_size != BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 V3 page layout currently requires block_size=16")
        if self.head_size % BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 V3 layout head_size must be a multiple of 16")
        if self.head_size_v % BYTE_V2_TILE_SIZE:
            raise ValueError("Byte-v2 V3 layout head_size_v must be a multiple of 16")
        if self.page_header_bytes != BYTE_V2_PAGE_HEADER_BYTES_V3:
            raise ValueError("Byte-v2 V3 layout requires a 128B page header")
        if self.kv_head_meta_bytes != BYTE_V2_KV_HEAD_META_BYTES_V3:
            raise ValueError("Byte-v2 V3 layout requires 32B per-KV-head metadata")
        if self.tile_payload_bytes != BYTE_V2_TILE_PAYLOAD_BYTES_V3:
            raise ValueError("Byte-v2 V3 layout requires 384B tile payloads")
        if self.raw_tail_bytes < 0:
            raise ValueError("Byte-v2 V3 raw_tail_bytes must be non-negative")

    @property
    def k_dim_tiles(self) -> int:
        return self.head_size // BYTE_V2_TILE_SIZE

    @property
    def v_dim_tiles(self) -> int:
        return self.head_size_v // BYTE_V2_TILE_SIZE

    @property
    def tiles_per_head(self) -> int:
        return self.k_dim_tiles + self.v_dim_tiles

    @property
    def total_tiles(self) -> int:
        return self.num_kv_heads * self.tiles_per_head

    @property
    def kv_head_meta_region_bytes(self) -> int:
        return _align_up(self.num_kv_heads * self.kv_head_meta_bytes, 128)

    @property
    def k_payload_offset(self) -> int:
        return self.page_header_bytes + self.kv_head_meta_region_bytes

    @property
    def k_payload_bytes(self) -> int:
        return self.num_kv_heads * self.k_dim_tiles * self.tile_payload_bytes

    @property
    def v_payload_offset(self) -> int:
        return self.k_payload_offset + self.k_payload_bytes

    @property
    def v_payload_bytes(self) -> int:
        return self.num_kv_heads * self.v_dim_tiles * self.tile_payload_bytes

    @property
    def compressed_payload_bytes(self) -> int:
        return (
            self.kv_head_meta_region_bytes
            + self.k_payload_bytes
            + self.v_payload_bytes
        )

    @property
    def compressed_page_bytes(self) -> int:
        return self.page_header_bytes + self.compressed_payload_bytes

    @property
    def raw_key_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * BYTE_V2_BF16_BYTES
        )

    @property
    def raw_value_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size_v
            * BYTE_V2_BF16_BYTES
        )

    @property
    def raw_block_bytes(self) -> int:
        return self.raw_key_bytes + self.raw_value_bytes

    @property
    def page_size_bytes(self) -> int:
        return self.compressed_page_bytes + self.raw_tail_bytes

    def kv_head_meta_offsets(self, kv_head: int) -> ByteV2KvHeadMetaOffsetsV3:
        if kv_head < 0 or kv_head >= self.num_kv_heads:
            raise ValueError("kv_head is out of range")
        base = self.page_header_bytes + kv_head * self.kv_head_meta_bytes
        return ByteV2KvHeadMetaOffsetsV3(
            k_base=base,
            v_base=base + 8,
            k_fallback_mask=base + 16,
            v_fallback_mask=base + 18,
            k_outlier_mask=base + 20,
            v_outlier_mask=base + 22,
        )

    def tile_payload_offset(
        self, kind: ByteV2KVKind, kv_head: int, dim_tile: int
    ) -> int:
        if kv_head < 0 or kv_head >= self.num_kv_heads:
            raise ValueError("kv_head is out of range")
        max_dim_tiles = self.k_dim_tiles if kind == "k" else self.v_dim_tiles
        if dim_tile < 0 or dim_tile >= max_dim_tiles:
            raise ValueError("dim_tile is out of range")
        if kind == "k":
            tile_index = kv_head * self.k_dim_tiles + dim_tile
            return self.k_payload_offset + tile_index * self.tile_payload_bytes
        tile_index = kv_head * self.v_dim_tiles + dim_tile
        return self.v_payload_offset + tile_index * self.tile_payload_bytes

    def raw_offsets(self, kind: ByteV2KVKind) -> tuple[int, int]:
        if self.raw_tail_bytes < self.raw_block_bytes:
            raise NotImplementedError(
                "Byte-v2 V3 page does not reserve enough raw tail storage"
            )
        raw_offset = self.compressed_page_bytes
        if kind == "k":
            return raw_offset, self.raw_key_bytes
        return raw_offset + self.raw_key_bytes, self.raw_value_bytes


def _validate_block(
    block: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    name: str,
) -> torch.Tensor:
    expected_shape = (block_size, num_kv_heads, head_size)
    if tuple(block.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {block.shape}")
    if block.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use bfloat16 dtype")
    return block.detach().cpu().contiguous()


def _validate_partial_block(
    block: torch.Tensor,
    *,
    max_block_size: int,
    num_kv_heads: int,
    head_size: int,
    name: str,
) -> torch.Tensor:
    if block.ndim != 3:
        raise ValueError(f"{name} must be a 3D BF16 tensor")
    if block.shape[0] > max_block_size:
        raise ValueError(f"{name} has too many rows for Byte-v2 block")
    expected_tail_shape = (num_kv_heads, head_size)
    if tuple(block.shape[1:]) != expected_tail_shape:
        raise ValueError(
            f"{name} trailing shape must be {expected_tail_shape}, "
            f"got {tuple(block.shape[1:])}"
        )
    if block.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use bfloat16 dtype")
    return block.detach().cpu().contiguous()


def _validate_page(page: torch.Tensor, layout: ByteV2PageLayout) -> torch.Tensor:
    if page.dtype != torch.uint8:
        raise ValueError("Byte-v2 page must use uint8 dtype")
    if page.device.type != "cpu":
        raise ValueError("Byte-v2 reference page helpers currently require CPU tensors")
    if tuple(page.shape) != (layout.page_size_bytes,):
        raise ValueError(
            "Byte-v2 page shape must be "
            f"({layout.page_size_bytes},), got {tuple(page.shape)}"
        )
    return page.cpu()


def _validate_kv_cache(
    kv_cache: torch.Tensor, layout: ByteV2PageLayout
) -> torch.Tensor:
    if kv_cache.dtype != torch.uint8:
        raise ValueError("Byte-v2 kv_cache must use uint8 dtype")
    if kv_cache.device.type != "cpu":
        raise ValueError(
            "Byte-v2 reference cache helpers currently require CPU tensors"
        )
    if kv_cache.ndim != 2 or kv_cache.shape[1] != layout.page_size_bytes:
        raise ValueError(
            "kv_cache must have shape "
            f"[num_blocks, {layout.page_size_bytes}], got {tuple(kv_cache.shape)}"
        )
    return kv_cache.cpu()


def _load_u16_le(page: torch.Tensor, offset: int) -> int:
    return int(page[offset].item()) | (int(page[offset + 1].item()) << 8)


def _store_u16_le(page: torch.Tensor, offset: int, value: int) -> None:
    page[offset] = value & 0xFF
    page[offset + 1] = (value >> 8) & 0xFF


def _store_u32_le(page: torch.Tensor, offset: int, value: int) -> None:
    for byte_idx in range(4):
        page[offset + byte_idx] = (value >> (8 * byte_idx)) & 0xFF


def count_byte_v2_page_statuses(
    kv_cache: torch.Tensor,
    layout: ByteV2PageLayout,
) -> ByteV2PageStatusCounts:
    """Count Byte-v2 page states from the page headers.

    This is intentionally derived from the flat uint8 pages rather than from
    cache-update return values. It gives later runtime code and tests a stable
    way to observe how many pages are compressed, still partial raw tails, or
    full raw fallbacks.
    """
    kv_cache = _validate_kv_cache(kv_cache, layout)

    empty = 0
    compressed = 0
    partial_compressed = 0
    partial_raw = 0
    full_raw_fallback = 0
    invalid = 0
    for page in kv_cache:
        status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
        valid_rows = int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item())
        if status == BYTE_V2_PAGE_STATUS_EMPTY and valid_rows == 0:
            empty += 1
        elif status == BYTE_V2_PAGE_STATUS_COMPRESSED:
            if 0 < valid_rows < layout.block_size:
                partial_compressed += 1
            elif valid_rows == layout.block_size:
                compressed += 1
            else:
                invalid += 1
        elif status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
            if 0 <= valid_rows < layout.block_size:
                partial_raw += 1
            elif valid_rows == layout.block_size:
                full_raw_fallback += 1
            else:
                invalid += 1
        else:
            invalid += 1

    return ByteV2PageStatusCounts(
        empty=empty,
        compressed=compressed,
        partial_compressed=partial_compressed,
        partial_raw=partial_raw,
        full_raw_fallback=full_raw_fallback,
        invalid=invalid,
    )


def _store_fast_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: ByteV2KVKind,
    kv_head: int,
    dim_tile: int,
    tile: torch.Tensor,
) -> bool:
    payload_tile = tile.t().contiguous() if kind == "k" else tile
    payload = compress_byte_v2_tensor(payload_tile)
    if payload.fallback_tiles:
        return False

    offsets = layout.tile_offsets(kind, kv_head, dim_tile)
    page[offsets.base] = payload.base[0]
    page[offsets.fallback] = payload.fallback[0]
    page[offsets.low_bytes : offsets.low_bytes + BYTE_V2_TILE_ELEMS] = (
        payload.low_bytes[0]
    )
    page[
        offsets.code_packed : offsets.code_packed + BYTE_V2_PACKED_TILE_ELEMS
    ] = payload.code_packed[0]
    return True


def _store_v3_header(page: torch.Tensor, layout: ByteV2PageLayoutV3) -> None:
    page[BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_COMPRESSED
    page[BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size
    page[BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET] = BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3
    page[BYTE_V2_PAGE_LAYOUT_FLAGS_OFFSET] = (
        BYTE_V2_PAGE_FLAG_HAS_TILE_FALLBACK_METADATA
        | BYTE_V2_PAGE_FLAG_COMPRESSED_ONLY
    )
    _store_u32_le(page, 8, layout.k_payload_offset)
    _store_u32_le(page, 12, layout.k_payload_offset)
    _store_u32_le(page, 16, layout.v_payload_offset)
    _store_u32_le(page, 20, layout.tile_payload_bytes)
    _store_u32_le(page, 24, layout.page_header_bytes)
    _store_u32_le(page, 28, layout.kv_head_meta_region_bytes)


def _store_v3_striped_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayoutV3,
    kind: ByteV2KVKind,
    kv_head: int,
    dim_tile: int,
    tile: torch.Tensor,
) -> bool:
    payload_tile = tile.t().contiguous() if kind == "k" else tile
    payload = compress_byte_v2_tensor(payload_tile)
    meta_offsets = layout.kv_head_meta_offsets(kv_head)
    fallback_mask_offset = (
        meta_offsets.k_fallback_mask
        if kind == "k"
        else meta_offsets.v_fallback_mask
    )
    base_offset = (
        meta_offsets.k_base + dim_tile
        if kind == "k"
        else meta_offsets.v_base + dim_tile
    )
    page[base_offset] = payload.base[0]

    if payload.fallback_tiles:
        mask = _load_u16_le(page, fallback_mask_offset)
        _store_u16_le(page, fallback_mask_offset, mask | (1 << dim_tile))
        return False

    tile_offset = layout.tile_payload_offset(kind, kv_head, dim_tile)
    low = payload.low_bytes[0]
    packed = payload.code_packed[0]
    for pair_idx in range(BYTE_V2_PACKED_TILE_ELEMS):
        elem0 = pair_idx * 2
        stripe = pair_idx // BYTE_V2_TILE_PAYLOAD_STRIPE_PAIRS_V3
        lane = pair_idx % BYTE_V2_TILE_PAYLOAD_STRIPE_PAIRS_V3
        stripe_offset = tile_offset + stripe * BYTE_V2_TILE_PAYLOAD_STRIPE_BYTES_V3
        page[stripe_offset + lane] = low[elem0]
        page[stripe_offset + 32 + lane] = low[elem0 + 1]
        page[stripe_offset + 64 + lane] = packed[pair_idx]
    return True


def pack_byte_v2_kv_block_to_page_v3(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    layout: ByteV2PageLayoutV3,
    page: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack one full K/V block into an experimental Byte-v2 V3 page."""
    key_block = _validate_block(
        key_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size,
        name="key_block",
    )
    value_block = _validate_block(
        value_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size_v,
        name="value_block",
    )
    if page is None:
        page = torch.zeros(layout.page_size_bytes, dtype=torch.uint8)
    else:
        page = _validate_page(page, layout)
        page.zero_()

    _store_v3_header(page, layout)
    saw_fallback_tile = False
    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            saw_fallback_tile |= not _store_v3_striped_tile(
                page, layout, "k", kv_head, dim_tile, tile
            )
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            saw_fallback_tile |= not _store_v3_striped_tile(
                page, layout, "v", kv_head, dim_tile, tile
            )

    if saw_fallback_tile:
        if layout.raw_tail_bytes >= layout.raw_block_bytes:
            return pack_byte_v2_raw_kv_block_to_page(
                key_block, value_block, layout.block_size, layout, page=page
            )
        raise NotImplementedError(
            "Byte-v2 V3 reference pack requires compressible tiles unless a "
            "raw tail is reserved"
        )
    return page


def pack_byte_v2_kv_block_to_page(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    layout: ByteV2PageLayout,
    page: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack one full K/V block into a flat Byte-v2 page.

    Args:
        key_block: BF16 tensor with shape `[block_size, num_kv_heads, head_size]`.
        value_block: BF16 tensor with shape `[block_size, num_kv_heads, head_size_v]`.
        layout: Byte-v2 page layout.
        page: Optional uint8 page to write into.

    Returns:
        The written page tensor.
    """
    if isinstance(layout, ByteV2PageLayoutV3):
        return pack_byte_v2_kv_block_to_page_v3(
            key_block, value_block, layout, page=page
        )

    key_block = _validate_block(
        key_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size,
        name="key_block",
    )
    value_block = _validate_block(
        value_block,
        block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size_v,
        name="value_block",
    )
    if page is None:
        page = torch.zeros(layout.page_size_bytes, dtype=torch.uint8)
    else:
        page = _validate_page(page, layout)
        page.zero_()

    page[BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_COMPRESSED
    page[BYTE_V2_PAGE_VALID_ROWS_OFFSET] = layout.block_size

    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            if not _store_fast_tile(page, layout, "k", kv_head, dim_tile, tile):
                return pack_byte_v2_raw_kv_block_to_page(
                    key_block, value_block, layout.block_size, layout, page=page
                )
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            tile = value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE]
            if not _store_fast_tile(page, layout, "v", kv_head, dim_tile, tile):
                return pack_byte_v2_raw_kv_block_to_page(
                    key_block, value_block, layout.block_size, layout, page=page
                )

    return page


def _store_raw_block(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: ByteV2KVKind,
    block: torch.Tensor,
) -> None:
    offset, _ = layout.raw_offsets(kind)
    raw_bytes = block.contiguous().view(torch.uint8).flatten()
    page[offset : offset + raw_bytes.numel()] = raw_bytes


def pack_byte_v2_raw_kv_block_to_page(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    valid_rows: int,
    layout: ByteV2PageLayout,
    page: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack a raw partial or fallback K/V block into a Byte-v2 page."""
    if valid_rows < 0 or valid_rows > layout.block_size:
        raise ValueError("valid_rows must be in [0, block_size]")
    key_block = _validate_partial_block(
        key_block,
        max_block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size,
        name="key_block",
    )
    value_block = _validate_partial_block(
        value_block,
        max_block_size=layout.block_size,
        num_kv_heads=layout.num_kv_heads,
        head_size=layout.head_size_v,
        name="value_block",
    )
    if key_block.shape[0] < valid_rows or value_block.shape[0] < valid_rows:
        raise ValueError("raw K/V blocks must contain valid_rows rows")
    if page is None:
        page = torch.zeros(layout.page_size_bytes, dtype=torch.uint8)
    else:
        page = _validate_page(page, layout)
        page.zero_()

    full_key = torch.zeros(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size,
        dtype=torch.bfloat16,
    )
    full_value = torch.zeros(
        layout.block_size,
        layout.num_kv_heads,
        layout.head_size_v,
        dtype=torch.bfloat16,
    )
    full_key[:valid_rows] = key_block[:valid_rows]
    full_value[:valid_rows] = value_block[:valid_rows]

    page[BYTE_V2_PAGE_STATUS_OFFSET] = BYTE_V2_PAGE_STATUS_RAW_FALLBACK
    page[BYTE_V2_PAGE_VALID_ROWS_OFFSET] = valid_rows
    _store_raw_block(page, layout, "k", full_key)
    _store_raw_block(page, layout, "v", full_value)
    return page


def _load_fast_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: ByteV2KVKind,
    kv_head: int,
    dim_tile: int,
) -> torch.Tensor:
    offsets = layout.tile_offsets(kind, kv_head, dim_tile)
    fallback = page[offsets.fallback : offsets.fallback + 1].clone()
    if int(fallback.item()):
        raise NotImplementedError(
            "Byte-v2 page fallback tile decoding is not implemented yet"
        )
    payload = ByteV2TensorPayload(
        base=page[offsets.base : offsets.base + 1].clone(),
        fallback=fallback,
        low_bytes=page[
            offsets.low_bytes : offsets.low_bytes + BYTE_V2_TILE_ELEMS
        ]
        .clone()
        .reshape(1, BYTE_V2_TILE_ELEMS),
        code_packed=page[
            offsets.code_packed : offsets.code_packed + BYTE_V2_PACKED_TILE_ELEMS
        ]
        .clone()
        .reshape(1, BYTE_V2_PACKED_TILE_ELEMS),
        fallback_raw=torch.zeros(1, BYTE_V2_TILE_ELEMS, dtype=torch.bfloat16),
        original_shape=(BYTE_V2_TILE_SIZE, BYTE_V2_TILE_SIZE),
        fallback_tiles=0,
        logical_compressed_bytes=0,
    )
    tile = decompress_byte_v2_tensor(payload)
    return tile.t().contiguous() if kind == "k" else tile


def unpack_byte_v2_kv_block_from_page(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack one Byte-v2 page into BF16 K/V block tensors."""
    if isinstance(layout, ByteV2PageLayoutV3):
        return unpack_byte_v2_kv_block_from_page_v3(page, layout)

    page = _validate_page(page, layout)
    status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
    if status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
        return unpack_byte_v2_raw_kv_block_from_page(page, layout)
    if status != BYTE_V2_PAGE_STATUS_COMPRESSED:
        raise NotImplementedError(
            f"Byte-v2 page status {status} is not supported by the fast layout"
        )

    key_block = torch.empty(
        layout.block_size, layout.num_kv_heads, layout.head_size, dtype=torch.bfloat16
    )
    value_block = torch.empty(
        layout.block_size, layout.num_kv_heads, layout.head_size_v, dtype=torch.bfloat16
    )

    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = _load_fast_tile(
                page, layout, "k", kv_head, dim_tile
            )
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = _load_fast_tile(
                page, layout, "v", kv_head, dim_tile
            )

    return key_block, value_block


def _load_v3_striped_tile(
    page: torch.Tensor,
    layout: ByteV2PageLayoutV3,
    kind: ByteV2KVKind,
    kv_head: int,
    dim_tile: int,
) -> torch.Tensor:
    meta_offsets = layout.kv_head_meta_offsets(kv_head)
    fallback_mask_offset = (
        meta_offsets.k_fallback_mask
        if kind == "k"
        else meta_offsets.v_fallback_mask
    )
    fallback_mask = _load_u16_le(page, fallback_mask_offset)
    if fallback_mask & (1 << dim_tile):
        raise NotImplementedError(
            "Byte-v2 V3 tile fallback decoding requires an external fallback pool"
        )

    base_offset = (
        meta_offsets.k_base + dim_tile
        if kind == "k"
        else meta_offsets.v_base + dim_tile
    )
    low = torch.empty(BYTE_V2_TILE_ELEMS, dtype=torch.uint8)
    packed = torch.empty(BYTE_V2_PACKED_TILE_ELEMS, dtype=torch.uint8)
    tile_offset = layout.tile_payload_offset(kind, kv_head, dim_tile)
    for pair_idx in range(BYTE_V2_PACKED_TILE_ELEMS):
        elem0 = pair_idx * 2
        stripe = pair_idx // BYTE_V2_TILE_PAYLOAD_STRIPE_PAIRS_V3
        lane = pair_idx % BYTE_V2_TILE_PAYLOAD_STRIPE_PAIRS_V3
        stripe_offset = tile_offset + stripe * BYTE_V2_TILE_PAYLOAD_STRIPE_BYTES_V3
        low[elem0] = page[stripe_offset + lane]
        low[elem0 + 1] = page[stripe_offset + 32 + lane]
        packed[pair_idx] = page[stripe_offset + 64 + lane]

    payload = ByteV2TensorPayload(
        base=page[base_offset : base_offset + 1].clone(),
        fallback=torch.zeros(1, dtype=torch.uint8),
        low_bytes=low.reshape(1, BYTE_V2_TILE_ELEMS),
        code_packed=packed.reshape(1, BYTE_V2_PACKED_TILE_ELEMS),
        fallback_raw=torch.zeros(1, BYTE_V2_TILE_ELEMS, dtype=torch.bfloat16),
        original_shape=(BYTE_V2_TILE_SIZE, BYTE_V2_TILE_SIZE),
        fallback_tiles=0,
        logical_compressed_bytes=0,
    )
    tile = decompress_byte_v2_tensor(payload)
    return tile.t().contiguous() if kind == "k" else tile


def unpack_byte_v2_kv_block_from_page_v3(
    page: torch.Tensor,
    layout: ByteV2PageLayoutV3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack one experimental Byte-v2 V3 page into BF16 K/V block tensors."""
    page = _validate_page(page, layout)
    status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
    if status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
        return unpack_byte_v2_raw_kv_block_from_page(page, layout)
    if status != BYTE_V2_PAGE_STATUS_COMPRESSED:
        raise NotImplementedError(
            f"Byte-v2 V3 page status {status} is not supported"
        )
    layout_version = int(page[BYTE_V2_PAGE_LAYOUT_VERSION_OFFSET].item())
    if layout_version != BYTE_V2_PAYLOAD_LAYOUT_VERSION_V3:
        raise ValueError(f"Byte-v2 V3 page has layout version {layout_version}")

    key_block = torch.empty(
        layout.block_size, layout.num_kv_heads, layout.head_size, dtype=torch.bfloat16
    )
    value_block = torch.empty(
        layout.block_size, layout.num_kv_heads, layout.head_size_v, dtype=torch.bfloat16
    )
    for kv_head in range(layout.num_kv_heads):
        for dim_tile in range(layout.k_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            key_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = (
                _load_v3_striped_tile(page, layout, "k", kv_head, dim_tile)
            )
        for dim_tile in range(layout.v_dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            value_block[:, kv_head, d0 : d0 + BYTE_V2_TILE_SIZE] = (
                _load_v3_striped_tile(page, layout, "v", kv_head, dim_tile)
            )
    return key_block, value_block


def _load_raw_block(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
    kind: ByteV2KVKind,
) -> torch.Tensor:
    offset, num_bytes = layout.raw_offsets(kind)
    head_size = layout.head_size if kind == "k" else layout.head_size_v
    raw_bytes = page[offset : offset + num_bytes].contiguous()
    return raw_bytes.view(torch.bfloat16).reshape(
        layout.block_size, layout.num_kv_heads, head_size
    ).clone()


def unpack_byte_v2_raw_kv_block_from_page(
    page: torch.Tensor,
    layout: ByteV2PageLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a raw partial/fallback Byte-v2 page into full-size K/V blocks."""
    page = _validate_page(page, layout)
    status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
    if status != BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
        raise NotImplementedError(f"Byte-v2 page status {status} is not raw")
    return _load_raw_block(page, layout, "k"), _load_raw_block(page, layout, "v")


def byte_v2_reshape_and_cache_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: ByteV2PageLayout,
) -> list[int]:
    """Reference Byte-v2 cache update for complete and partial physical blocks.

    This CPU-only helper mirrors the future CUDA ``byte_v2_reshape_and_cache``
    op at the semantic level. It groups tokens by ``slot_mapping``. Complete
    blocks are compressed and finalized; partial blocks are stored as raw
    fallback pages and can later be finalized when they reach 16 rows.

    Args:
        key: BF16 tensor with shape `[num_tokens, num_kv_heads, head_size]`.
        value: BF16 tensor with shape `[num_tokens, num_kv_heads, head_size_v]`.
        kv_cache: uint8 tensor with shape `[num_blocks, page_size_bytes]`.
        slot_mapping: tensor with one slot per token. Negative slots are ignored.
        layout: Byte-v2 page layout.

    Returns:
        Physical block ids that were packed.
    """
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("Byte-v2 reference cache update expects BF16 key/value")
    expected_key_shape = (key.shape[0], layout.num_kv_heads, layout.head_size)
    expected_value_shape = (key.shape[0], layout.num_kv_heads, layout.head_size_v)
    if tuple(key.shape) != expected_key_shape:
        raise ValueError(f"key must have shape {expected_key_shape}, got {key.shape}")
    if tuple(value.shape) != expected_value_shape:
        raise ValueError(
            f"value must have shape {expected_value_shape}, got {value.shape}"
        )
    kv_cache = _validate_kv_cache(kv_cache, layout)

    key = key.detach().cpu().contiguous()
    value = value.detach().cpu().contiguous()
    slots = slot_mapping.detach().cpu().flatten().to(torch.int64)
    if slots.numel() < key.shape[0]:
        raise ValueError("slot_mapping must contain at least one slot per token")
    slots = slots[: key.shape[0]]

    block_to_tokens: dict[int, dict[int, int]] = {}
    for token_idx, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block_id = slot // layout.block_size
        block_offset = slot % layout.block_size
        if block_id < 0 or block_id >= kv_cache.shape[0]:
            raise ValueError(f"slot_mapping points to invalid block id {block_id}")
        offsets = block_to_tokens.setdefault(block_id, {})
        if block_offset in offsets:
            raise ValueError(
                f"slot_mapping contains duplicate block offset {block_offset} "
                f"for block {block_id}"
            )
        offsets[block_offset] = token_idx

    packed_block_ids: list[int] = []
    for block_id, offsets in sorted(block_to_tokens.items()):
        page = kv_cache[block_id]
        status = int(page[BYTE_V2_PAGE_STATUS_OFFSET].item())
        existing_valid_rows = 0
        if status == BYTE_V2_PAGE_STATUS_RAW_FALLBACK:
            key_block, value_block = unpack_byte_v2_raw_kv_block_from_page(
                page, layout
            )
            existing_valid_rows = int(page[BYTE_V2_PAGE_VALID_ROWS_OFFSET].item())
        elif status == BYTE_V2_PAGE_STATUS_EMPTY:
            key_block = torch.zeros(
                layout.block_size,
                layout.num_kv_heads,
                layout.head_size,
                dtype=torch.bfloat16,
            )
            value_block = torch.zeros(
                layout.block_size,
                layout.num_kv_heads,
                layout.head_size_v,
                dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"cannot update finalized Byte-v2 block {block_id}")

        for block_offset, token_idx in offsets.items():
            key_block[block_offset] = key[token_idx]
            value_block[block_offset] = value[token_idx]

        present_offsets = set(range(existing_valid_rows)) | set(offsets)
        valid_rows = max(present_offsets) + 1
        if present_offsets != set(range(valid_rows)):
            raise ValueError(
                "Byte-v2 partial raw block updates must form a contiguous "
                f"prefix, got offsets {sorted(present_offsets)}"
            )

        if valid_rows != layout.block_size:
            pack_byte_v2_raw_kv_block_to_page(
                key_block, value_block, valid_rows, layout, page=page
            )
            continue

        pack_byte_v2_kv_block_to_page(key_block, value_block, layout, page=page)
        if int(page[BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
            BYTE_V2_PAGE_STATUS_COMPRESSED
        ):
            packed_block_ids.append(block_id)

    return packed_block_ids
