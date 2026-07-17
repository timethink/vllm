# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ByteV2 layout policy helpers.

This module intentionally only describes layout arithmetic. The CUDA kernels
own the exact wire format, but all tile-related constants should flow through
these policy objects instead of being repeated as literal ``16`` or ``128``
values across Python and CUDA entry points.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

BYTE_V2_MAX_MACRO_PAGES = 8
BYTE_V2_MACRO_DESCRIPTOR_BYTES = 128


def _check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _required_bits(max_inclusive_value: int) -> int:
    _check_positive("max_inclusive_value", max_inclusive_value + 1)
    return max(1, max_inclusive_value.bit_length())


@dataclass(frozen=True, slots=True)
class ByteV2TilePolicy:
    """Tile parameters shared by ByteV2 Python and CUDA code.

    Defaults match the conservative Phase 1 policy:
    16x16 codec tiles, 16-token vLLM allocation blocks, and BN64 compute
    macro tiles for h128 GQA decode.
    """

    codec_token_block: int = 16
    codec_dim_block: int = 16
    alloc_block_tokens: int = 16
    compute_block_n: int = 64
    head_dim: int = 128
    head_dim_v: int = 128

    def __post_init__(self) -> None:
        for name in (
            "codec_token_block",
            "codec_dim_block",
            "alloc_block_tokens",
            "compute_block_n",
            "head_dim",
            "head_dim_v",
        ):
            _check_positive(name, getattr(self, name))

        if self.alloc_block_tokens % self.codec_token_block != 0:
            raise ValueError(
                "alloc_block_tokens must be divisible by codec_token_block"
            )
        if self.head_dim % self.codec_dim_block != 0:
            raise ValueError("head_dim must be divisible by codec_dim_block")
        if self.head_dim_v % self.codec_dim_block != 0:
            raise ValueError("head_dim_v must be divisible by codec_dim_block")
        if self.compute_block_n % self.alloc_block_tokens != 0:
            raise ValueError("compute_block_n must be divisible by alloc_block_tokens")
        if self.codec_tile_elems % 2 != 0:
            raise ValueError("codec tile element count must be even")

    def with_updates(self, **kwargs: int) -> ByteV2TilePolicy:
        """Return a policy copy with selected fields changed."""
        return replace(self, **kwargs)

    @property
    def codec_tile_elems(self) -> int:
        return self.codec_token_block * self.codec_dim_block

    @property
    def codec_packed_elems(self) -> int:
        return self.codec_tile_elems // 2

    @property
    def k_dim_tiles(self) -> int:
        return self.head_dim // self.codec_dim_block

    @property
    def v_dim_tiles(self) -> int:
        return self.head_dim_v // self.codec_dim_block

    @property
    def codec_token_tiles_per_alloc_block(self) -> int:
        return self.alloc_block_tokens // self.codec_token_block

    @property
    def alloc_blocks_per_compute_tile(self) -> int:
        return self.compute_block_n // self.alloc_block_tokens

    @property
    def codec_token_tiles_per_compute_tile(self) -> int:
        return self.compute_block_n // self.codec_token_block

    @property
    def codec_tiles_per_k_page(self) -> int:
        return self.codec_token_tiles_per_alloc_block * self.k_dim_tiles

    @property
    def codec_tiles_per_v_page(self) -> int:
        return self.codec_token_tiles_per_alloc_block * self.v_dim_tiles


DEFAULT_BYTE_V2_TILE_POLICY = ByteV2TilePolicy()


def byte_v2_tile_policy_from_env(
    *,
    block_size: int | None = None,
    head_dim: int | None = None,
    head_dim_v: int | None = None,
) -> ByteV2TilePolicy:
    """Return the default policy with benchmark-only env overrides applied."""
    compute_block_n = DEFAULT_BYTE_V2_TILE_POLICY.compute_block_n
    value = os.environ.get("BYTE_V2_COMPUTE_BLOCK_N")
    if value is not None:
        try:
            parsed = int(value)
        except ValueError:
            parsed = compute_block_n
        if parsed > 0:
            compute_block_n = parsed

    updates = {"compute_block_n": compute_block_n}
    if block_size is not None:
        updates["alloc_block_tokens"] = block_size
    if head_dim is not None:
        updates["head_dim"] = head_dim
    if head_dim_v is not None:
        updates["head_dim_v"] = head_dim_v
    return DEFAULT_BYTE_V2_TILE_POLICY.with_updates(**updates)


@dataclass(frozen=True, slots=True)
class ByteV2CodecPayloadPolicy:
    """Payload sizing for one codec subtile.

    The default assumes one low byte per element and a 4-bit sign-aware high
    code. The field name remains historical because this class only computes
    payload byte sizes.
    """

    low_bytes_per_elem: int = 1
    exponent_code_bits: int = 4
    outlier_high_sideband: bool = False

    def __post_init__(self) -> None:
        _check_positive("low_bytes_per_elem", self.low_bytes_per_elem)
        _check_positive("exponent_code_bits", self.exponent_code_bits)

    def low_bytes_per_codec_tile(self, tile_policy: ByteV2TilePolicy) -> int:
        return tile_policy.codec_tile_elems * self.low_bytes_per_elem

    def code_bytes_per_codec_tile(self, tile_policy: ByteV2TilePolicy) -> int:
        return _ceil_div(tile_policy.codec_tile_elems * self.exponent_code_bits, 8)

    def bytes_per_codec_tile(self, tile_policy: ByteV2TilePolicy) -> int:
        return self.low_bytes_per_codec_tile(
            tile_policy
        ) + self.code_bytes_per_codec_tile(tile_policy)


@dataclass(frozen=True, slots=True)
class ByteV2OutlierEntryPolicy:
    """Bit allocation for outlier sideband element indices and values."""

    elem_bits: int = 8
    value_bits: int = 16

    def __post_init__(self) -> None:
        _check_positive("elem_bits", self.elem_bits)
        _check_positive("value_bits", self.value_bits)

    @classmethod
    def from_tile_policy(
        cls,
        tile_policy: ByteV2TilePolicy = DEFAULT_BYTE_V2_TILE_POLICY,
        *,
        value_bits: int = 16,
    ) -> ByteV2OutlierEntryPolicy:
        return cls(
            elem_bits=_required_bits(tile_policy.codec_tile_elems - 1),
            value_bits=value_bits,
        )

    @property
    def entry_bits(self) -> int:
        return self.elem_bits + self.value_bits

    @property
    def entry_bytes(self) -> int:
        return _ceil_div(self.entry_bits, 8)

    @property
    def max_elem_index(self) -> int:
        return (1 << self.elem_bits) - 1

    @property
    def max_value_bits(self) -> int:
        return (1 << self.value_bits) - 1

    def supports_tile_policy(self, tile_policy: ByteV2TilePolicy) -> bool:
        return self.max_elem_index >= tile_policy.codec_tile_elems - 1

    def encode(self, elem_index: int, value_bits: int) -> int:
        if not 0 <= elem_index <= self.max_elem_index:
            raise ValueError(
                f"elem_index must fit in {self.elem_bits} bits, got {elem_index}"
            )
        if not 0 <= value_bits <= self.max_value_bits:
            raise ValueError(
                f"value_bits must fit in {self.value_bits} bits, got {value_bits}"
            )
        return (value_bits << self.elem_bits) | elem_index

    def decode_elem_index(self, entry: int) -> int:
        return entry & self.max_elem_index

    def decode_value_bits(self, entry: int) -> int:
        return (entry >> self.elem_bits) & self.max_value_bits

    def decode(self, entry: int) -> tuple[int, int]:
        return self.decode_elem_index(entry), self.decode_value_bits(entry)


@dataclass(frozen=True, slots=True)
class ByteV2PageLayoutV4:
    """Page-local V4 layout sizing derived from a tile policy."""

    tile_policy: ByteV2TilePolicy = field(
        default_factory=lambda: DEFAULT_BYTE_V2_TILE_POLICY
    )
    codec_payload_policy: ByteV2CodecPayloadPolicy = field(
        default_factory=ByteV2CodecPayloadPolicy
    )
    num_kv_heads: int = 8
    page_header_bytes: int = 128
    kv_head_meta_bytes: int = 64
    alignment_bytes: int = 128
    raw_elem_bytes: int = 2
    outlier_value_bits: int = 8
    outlier_entries_per_tile: int = 256
    include_raw_payload: bool = False

    def __post_init__(self) -> None:
        for name in (
            "num_kv_heads",
            "page_header_bytes",
            "kv_head_meta_bytes",
            "alignment_bytes",
            "raw_elem_bytes",
            "outlier_value_bits",
            "outlier_entries_per_tile",
        ):
            _check_positive(name, getattr(self, name))
        if self.kv_head_meta_bytes < self.kv_head_required_meta_bytes:
            raise ValueError(
                "kv_head_meta_bytes is smaller than required ByteV2 metadata"
            )
        if (
            not self.include_raw_payload
            and self.outlier_entries_per_tile < self.tile_policy.codec_tile_elems
        ):
            raise ValueError(
                "outlier_entries_per_tile must cover the full codec tile when "
                "raw payload is disabled"
            )

    @property
    def macro_pages(self) -> int:
        return self.tile_policy.alloc_blocks_per_compute_tile

    @property
    def metadata_bytes(self) -> int:
        return self.page_header_bytes + self.num_kv_heads * self.kv_head_meta_bytes

    @property
    def aligned_metadata_bytes(self) -> int:
        return _align_up(self.metadata_bytes, self.alignment_bytes)

    @property
    def codec_payload_bytes_per_tile(self) -> int:
        return self.codec_payload_policy.bytes_per_codec_tile(self.tile_policy)

    @property
    def k_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.codec_tiles_per_k_page * self.codec_payload_bytes_per_tile
        )

    @property
    def v_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.codec_tiles_per_v_page * self.codec_payload_bytes_per_tile
        )

    @property
    def aligned_k_payload_bytes_per_kv_head(self) -> int:
        return _align_up(self.k_payload_bytes_per_kv_head, self.alignment_bytes)

    @property
    def aligned_v_payload_bytes_per_kv_head(self) -> int:
        return _align_up(self.v_payload_bytes_per_kv_head, self.alignment_bytes)

    @property
    def payload_bytes(self) -> int:
        return (
            self.compressed_payload_bytes
            + self.outlier_payload_bytes
            + self.raw_payload_bytes
        )

    @property
    def compressed_payload_bytes(self) -> int:
        return self.num_kv_heads * (
            self.aligned_k_payload_bytes_per_kv_head
            + self.aligned_v_payload_bytes_per_kv_head
        )

    @property
    def outlier_entry_policy(self) -> ByteV2OutlierEntryPolicy:
        return ByteV2OutlierEntryPolicy.from_tile_policy(
            self.tile_policy,
            value_bits=self.outlier_value_bits,
        )

    @property
    def outlier_entry_bytes(self) -> int:
        return self.outlier_entry_policy.entry_bytes

    @property
    def outlier_payload_bytes_per_tile(self) -> int:
        return self.outlier_entries_per_tile * self.outlier_entry_bytes

    @property
    def k_outlier_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.codec_tiles_per_k_page
            * self.outlier_payload_bytes_per_tile
        )

    @property
    def v_outlier_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.codec_tiles_per_v_page
            * self.outlier_payload_bytes_per_tile
        )

    @property
    def aligned_k_outlier_payload_bytes_per_kv_head(self) -> int:
        return _align_up(
            self.k_outlier_payload_bytes_per_kv_head,
            self.alignment_bytes,
        )

    @property
    def aligned_v_outlier_payload_bytes_per_kv_head(self) -> int:
        return _align_up(
            self.v_outlier_payload_bytes_per_kv_head,
            self.alignment_bytes,
        )

    @property
    def outlier_payload_bytes(self) -> int:
        return self.num_kv_heads * (
            self.aligned_k_outlier_payload_bytes_per_kv_head
            + self.aligned_v_outlier_payload_bytes_per_kv_head
        )

    @property
    def raw_k_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.alloc_block_tokens
            * self.tile_policy.head_dim
            * self.raw_elem_bytes
        )

    @property
    def raw_v_payload_bytes_per_kv_head(self) -> int:
        return (
            self.tile_policy.alloc_block_tokens
            * self.tile_policy.head_dim_v
            * self.raw_elem_bytes
        )

    @property
    def aligned_raw_k_payload_bytes_per_kv_head(self) -> int:
        return _align_up(self.raw_k_payload_bytes_per_kv_head, self.alignment_bytes)

    @property
    def aligned_raw_v_payload_bytes_per_kv_head(self) -> int:
        return _align_up(self.raw_v_payload_bytes_per_kv_head, self.alignment_bytes)

    @property
    def raw_payload_bytes(self) -> int:
        if not self.include_raw_payload:
            return 0
        return self.num_kv_heads * (
            self.aligned_raw_k_payload_bytes_per_kv_head
            + self.aligned_raw_v_payload_bytes_per_kv_head
        )

    @property
    def page_size_bytes(self) -> int:
        return self.aligned_metadata_bytes + self.payload_bytes

    @property
    def k_payload_base_bytes(self) -> int:
        return self.aligned_metadata_bytes

    @property
    def v_payload_base_bytes(self) -> int:
        return (
            self.k_payload_base_bytes
            + self.num_kv_heads * self.aligned_k_payload_bytes_per_kv_head
        )

    @property
    def raw_k_payload_base_bytes(self) -> int:
        return (
            self.v_outlier_payload_base_bytes
            + self.num_kv_heads * self.aligned_v_outlier_payload_bytes_per_kv_head
        )

    @property
    def raw_v_payload_base_bytes(self) -> int:
        if not self.include_raw_payload:
            return self.raw_k_payload_base_bytes
        return (
            self.raw_k_payload_base_bytes
            + self.num_kv_heads * self.aligned_raw_k_payload_bytes_per_kv_head
        )

    @property
    def k_outlier_payload_base_bytes(self) -> int:
        return (
            self.v_payload_base_bytes
            + self.num_kv_heads * self.aligned_v_payload_bytes_per_kv_head
        )

    @property
    def v_outlier_payload_base_bytes(self) -> int:
        return (
            self.k_outlier_payload_base_bytes
            + self.num_kv_heads * self.aligned_k_outlier_payload_bytes_per_kv_head
        )

    def kv_head_meta_offset(self, *, kv_head: int) -> int:
        """Return the page-local metadata base offset for one KV head."""
        self._check_kv_head(kv_head)
        return self.page_header_bytes + kv_head * self.kv_head_meta_bytes

    def k_fallback_mask_offset(self, *, kv_head: int) -> int:
        """Return the page-local K tile fallback mask offset."""
        return self.kv_head_meta_offset(kv_head=kv_head)

    def v_fallback_mask_offset(self, *, kv_head: int) -> int:
        """Return the page-local V tile fallback mask offset."""
        return self.kv_head_meta_offset(kv_head=kv_head) + 4

    def k_outlier_mask_offset(self, *, kv_head: int) -> int:
        """Return the page-local K tile outlier overlay mask offset."""
        return self.kv_head_meta_offset(kv_head=kv_head) + 24

    def v_outlier_mask_offset(self, *, kv_head: int) -> int:
        """Return the page-local V tile outlier overlay mask offset."""
        return self.kv_head_meta_offset(kv_head=kv_head) + 28

    @property
    def kv_head_required_meta_bytes(self) -> int:
        return (
            32
            + self.tile_policy.codec_tiles_per_k_page
            + self.tile_policy.codec_tiles_per_v_page
        )

    def k_base_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local K tile high-byte base offset."""
        self._check_dim_tile(dim_tile, self.tile_policy.k_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = (
            dim_tile * self.tile_policy.codec_token_tiles_per_alloc_block + token_tile
        )
        return self.kv_head_meta_offset(kv_head=kv_head) + 8 + tile_index

    def v_base_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local V tile high-byte base offset."""
        self._check_dim_tile(dim_tile, self.tile_policy.v_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = token_tile * self.tile_policy.v_dim_tiles + dim_tile
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 8
            + self.tile_policy.codec_tiles_per_k_page
            + tile_index
        )

    def k_outlier_count_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local K tile outlier entry count offset."""
        self._check_dim_tile(dim_tile, self.tile_policy.k_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = (
            dim_tile * self.tile_policy.codec_token_tiles_per_alloc_block + token_tile
        )
        return self.kv_head_meta_offset(kv_head=kv_head) + 32 + tile_index

    def v_outlier_count_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local V tile outlier entry count offset."""
        self._check_dim_tile(dim_tile, self.tile_policy.v_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = token_tile * self.tile_policy.v_dim_tiles + dim_tile
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 32
            + self.tile_policy.codec_tiles_per_k_page
            + tile_index
        )

    def _check_kv_head(self, kv_head: int) -> None:
        if not 0 <= kv_head < self.num_kv_heads:
            raise ValueError(f"kv_head must be in [0, {self.num_kv_heads})")

    def _check_token_tile(self, token_tile: int) -> None:
        limit = self.tile_policy.codec_token_tiles_per_alloc_block
        if not 0 <= token_tile < limit:
            raise ValueError(f"token_tile must be in [0, {limit})")

    @staticmethod
    def _check_dim_tile(dim_tile: int, limit: int) -> None:
        if not 0 <= dim_tile < limit:
            raise ValueError(f"dim_tile must be in [0, {limit})")

    def k_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local byte offset for one K codec tile."""
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.k_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = (
            dim_tile * self.tile_policy.codec_token_tiles_per_alloc_block + token_tile
        )
        return (
            self.k_payload_base_bytes
            + kv_head * self.aligned_k_payload_bytes_per_kv_head
            + tile_index * self.codec_payload_bytes_per_tile
        )

    def v_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        """Return the page-local byte offset for one V codec tile."""
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.v_dim_tiles)
        self._check_token_tile(token_tile)
        tile_index = token_tile * self.tile_policy.v_dim_tiles + dim_tile
        return (
            self.v_payload_base_bytes
            + kv_head * self.aligned_v_payload_bytes_per_kv_head
            + tile_index * self.codec_payload_bytes_per_tile
        )

    def k_outlier_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
        entry_idx: int = 0,
    ) -> int:
        """Return the page-local byte offset for one K outlier entry."""
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.k_dim_tiles)
        self._check_token_tile(token_tile)
        if not 0 <= entry_idx < self.outlier_entries_per_tile:
            raise ValueError("entry_idx must be inside the outlier tile payload")
        tile_index = (
            dim_tile * self.tile_policy.codec_token_tiles_per_alloc_block + token_tile
        )
        return (
            self.k_outlier_payload_base_bytes
            + kv_head * self.aligned_k_outlier_payload_bytes_per_kv_head
            + tile_index * self.outlier_payload_bytes_per_tile
            + entry_idx * self.outlier_entry_bytes
        )

    def v_outlier_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
        entry_idx: int = 0,
    ) -> int:
        """Return the page-local byte offset for one V outlier entry."""
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.v_dim_tiles)
        self._check_token_tile(token_tile)
        if not 0 <= entry_idx < self.outlier_entries_per_tile:
            raise ValueError("entry_idx must be inside the outlier tile payload")
        tile_index = token_tile * self.tile_policy.v_dim_tiles + dim_tile
        return (
            self.v_outlier_payload_base_bytes
            + kv_head * self.aligned_v_outlier_payload_bytes_per_kv_head
            + tile_index * self.outlier_payload_bytes_per_tile
            + entry_idx * self.outlier_entry_bytes
        )

    def raw_key_offset(self, *, kv_head: int, row: int, dim: int) -> int:
        """Return the page-local byte offset for one raw bf16 K element."""
        if not self.include_raw_payload:
            raise ValueError("raw payload is disabled for this ByteV2 layout")
        self._check_kv_head(kv_head)
        if not 0 <= row < self.tile_policy.alloc_block_tokens:
            raise ValueError("row must be inside the allocation block")
        if not 0 <= dim < self.tile_policy.head_dim:
            raise ValueError("dim must be inside the K head dimension")
        return (
            self.raw_k_payload_base_bytes
            + kv_head * self.aligned_raw_k_payload_bytes_per_kv_head
            + (row * self.tile_policy.head_dim + dim) * self.raw_elem_bytes
        )

    def raw_value_offset(self, *, kv_head: int, row: int, dim: int) -> int:
        """Return the page-local byte offset for one raw bf16 V element."""
        if not self.include_raw_payload:
            raise ValueError("raw payload is disabled for this ByteV2 layout")
        self._check_kv_head(kv_head)
        if not 0 <= row < self.tile_policy.alloc_block_tokens:
            raise ValueError("row must be inside the allocation block")
        if not 0 <= dim < self.tile_policy.head_dim_v:
            raise ValueError("dim must be inside the V head dimension")
        return (
            self.raw_v_payload_base_bytes
            + kv_head * self.aligned_raw_v_payload_bytes_per_kv_head
            + (row * self.tile_policy.head_dim_v + dim) * self.raw_elem_bytes
        )


@dataclass(frozen=True, slots=True)
class ByteV2PageLayoutV5(ByteV2PageLayoutV4):
    """Compact V5 layout with one shared outlier pool per cache page."""

    kv_head_meta_bytes: int = 96
    outlier_pool_entries: int = 1024

    def __post_init__(self) -> None:
        ByteV2PageLayoutV4.__post_init__(self)
        _check_positive("outlier_pool_entries", self.outlier_pool_entries)
        if self.include_raw_payload:
            raise ValueError("ByteV2 V5 does not store a page-local raw payload")
        if self.outlier_pool_entries < self.tile_policy.codec_tile_elems:
            raise ValueError("outlier_pool_entries must hold one full codec tile")
        if self.outlier_pool_entries > 65535:
            raise ValueError("outlier_pool_entries must fit in a uint16 offset")

    @property
    def kv_head_required_meta_bytes(self) -> int:
        tiles = (
            self.tile_policy.codec_tiles_per_k_page
            + self.tile_policy.codec_tiles_per_v_page
        )
        return 32 + 4 * tiles

    @property
    def outlier_pool_bytes(self) -> int:
        return self.outlier_pool_entries * self.outlier_entry_bytes

    @property
    def outlier_payload_bytes(self) -> int:
        return self.outlier_pool_bytes

    @property
    def payload_bytes(self) -> int:
        return self.compressed_payload_bytes + self.outlier_pool_bytes

    @property
    def outlier_pool_base_bytes(self) -> int:
        return (
            self.v_payload_base_bytes
            + self.num_kv_heads * self.aligned_v_payload_bytes_per_kv_head
        )

    @property
    def k_outlier_payload_base_bytes(self) -> int:
        return self.outlier_pool_base_bytes

    @property
    def v_outlier_payload_base_bytes(self) -> int:
        return self.outlier_pool_base_bytes

    @property
    def raw_k_payload_base_bytes(self) -> int:
        return self.outlier_pool_base_bytes + self.outlier_pool_bytes

    @property
    def raw_v_payload_base_bytes(self) -> int:
        return self.raw_k_payload_base_bytes

    @property
    def outlier_pool_used_offset(self) -> int:
        return 0

    @property
    def outlier_pool_overflow_offset(self) -> int:
        return 4

    def _k_tile_index(self, dim_tile: int, token_tile: int) -> int:
        return (
            dim_tile * self.tile_policy.codec_token_tiles_per_alloc_block + token_tile
        )

    def _v_tile_index(self, dim_tile: int, token_tile: int) -> int:
        return token_tile * self.tile_policy.v_dim_tiles + dim_tile

    def k_outlier_count_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.k_dim_tiles)
        self._check_token_tile(token_tile)
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 32
            + 2 * self._k_tile_index(dim_tile, token_tile)
        )

    def v_outlier_count_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        self._check_kv_head(kv_head)
        self._check_dim_tile(dim_tile, self.tile_policy.v_dim_tiles)
        self._check_token_tile(token_tile)
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 32
            + 2
            * (
                self.tile_policy.codec_tiles_per_k_page
                + self._v_tile_index(dim_tile, token_tile)
            )
        )

    def k_outlier_pool_index_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        tiles = (
            self.tile_policy.codec_tiles_per_k_page
            + self.tile_policy.codec_tiles_per_v_page
        )
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 32
            + 2 * tiles
            + 2 * self._k_tile_index(dim_tile, token_tile)
        )

    def v_outlier_pool_index_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
    ) -> int:
        tiles = (
            self.tile_policy.codec_tiles_per_k_page
            + self.tile_policy.codec_tiles_per_v_page
        )
        return (
            self.kv_head_meta_offset(kv_head=kv_head)
            + 32
            + 2 * tiles
            + 2
            * (
                self.tile_policy.codec_tiles_per_k_page
                + self._v_tile_index(dim_tile, token_tile)
            )
        )

    def k_outlier_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
        pool_entry_index: int = 0,
        entry_idx: int = 0,
    ) -> int:
        del kv_head, dim_tile, token_tile
        if not 0 <= pool_entry_index < self.outlier_pool_entries:
            raise ValueError("pool_entry_index must be inside the outlier pool")
        if not 0 <= pool_entry_index + entry_idx < self.outlier_pool_entries:
            raise ValueError("entry_idx must be inside the outlier pool")
        return (
            self.outlier_pool_base_bytes
            + (pool_entry_index + entry_idx) * self.outlier_entry_bytes
        )

    def v_outlier_payload_offset(
        self,
        *,
        kv_head: int,
        dim_tile: int,
        token_tile: int = 0,
        pool_entry_index: int = 0,
        entry_idx: int = 0,
    ) -> int:
        return self.k_outlier_payload_offset(
            kv_head=kv_head,
            dim_tile=dim_tile,
            token_tile=token_tile,
            pool_entry_index=pool_entry_index,
            entry_idx=entry_idx,
        )


@dataclass(frozen=True, slots=True)
class ByteV2FallbackStats:
    """Summary of page-local tile fallback and overlay potential."""

    total_tiles: int
    fallback_tiles: int
    outlier_entries: int
    raw_tile_bytes: int
    estimated_overlay_bytes: int
    overlay_tiles: int = 0
    overlay_entries: int = 0
    overlay_bytes: int = 0

    @property
    def fallback_ratio(self) -> float:
        return self.fallback_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def overlay_ratio(self) -> float:
        return self.overlay_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def outlier_entries_per_fallback_tile(self) -> float:
        return (
            self.outlier_entries / self.fallback_tiles if self.fallback_tiles else 0.0
        )

    @property
    def overlay_to_raw_tile_bytes_ratio(self) -> float:
        return (
            self.estimated_overlay_bytes / self.raw_tile_bytes
            if self.raw_tile_bytes
            else 0.0
        )


def _page_byte(page: Sequence[int], offset: int) -> int:
    return int(page[offset])


def _page_u16(page: Sequence[int], offset: int) -> int:
    return _page_byte(page, offset) | (_page_byte(page, offset + 1) << 8)


def _page_u32(page: Sequence[int], offset: int) -> int:
    return (
        _page_byte(page, offset)
        | (_page_byte(page, offset + 1) << 8)
        | (_page_byte(page, offset + 2) << 16)
        | (_page_byte(page, offset + 3) << 24)
    )


def _best_high_byte_window_outliers(high_bytes: list[int]) -> int:
    counts = [0] * 128
    for high_byte in high_bytes:
        counts[high_byte & 0x7F] += 1
    best_in_window = 0
    for base in range(121):
        best_in_window = max(best_in_window, sum(counts[base : base + 8]))
    return len(high_bytes) - best_in_window


def _fallback_tile_outlier_entries(
    page: Sequence[int],
    *,
    layout: ByteV2PageLayoutV4,
    kv_head: int,
    tile_index: int,
    is_value: bool,
) -> int:
    policy = layout.tile_policy
    if is_value:
        token_tile = tile_index // policy.v_dim_tiles
        dim_tile = tile_index % policy.v_dim_tiles
        raw_offset = layout.raw_value_offset
        head_dim = policy.head_dim_v
    else:
        dim_tile = tile_index // policy.codec_token_tiles_per_alloc_block
        token_tile = tile_index % policy.codec_token_tiles_per_alloc_block
        raw_offset = layout.raw_key_offset
        head_dim = policy.head_dim

    high_bytes = []
    row_start = token_tile * policy.codec_token_block
    dim_start = dim_tile * policy.codec_dim_block
    for row_offset in range(policy.codec_token_block):
        row = row_start + row_offset
        if row >= policy.alloc_block_tokens:
            break
        for dim_offset in range(policy.codec_dim_block):
            dim = dim_start + dim_offset
            if dim >= head_dim:
                break
            offset = raw_offset(kv_head=kv_head, row=row, dim=dim)
            high_bytes.append(_page_u16(page, offset) >> 8)
    return _best_high_byte_window_outliers(high_bytes)


def collect_byte_v2_fallback_stats(
    pages: Sequence[Sequence[int]],
    *,
    layout: ByteV2PageLayoutV4 | None = None,
    outlier_policy: ByteV2OutlierEntryPolicy | None = None,
) -> ByteV2FallbackStats:
    """Collect fallback-tile stats from CPU-visible ByteV2 cache pages.

    The overlay estimate models a high-byte outlier entry: one codec-tile
    element index plus an 8-bit bf16 high byte. Low bytes are already present in
    the compressed payload.
    """
    if layout is None:
        layout = ByteV2PageLayoutV4()
    if outlier_policy is None:
        outlier_policy = ByteV2OutlierEntryPolicy.from_tile_policy(
            layout.tile_policy,
            value_bits=8,
        )
    if not outlier_policy.supports_tile_policy(layout.tile_policy):
        raise ValueError("outlier_policy must support the layout tile policy")

    fallback_tiles = 0
    outlier_entries = 0
    overlay_tiles = 0
    overlay_entries = 0
    for page in pages:
        for kv_head in range(layout.num_kv_heads):
            k_mask = _page_u32(page, layout.k_fallback_mask_offset(kv_head=kv_head))
            v_mask = _page_u32(page, layout.v_fallback_mask_offset(kv_head=kv_head))
            k_outlier_mask = _page_u32(
                page,
                layout.k_outlier_mask_offset(kv_head=kv_head),
            )
            v_outlier_mask = _page_u32(
                page,
                layout.v_outlier_mask_offset(kv_head=kv_head),
            )
            for tile_index in range(layout.tile_policy.codec_tiles_per_k_page):
                if k_mask & (1 << tile_index):
                    fallback_tiles += 1
                    outlier_entries += _fallback_tile_outlier_entries(
                        page,
                        layout=layout,
                        kv_head=kv_head,
                        tile_index=tile_index,
                        is_value=False,
                    )
                if k_outlier_mask & (1 << tile_index):
                    dim_tile = (
                        tile_index
                        // layout.tile_policy.codec_token_tiles_per_alloc_block
                    )
                    token_tile = (
                        tile_index
                        % layout.tile_policy.codec_token_tiles_per_alloc_block
                    )
                    overlay_tiles += 1
                    overlay_entries += _page_byte(
                        page,
                        layout.k_outlier_count_offset(
                            kv_head=kv_head,
                            dim_tile=dim_tile,
                            token_tile=token_tile,
                        ),
                    )
            for tile_index in range(layout.tile_policy.codec_tiles_per_v_page):
                if v_mask & (1 << tile_index):
                    fallback_tiles += 1
                    outlier_entries += _fallback_tile_outlier_entries(
                        page,
                        layout=layout,
                        kv_head=kv_head,
                        tile_index=tile_index,
                        is_value=True,
                    )
                if v_outlier_mask & (1 << tile_index):
                    token_tile = tile_index // layout.tile_policy.v_dim_tiles
                    dim_tile = tile_index % layout.tile_policy.v_dim_tiles
                    overlay_tiles += 1
                    overlay_entries += _page_byte(
                        page,
                        layout.v_outlier_count_offset(
                            kv_head=kv_head,
                            dim_tile=dim_tile,
                            token_tile=token_tile,
                        ),
                    )

    total_tiles = (
        len(pages)
        * layout.num_kv_heads
        * (
            layout.tile_policy.codec_tiles_per_k_page
            + layout.tile_policy.codec_tiles_per_v_page
        )
    )
    raw_tile_bytes = (
        fallback_tiles * layout.tile_policy.codec_tile_elems * layout.raw_elem_bytes
    )
    return ByteV2FallbackStats(
        total_tiles=total_tiles,
        fallback_tiles=fallback_tiles,
        outlier_entries=outlier_entries,
        raw_tile_bytes=raw_tile_bytes,
        estimated_overlay_bytes=outlier_entries * outlier_policy.entry_bytes,
        overlay_tiles=overlay_tiles,
        overlay_entries=overlay_entries,
        overlay_bytes=overlay_entries * outlier_policy.entry_bytes,
    )


@dataclass(frozen=True, slots=True)
class ByteV2RawStagingLayout:
    """Raw bf16 staging slot used by decode append before page finalization."""

    tile_policy: ByteV2TilePolicy = field(
        default_factory=lambda: DEFAULT_BYTE_V2_TILE_POLICY
    )
    num_kv_heads: int = 8
    raw_elem_bytes: int = 2
    alignment_bytes: int = 128

    def __post_init__(self) -> None:
        for name in ("num_kv_heads", "raw_elem_bytes", "alignment_bytes"):
            _check_positive(name, getattr(self, name))

    @property
    def key_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.tile_policy.alloc_block_tokens
            * self.tile_policy.head_dim
            * self.raw_elem_bytes
        )

    @property
    def value_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.tile_policy.alloc_block_tokens
            * self.tile_policy.head_dim_v
            * self.raw_elem_bytes
        )

    @property
    def aligned_key_bytes(self) -> int:
        return _align_up(self.key_bytes, self.alignment_bytes)

    @property
    def aligned_value_bytes(self) -> int:
        return _align_up(self.value_bytes, self.alignment_bytes)

    @property
    def value_base_bytes(self) -> int:
        return self.aligned_key_bytes

    @property
    def slot_size_bytes(self) -> int:
        return self.aligned_key_bytes + self.aligned_value_bytes

    def _check_kv_head(self, kv_head: int) -> None:
        if not 0 <= kv_head < self.num_kv_heads:
            raise ValueError(f"kv_head must be in [0, {self.num_kv_heads})")

    def _check_row(self, row: int) -> None:
        limit = self.tile_policy.alloc_block_tokens
        if not 0 <= row < limit:
            raise ValueError(f"row must be in [0, {limit})")

    @staticmethod
    def _check_dim(dim: int, limit: int) -> None:
        if not 0 <= dim < limit:
            raise ValueError(f"dim must be in [0, {limit})")

    def key_offset(self, *, kv_head: int, row: int, dim: int) -> int:
        """Return the slot-local byte offset for one raw K element."""
        self._check_kv_head(kv_head)
        self._check_row(row)
        self._check_dim(dim, self.tile_policy.head_dim)
        return (
            (kv_head * self.tile_policy.alloc_block_tokens + row)
            * self.tile_policy.head_dim
            + dim
        ) * self.raw_elem_bytes

    def value_offset(self, *, kv_head: int, row: int, dim: int) -> int:
        """Return the slot-local byte offset for one raw V element."""
        self._check_kv_head(kv_head)
        self._check_row(row)
        self._check_dim(dim, self.tile_policy.head_dim_v)
        return (
            self.value_base_bytes
            + (
                (kv_head * self.tile_policy.alloc_block_tokens + row)
                * self.tile_policy.head_dim_v
                + dim
            )
            * self.raw_elem_bytes
        )


@dataclass(frozen=True, slots=True)
class ByteV2MacroDescriptor:
    """Fixed-width macro descriptor consumed by future decode kernels."""

    active_pages: int
    physical_blocks: tuple[int, ...]
    valid_rows: tuple[int, ...]
    compressed_mask: int
    outlier_page_mask: int
    k_payload_offsets: tuple[int, ...]
    v_payload_offsets: tuple[int, ...]
    descriptor_bytes: int = BYTE_V2_MACRO_DESCRIPTOR_BYTES

    def __post_init__(self) -> None:
        if not 0 <= self.active_pages <= BYTE_V2_MAX_MACRO_PAGES:
            raise ValueError(f"active_pages must be in [0, {BYTE_V2_MAX_MACRO_PAGES}]")
        for name in (
            "physical_blocks",
            "valid_rows",
            "k_payload_offsets",
            "v_payload_offsets",
        ):
            if len(getattr(self, name)) != BYTE_V2_MAX_MACRO_PAGES:
                raise ValueError(f"{name} must have {BYTE_V2_MAX_MACRO_PAGES} slots")
        max_mask = (1 << BYTE_V2_MAX_MACRO_PAGES) - 1
        if self.compressed_mask & ~max_mask:
            raise ValueError("compressed_mask has bits outside the descriptor")
        if self.outlier_page_mask & ~max_mask:
            raise ValueError("outlier_page_mask has bits outside the descriptor")
        if self.descriptor_bytes % BYTE_V2_MACRO_DESCRIPTOR_BYTES != 0:
            raise ValueError("descriptor_bytes must be 128-byte aligned")


@dataclass(frozen=True, slots=True)
class ByteV2PagedKVManager:
    """Page table helper for FlashAttention-style ByteV2 decode kernels."""

    layout: ByteV2PageLayoutV4 = field(default_factory=ByteV2PageLayoutV4)
    max_macro_pages: int = BYTE_V2_MAX_MACRO_PAGES

    def __post_init__(self) -> None:
        _check_positive("max_macro_pages", self.max_macro_pages)
        if self.layout.macro_pages > self.max_macro_pages:
            raise ValueError(
                "layout.macro_pages must not exceed descriptor macro pages"
            )

    def build_macro_descriptor(
        self,
        block_table: Sequence[int],
        *,
        first_block_idx: int,
        seq_len: int,
        kv_head: int,
        compressed_mask: int | None = None,
        outlier_page_mask: int = 0,
    ) -> ByteV2MacroDescriptor:
        if first_block_idx < 0:
            raise ValueError("first_block_idx must be non-negative")
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")

        active_pages = self.layout.macro_pages
        valid_mask = 0
        physical_blocks = [-1] * self.max_macro_pages
        valid_rows = [0] * self.max_macro_pages
        k_offsets = [0] * self.max_macro_pages
        v_offsets = [0] * self.max_macro_pages
        k_payload_offset = self.layout.k_payload_offset(
            kv_head=kv_head,
            dim_tile=0,
        )
        v_payload_offset = self.layout.v_payload_offset(
            kv_head=kv_head,
            dim_tile=0,
        )

        for page_idx in range(active_pages):
            block_idx = first_block_idx + page_idx
            token_start = block_idx * self.layout.tile_policy.alloc_block_tokens
            rows = min(
                max(seq_len - token_start, 0),
                self.layout.tile_policy.alloc_block_tokens,
            )
            if rows == 0:
                continue
            if block_idx >= len(block_table):
                raise ValueError("block_table is shorter than the requested macro tile")
            physical_block = int(block_table[block_idx])
            if physical_block < 0:
                raise ValueError("physical block ids must be non-negative")
            physical_blocks[page_idx] = physical_block
            valid_rows[page_idx] = rows
            k_offsets[page_idx] = k_payload_offset
            v_offsets[page_idx] = v_payload_offset
            valid_mask |= 1 << page_idx

        if compressed_mask is None:
            compressed_mask = valid_mask
        if compressed_mask & ~valid_mask:
            raise ValueError("compressed_mask cannot include invalid pages")
        if outlier_page_mask & ~valid_mask:
            raise ValueError("outlier_page_mask cannot include invalid pages")

        return ByteV2MacroDescriptor(
            active_pages=active_pages,
            physical_blocks=tuple(physical_blocks),
            valid_rows=tuple(valid_rows),
            compressed_mask=compressed_mask,
            outlier_page_mask=outlier_page_mask,
            k_payload_offsets=tuple(k_offsets),
            v_payload_offsets=tuple(v_offsets),
        )
