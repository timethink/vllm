# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline packer for the experimental fixed-page Static-W16 FA2 reader.

The 49,792-byte wire is deliberately independent of the production ByteV2
ABI.  A page stores K followed by V, with each side ordered as
``[kv_head, row, dim]``:

* bytes ``[0, 32768)``: one sign/mantissa byte per BF16 value;
* bytes ``[32768, 49152)``: two four-bit exponent codes per byte;
* bytes ``[49152, 49216)``: 16 per-chunk metadata descriptors;
* bytes ``[49216, 49280)``: status, bases, valid rows, and page total;
* bytes ``[49280, 49792)``: one page-shared pool of 128 32-bit escapes.

Within a code byte, the even value occupies the low nibble and the odd value
occupies the high nibble.  The canonical descriptor is
``base | (start << 8) | (count << 16)``. Status bit 1 seals that prepare-time
metadata. The legacy tagged descriptor remains available for reader A/B
tests. An escape is
``(true_exponent << 16) | position``, where position is relative to one
2,048-value K/V-head chunk.

This is an offline experiment packer.  Overflow and active partial pages are
marked in the header and can optionally be copied to a dense raw sidecar for
the direct FA2 reader's exact fallback path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

NUM_SIDES = 2
NUM_KV_HEADS = 8
ALLOC_BLOCK_TOKENS = 16
HEAD_DIM = 128
CHUNKS_PER_PAGE = NUM_SIDES * NUM_KV_HEADS
VALUES_PER_CHUNK = ALLOC_BLOCK_TOKENS * HEAD_DIM
VALUES_PER_PAGE = CHUNKS_PER_PAGE * VALUES_PER_CHUNK
RAW_PAGE_BYTES = VALUES_PER_PAGE * 2

SIGN_MANTISSA_OFFSET = 0
SIGN_MANTISSA_BYTES = VALUES_PER_PAGE
CODE_OFFSET = SIGN_MANTISSA_OFFSET + SIGN_MANTISSA_BYTES
CODE_BYTES = VALUES_PER_PAGE // 2
HEADER_OFFSET = CODE_OFFSET + CODE_BYTES
HEADER_BYTES = 128
ESCAPE_OFFSET = HEADER_OFFSET + HEADER_BYTES
ESCAPE_POOL_ENTRIES = 128
ESCAPE_ENTRY_BYTES = 4
ESCAPE_BYTES = ESCAPE_POOL_ENTRIES * ESCAPE_ENTRY_BYTES
FIXED_PAGE_BYTES = ESCAPE_OFFSET + ESCAPE_BYTES

HEADER_DESCRIPTORS_OFFSET = 0
HEADER_STATUS_OFFSET = 64
HEADER_K_BASE_OFFSET = 68
HEADER_V_BASE_OFFSET = 72
HEADER_VALID_ROWS_OFFSET = 76
HEADER_PAGE_TOTAL_OFFSET = 80
DESCRIPTOR_TAG = 0x5732
STATUS_RAW_FALLBACK = 1
STATUS_CANONICAL_METADATA = 2

DEFAULT_K_EXPONENT_BASE = 115
DEFAULT_V_EXPONENT_BASE = 110

assert VALUES_PER_PAGE == 32_768
assert RAW_PAGE_BYTES == 65_536
assert CODE_OFFSET == 32_768
assert HEADER_OFFSET == 49_152
assert ESCAPE_OFFSET == 49_280
assert FIXED_PAGE_BYTES == 49_792


@dataclass(slots=True)
class StaticW16PackStats:
    """Aggregate statistics for one packing pass."""

    pages: int = 0
    values: int = 0
    escape_entries: int = 0
    max_escape_entries_per_page: int = 0
    max_escape_entries_per_chunk: int = 0
    overflow_chunks: int = 0
    overflow_pages: int = 0
    partial_pages: int = 0
    raw_fallback_pages: int = 0

    @property
    def escape_fraction(self) -> float:
        """Return the escaped fraction among valid values."""
        return self.escape_entries / self.values if self.values else 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["escape_fraction"] = self.escape_fraction
        return result


@dataclass(slots=True)
class StaticW16PackedPages:
    """Fixed pages and optional dense fallback storage."""

    cache: torch.Tensor
    raw_sidecar: torch.Tensor | None
    page_to_raw_slot: torch.Tensor | None
    stats: StaticW16PackStats
    canonical_metadata: bool = False


def _require_page_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use BF16")
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
        ALLOC_BLOCK_TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM,
    ):
        raise ValueError(f"{name} must have shape [pages, 16, 8, 128]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_valid_rows(
    valid_rows: torch.Tensor | None,
    *,
    pages: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_rows is None:
        return torch.full(
            (pages,),
            ALLOC_BLOCK_TOKENS,
            dtype=torch.int32,
            device=device,
        )
    if (
        valid_rows.ndim != 1
        or valid_rows.shape[0] != pages
        or valid_rows.device != device
        or valid_rows.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("valid_rows must be an integer tensor [pages] on key.device")
    if bool(((valid_rows < 0) | (valid_rows > ALLOC_BLOCK_TOKENS)).any().item()):
        raise ValueError("valid_rows values must be in [0, 16]")
    return valid_rows.to(torch.int32).contiguous()


def _normalize_bases(
    base: int | torch.Tensor,
    *,
    name: str,
    pages: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(base, int):
        if not 0 <= base <= 240:
            raise ValueError(f"{name} must be in [0, 240]")
        return torch.full((pages,), base, dtype=torch.int32, device=device)
    if (
        base.ndim != 1
        or base.shape[0] != pages
        or base.device != device
        or base.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError(f"{name} must be an integer tensor [pages] on key.device")
    result = base.to(torch.int32).contiguous()
    if bool(((result < 0) | (result > 240)).any().item()):
        raise ValueError(f"{name} values must be in [0, 240]")
    return result


def _page_wire(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return BF16 pages ordered as ``[page, K/V, head, row, dim]``."""
    return torch.stack(
        (
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        ),
        dim=1,
    ).contiguous()


def _pack_chunk(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    raw_sidecar: torch.Tensor | None,
    page_to_raw_slot: torch.Tensor | None,
    valid_rows: torch.Tensor,
    k_bases: torch.Tensor,
    v_bases: torch.Tensor,
    *,
    page_start: int,
    canonical_metadata: bool,
) -> StaticW16PackStats:
    pages = key.shape[0]
    page_wire = _page_wire(key, value)
    bits = page_wire.view(torch.int16).to(torch.int32) & 0xFFFF
    exponents = (bits >> 7) & 0xFF
    sign_mantissas = ((bits >> 8) & 0x80) | (bits & 0x7F)
    bases = torch.stack((k_bases, v_bases), dim=1).view(pages, 2, 1, 1, 1)
    deltas = exponents - bases
    row_is_valid = torch.arange(
        ALLOC_BLOCK_TOKENS,
        dtype=torch.int32,
        device=key.device,
    ).view(1, 1, 1, ALLOC_BLOCK_TOKENS, 1) < valid_rows.view(pages, 1, 1, 1, 1)
    escapes = row_is_valid & ((deltas < 0) | (deltas >= 16))
    codes = deltas.clamp(0, 15)
    packed_codes = codes[..., 0::2] | (codes[..., 1::2] << 4)

    cache.zero_()
    cache[:, SIGN_MANTISSA_OFFSET:CODE_OFFSET].view(pages, 2, 8, 16, 128).copy_(
        sign_mantissas.to(torch.uint8)
    )
    cache[:, CODE_OFFSET:HEADER_OFFSET].view(pages, 2, 8, 16, 64).copy_(
        packed_codes.to(torch.uint8)
    )

    flat_escapes = escapes.view(pages, CHUNKS_PER_PAGE, VALUES_PER_CHUNK)
    flat_exponents = exponents.view(pages, CHUNKS_PER_PAGE, VALUES_PER_CHUNK)
    counts = flat_escapes.sum(dim=-1, dtype=torch.int32)
    page_escape_counts = counts.sum(dim=1)
    overflow_chunks = counts > ESCAPE_POOL_ENTRIES
    overflow_pages = page_escape_counts > ESCAPE_POOL_ENTRIES
    partial_pages = valid_rows != ALLOC_BLOCK_TOKENS
    fallback_pages = overflow_pages | partial_pages

    header = cache[:, HEADER_OFFSET:ESCAPE_OFFSET].view(torch.int32)
    header.zero_()
    starts = counts.cumsum(dim=1, dtype=torch.int32) - counts
    # Match the native writer's fail-closed wire exactly: descriptors retain
    # the actual reserved range (saturated to their byte fields) even when a
    # raw-fallback page exceeds the physical pool. Compact pages cannot
    # saturate because their aggregate demand is at most 128.
    encoded_counts = counts.clamp(max=0xFF)
    encoded_starts = starts.clamp(max=0xFF)
    if canonical_metadata:
        descriptor_bases = torch.cat(
            (
                k_bases.view(pages, 1).expand(-1, NUM_KV_HEADS),
                v_bases.view(pages, 1).expand(-1, NUM_KV_HEADS),
            ),
            dim=1,
        )
        descriptors = descriptor_bases | (encoded_starts << 8) | (encoded_counts << 16)
    else:
        descriptors = (DESCRIPTOR_TAG << 16) | (encoded_counts << 8) | encoded_starts
    header[:, :CHUNKS_PER_PAGE].copy_(descriptors)
    header[:, HEADER_STATUS_OFFSET // 4].copy_(
        fallback_pages.to(torch.int32) * STATUS_RAW_FALLBACK
        + int(canonical_metadata) * STATUS_CANONICAL_METADATA
    )
    header[:, HEADER_K_BASE_OFFSET // 4].copy_(k_bases)
    header[:, HEADER_V_BASE_OFFSET // 4].copy_(v_bases)
    header[:, HEADER_VALID_ROWS_OFFSET // 4].copy_(valid_rows)
    header[:, HEADER_PAGE_TOTAL_OFFSET // 4].copy_(page_escape_counts)

    pool = cache[:, ESCAPE_OFFSET:].view(torch.int32)
    pool.zero_()
    ranks = flat_escapes.cumsum(dim=-1, dtype=torch.int32) - 1
    pool_slots = starts.unsqueeze(-1) + ranks
    stored = flat_escapes & (pool_slots < ESCAPE_POOL_ENTRIES)
    page_indices, chunk_indices, positions = stored.nonzero(as_tuple=True)
    if page_indices.numel():
        slots = pool_slots[page_indices, chunk_indices, positions].to(torch.int64)
        entries = (
            flat_exponents[page_indices, chunk_indices, positions] << 16
        ) | positions.to(torch.int32)
        pool[page_indices, slots] = entries

    if bool(fallback_pages.any().item()):
        if raw_sidecar is None or page_to_raw_slot is None:
            raise RuntimeError(
                "Static-W16 overflow/partial page requires include_raw_fallback=True"
            )
        local_indices = fallback_pages.nonzero(as_tuple=False).flatten()
        global_indices = local_indices + page_start
        raw_bytes = (
            page_wire[local_indices]
            .view(torch.uint8)
            .reshape(local_indices.numel(), RAW_PAGE_BYTES)
        )
        raw_sidecar[local_indices] = raw_bytes
        page_to_raw_slot[local_indices] = global_indices.to(torch.int32)

    return StaticW16PackStats(
        pages=pages,
        values=int(valid_rows.sum().item()) * NUM_SIDES * NUM_KV_HEADS * HEAD_DIM,
        escape_entries=int(page_escape_counts.sum().item()),
        max_escape_entries_per_page=int(page_escape_counts.max().item()),
        max_escape_entries_per_chunk=int(counts.max().item()),
        overflow_chunks=int(overflow_chunks.sum().item()),
        overflow_pages=int(overflow_pages.sum().item()),
        partial_pages=int(partial_pages.sum().item()),
        raw_fallback_pages=int(fallback_pages.sum().item()),
    )


def pack_static_w16_fixed_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
    k_exponent_base: int | torch.Tensor = DEFAULT_K_EXPONENT_BASE,
    v_exponent_base: int | torch.Tensor = DEFAULT_V_EXPONENT_BASE,
    chunk_pages: int = 64,
    include_raw_fallback: bool = True,
    canonical_metadata: bool = True,
) -> StaticW16PackedPages:
    """Pack raw K/V into the fixed Static-W16 reader wire.

    Args:
        key: Contiguous BF16 tensor shaped ``[pages, 16, 8, 128]``.
        value: Tensor with the same shape, dtype, and device as ``key``.
        valid_rows: Optional valid-row count for each 16-token page.
        k_exponent_base: Scalar or per-page base for K's 16-code window.
        v_exponent_base: Scalar or per-page base for V's 16-code window.
        chunk_pages: Pages vectorized by each offline packing step.
        include_raw_fallback: Allocate a dense test sidecar and page map.
        canonical_metadata: Pack prepare-time ``base/start/count`` descriptors.
            Set false only to exercise the legacy tagged reader.

    Returns:
        Fixed pages, optional raw fallback tensors, and aggregate statistics.

    Raises:
        ValueError: If an input or base does not satisfy the fixed geometry.
        RuntimeError: If a page overflows without raw fallback storage.
    """
    _require_page_tensor("key", key)
    _require_page_tensor("value", value)
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("key and value must have matching shape and device")
    if chunk_pages <= 0:
        raise ValueError("chunk_pages must be positive")
    pages = key.shape[0]
    if pages <= 0:
        raise ValueError("key and value must contain at least one page")
    rows = _normalize_valid_rows(valid_rows, pages=pages, device=key.device)
    k_bases = _normalize_bases(
        k_exponent_base,
        name="k_exponent_base",
        pages=pages,
        device=key.device,
    )
    v_bases = _normalize_bases(
        v_exponent_base,
        name="v_exponent_base",
        pages=pages,
        device=key.device,
    )

    cache = torch.empty((pages, FIXED_PAGE_BYTES), dtype=torch.uint8, device=key.device)
    raw_sidecar = None
    page_to_raw_slot = None
    if include_raw_fallback:
        raw_sidecar = torch.zeros(
            (pages, RAW_PAGE_BYTES), dtype=torch.uint8, device=key.device
        )
        page_to_raw_slot = torch.full(
            (pages,), -1, dtype=torch.int32, device=key.device
        )

    total = StaticW16PackStats()
    for start in range(0, pages, chunk_pages):
        stop = min(start + chunk_pages, pages)
        chunk = _pack_chunk(
            key[start:stop],
            value[start:stop],
            cache[start:stop],
            None if raw_sidecar is None else raw_sidecar[start:stop],
            None if page_to_raw_slot is None else page_to_raw_slot[start:stop],
            rows[start:stop],
            k_bases[start:stop],
            v_bases[start:stop],
            page_start=start,
            canonical_metadata=canonical_metadata,
        )
        total.pages += chunk.pages
        total.values += chunk.values
        total.escape_entries += chunk.escape_entries
        total.max_escape_entries_per_page = max(
            total.max_escape_entries_per_page,
            chunk.max_escape_entries_per_page,
        )
        total.max_escape_entries_per_chunk = max(
            total.max_escape_entries_per_chunk,
            chunk.max_escape_entries_per_chunk,
        )
        total.overflow_chunks += chunk.overflow_chunks
        total.overflow_pages += chunk.overflow_pages
        total.partial_pages += chunk.partial_pages
        total.raw_fallback_pages += chunk.raw_fallback_pages
    return StaticW16PackedPages(
        cache=cache,
        raw_sidecar=raw_sidecar,
        page_to_raw_slot=page_to_raw_slot,
        stats=total,
        canonical_metadata=canonical_metadata,
    )


def unpack_static_w16_fixed_pages(
    packed: StaticW16PackedPages,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the experimental wire for packer correctness tests."""
    cache = packed.cache
    if cache.dtype != torch.uint8 or cache.ndim != 2:
        raise ValueError("cache must be a rank-2 uint8 tensor")
    if cache.shape[1] != FIXED_PAGE_BYTES or not cache.is_contiguous():
        raise ValueError("cache has the wrong Static-W16 page geometry")
    pages = cache.shape[0]
    header = cache[:, HEADER_OFFSET:ESCAPE_OFFSET].view(torch.int32)
    descriptors = header[:, :CHUNKS_PER_PAGE]
    status = header[:, HEADER_STATUS_OFFSET // 4]
    known_status = STATUS_RAW_FALLBACK | STATUS_CANONICAL_METADATA
    if bool((status & ~known_status).any().item()):
        raise ValueError("cache contains an unknown page-status bit")
    canonical = (status & STATUS_CANONICAL_METADATA) != 0
    legacy = ~canonical
    tags = (descriptors >> 16) & 0xFFFF
    if bool((legacy.unsqueeze(1) & (tags != DESCRIPTOR_TAG)).any().item()):
        raise ValueError("cache contains an unknown legacy Static-W16 wire tag")
    if bool((canonical.unsqueeze(1) & ((descriptors >> 24) != 0)).any().item()):
        raise ValueError("canonical Static-W16 descriptor has reserved bits")
    legacy_starts = descriptors & 0xFF
    legacy_counts = (descriptors >> 8) & 0xFF
    canonical_bases = descriptors & 0xFF
    canonical_starts = (descriptors >> 8) & 0xFF
    canonical_counts = (descriptors >> 16) & 0xFF
    starts = torch.where(canonical.unsqueeze(1), canonical_starts, legacy_starts)
    counts = torch.where(canonical.unsqueeze(1), canonical_counts, legacy_counts)
    compact = (status & STATUS_RAW_FALLBACK) == 0
    page_totals = header[:, HEADER_PAGE_TOTAL_OFFSET // 4]
    if bool(
        (compact & ((page_totals < 0) | (page_totals > ESCAPE_POOL_ENTRIES)))
        .any()
        .item()
    ):
        raise ValueError("compact page has an invalid shared-pool total")
    if bool((compact & (counts.sum(dim=1) != page_totals)).any().item()):
        raise ValueError("compact page descriptors do not match page total")
    if bool(
        (compact.unsqueeze(1) & ((starts + counts) > ESCAPE_POOL_ENTRIES)).any().item()
    ):
        raise ValueError("compact page descriptor exceeds the shared pool")
    pool_slots = torch.arange(
        ESCAPE_POOL_ENTRIES, dtype=torch.int32, device=cache.device
    ).view(1, 1, ESCAPE_POOL_ENTRIES)
    range_owners = (pool_slots >= starts.unsqueeze(-1)) & (
        pool_slots < (starts + counts).unsqueeze(-1)
    )
    expected_owners = pool_slots[:, 0] < page_totals.unsqueeze(-1)
    bad_coverage = range_owners.sum(dim=1) != expected_owners.to(torch.int64)
    if bool((bad_coverage & compact.unsqueeze(-1)).any().item()):
        raise ValueError("compact page ranges have a gap or overlap")

    sign_mantissas = (
        cache[:, :CODE_OFFSET]
        .view(pages, NUM_SIDES, NUM_KV_HEADS, ALLOC_BLOCK_TOKENS, HEAD_DIM)
        .to(torch.int32)
    )
    packed_codes = (
        cache[:, CODE_OFFSET:HEADER_OFFSET]
        .view(pages, NUM_SIDES, NUM_KV_HEADS, ALLOC_BLOCK_TOKENS, -1)
        .to(torch.int32)
    )
    codes = torch.empty_like(sign_mantissas)
    codes[..., 0::2] = packed_codes & 0xF
    codes[..., 1::2] = (packed_codes >> 4) & 0xF
    header_bases = (
        torch.stack(
            (
                header[:, HEADER_K_BASE_OFFSET // 4],
                header[:, HEADER_V_BASE_OFFSET // 4],
            ),
            dim=1,
        )
        .view(pages, NUM_SIDES, 1)
        .expand(-1, -1, NUM_KV_HEADS)
    )
    if bool(((header_bases < 0) | (header_bases > 240)).any().item()):
        raise ValueError("Static-W16 header base is out of range")
    canonical_bases_by_side = canonical_bases.view(pages, NUM_SIDES, NUM_KV_HEADS)
    if bool(
        (canonical.view(pages, 1, 1) & (canonical_bases_by_side != header_bases))
        .any()
        .item()
    ):
        raise ValueError("canonical descriptors do not match header bases")
    bases = torch.where(
        canonical.view(pages, 1, 1),
        canonical_bases_by_side,
        header_bases,
    ).view(pages, NUM_SIDES, NUM_KV_HEADS, 1, 1)
    exponents = bases + codes
    bits = (sign_mantissas & 0x7F) | ((sign_mantissas & 0x80) << 8) | (exponents << 7)

    flat_bits = bits.view(pages, CHUNKS_PER_PAGE, VALUES_PER_CHUNK)
    pool = cache[:, ESCAPE_OFFSET:].view(torch.int32)
    slots = torch.arange(
        ESCAPE_POOL_ENTRIES, dtype=torch.int32, device=cache.device
    ).view(1, 1, ESCAPE_POOL_ENTRIES)
    patch = (slots < counts.unsqueeze(-1)) & compact.view(pages, 1, 1)
    page_indices, chunk_indices, slot_indices = patch.nonzero(as_tuple=True)
    if page_indices.numel():
        pool_indices = (starts[page_indices, chunk_indices] + slot_indices).to(
            torch.int64
        )
        entries = pool[page_indices, pool_indices]
        positions = (entries & 0xFFFF).to(torch.int64)
        if bool((positions >= VALUES_PER_CHUNK).any().item()):
            raise ValueError("compact page escape position is out of range")
        true_exponents = (entries >> 16) & 0xFF
        sm = flat_bits[page_indices, chunk_indices, positions]
        flat_bits[page_indices, chunk_indices, positions] = (sm & 0x807F) | (
            true_exponents << 7
        )

    fallback = (status & STATUS_RAW_FALLBACK) != 0
    if bool(fallback.any().item()):
        if packed.raw_sidecar is None or packed.page_to_raw_slot is None:
            raise RuntimeError("raw fallback page is missing its sidecar")
        page_indices = fallback.nonzero(as_tuple=False).flatten()
        raw_slots = packed.page_to_raw_slot[page_indices].to(torch.int64)
        if bool((raw_slots < 0).any().item()):
            raise RuntimeError("raw fallback page has an invalid sidecar slot")
        raw_bits = (
            packed.raw_sidecar[raw_slots]
            .view(torch.int16)
            .view(
                page_indices.numel(),
                NUM_SIDES,
                NUM_KV_HEADS,
                ALLOC_BLOCK_TOKENS,
                HEAD_DIM,
            )
            .to(torch.int32)
            & 0xFFFF
        )
        bits[page_indices] = raw_bits

    decoded = (bits & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    key = decoded[:, 0].permute(0, 2, 1, 3).contiguous()
    value = decoded[:, 1].permute(0, 2, 1, 3).contiguous()
    return key, value
