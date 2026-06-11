# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference Byte-v2 BF16 tile codec.

This module is intentionally a small CPU/PyTorch reference implementation. It
is used for unit tests and as an oracle for the CUDA cache-update kernel added
in later Byte-v2 integration steps.
"""

from dataclasses import dataclass

import torch

BYTE_V2_TILE_SIZE = 16
BYTE_V2_TILE_ELEMS = BYTE_V2_TILE_SIZE * BYTE_V2_TILE_SIZE
BYTE_V2_PACKED_TILE_ELEMS = BYTE_V2_TILE_ELEMS // 2
BYTE_V2_FAST_TILE_PAYLOAD_BYTES = (
    1 + BYTE_V2_TILE_ELEMS + BYTE_V2_PACKED_TILE_ELEMS + 1
)


@dataclass(frozen=True)
class ByteV2TensorPayload:
    base: torch.Tensor
    fallback: torch.Tensor
    low_bytes: torch.Tensor
    code_packed: torch.Tensor
    fallback_raw: torch.Tensor
    original_shape: tuple[int, int]
    fallback_tiles: int
    logical_compressed_bytes: int

    @property
    def total_tiles(self) -> int:
        return int(self.base.numel())


def _bf16_to_u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _u16_to_bf16(bits: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    signed = bits.to(torch.int32)
    signed = torch.where(signed >= 0x8000, signed - 0x10000, signed)
    return signed.to(torch.int16).contiguous().view(torch.bfloat16).reshape(shape)


def _validate_input(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dtype != torch.bfloat16 or tensor.dim() != 2:
        raise ValueError("Byte-v2 reference codec expects a 2D bfloat16 tensor")
    n, dim = map(int, tensor.shape)
    if n % BYTE_V2_TILE_SIZE or dim % BYTE_V2_TILE_SIZE:
        raise ValueError("Byte-v2 reference codec dimensions must be multiples of 16")
    return n, dim


def _best_window_base(exp: torch.Tensor) -> tuple[int, int]:
    hist = torch.bincount(exp.to(torch.int64), minlength=256)
    window = int(hist[:16].sum().item())
    best = window
    best_start = 0
    for start in range(1, 241):
        window += int(hist[start + 15].item()) - int(hist[start - 1].item())
        if window > best:
            best = window
            best_start = start
    return best_start, best


def compress_byte_v2_tensor(tensor: torch.Tensor) -> ByteV2TensorPayload:
    """Compress a 2D BF16 tensor in prototype Byte-v2 tile order."""
    n, dim = _validate_input(tensor)
    tensor = tensor.detach().cpu().contiguous()
    u16 = _bf16_to_u16(tensor).reshape(n, dim)

    n_tiles = n // BYTE_V2_TILE_SIZE
    dim_tiles = dim // BYTE_V2_TILE_SIZE
    total_tiles = n_tiles * dim_tiles

    base = torch.empty(total_tiles, dtype=torch.uint8)
    fallback = torch.empty(total_tiles, dtype=torch.uint8)
    low_bytes = torch.empty(total_tiles, BYTE_V2_TILE_ELEMS, dtype=torch.uint8)
    code_packed = torch.empty(
        total_tiles, BYTE_V2_PACKED_TILE_ELEMS, dtype=torch.uint8
    )
    fallback_raw_bits = torch.zeros(
        total_tiles, BYTE_V2_TILE_ELEMS, dtype=torch.int32
    )

    fallback_count = 0
    tile_id = 0
    for n_tile in range(n_tiles):
        n0 = n_tile * BYTE_V2_TILE_SIZE
        for dim_tile in range(dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            vals = u16[
                n0 : n0 + BYTE_V2_TILE_SIZE, d0 : d0 + BYTE_V2_TILE_SIZE
            ].reshape(-1)
            exp = (vals >> 7) & 0xFF
            base_i, covered = _best_window_base(exp)
            base[tile_id] = base_i

            if covered != BYTE_V2_TILE_ELEMS:
                fallback[tile_id] = 1
                fallback_count += 1
                fallback_raw_bits[tile_id] = vals
                low_bytes[tile_id].zero_()
                code_packed[tile_id].zero_()
            else:
                fallback[tile_id] = 0
                low = (vals & 0xFF).to(torch.uint8)
                delta = exp - base_i
                sign = (vals >> 15) & 1
                code = ((sign << 3) | ((delta >> 1) & 0x07)).to(torch.uint8)
                low_bytes[tile_id] = low
                code_packed[tile_id] = code[0::2] | (code[1::2] << 4)
            tile_id += 1

    logical_bytes = (
        total_tiles
        + ((total_tiles + 7) // 8)
        + (total_tiles - fallback_count)
        * (BYTE_V2_TILE_ELEMS + BYTE_V2_PACKED_TILE_ELEMS)
        + fallback_count * (BYTE_V2_TILE_ELEMS * 2)
    )
    fallback_raw = _u16_to_bf16(
        fallback_raw_bits.reshape(-1), (total_tiles, BYTE_V2_TILE_ELEMS)
    )

    return ByteV2TensorPayload(
        base=base,
        fallback=fallback,
        low_bytes=low_bytes,
        code_packed=code_packed,
        fallback_raw=fallback_raw,
        original_shape=(n, dim),
        fallback_tiles=fallback_count,
        logical_compressed_bytes=int(logical_bytes),
    )


def decompress_byte_v2_tensor(
    payload: ByteV2TensorPayload,
    shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Decompress a Byte-v2 payload produced by :func:`compress_byte_v2_tensor`."""
    if shape is None:
        shape = payload.original_shape
    n, dim = shape
    if n % BYTE_V2_TILE_SIZE or dim % BYTE_V2_TILE_SIZE:
        raise ValueError("Byte-v2 decompression shape must be a multiple of 16")

    n_tiles = n // BYTE_V2_TILE_SIZE
    dim_tiles = dim // BYTE_V2_TILE_SIZE
    if payload.total_tiles != n_tiles * dim_tiles:
        raise ValueError("Byte-v2 payload tile count does not match output shape")

    out_bits = torch.empty(n, dim, dtype=torch.int32)
    tile_id = 0
    for n_tile in range(n_tiles):
        n0 = n_tile * BYTE_V2_TILE_SIZE
        for dim_tile in range(dim_tiles):
            d0 = dim_tile * BYTE_V2_TILE_SIZE
            if int(payload.fallback[tile_id].item()):
                tile_bits = _bf16_to_u16(payload.fallback_raw[tile_id])
            else:
                base_i = int(payload.base[tile_id].item())
                low = payload.low_bytes[tile_id].to(torch.int32)
                packed = payload.code_packed[tile_id].to(torch.int32)
                code = torch.empty(BYTE_V2_TILE_ELEMS, dtype=torch.int32)
                code[0::2] = packed & 0x0F
                code[1::2] = (packed >> 4) & 0x0F

                low_exp_lsb = low >> 7
                delta_hi = code & 0x07
                exp_hi = (
                    (base_i >> 1)
                    + delta_hi
                    + ((base_i & 1) & (low_exp_lsb ^ 1))
                )
                high = ((code & 0x08) << 4) | exp_hi
                tile_bits = (high << 8) | low

            out_bits[
                n0 : n0 + BYTE_V2_TILE_SIZE,
                d0 : d0 + BYTE_V2_TILE_SIZE,
            ] = tile_bits.reshape(BYTE_V2_TILE_SIZE, BYTE_V2_TILE_SIZE)
            tile_id += 1

    return _u16_to_bf16(out_bits.reshape(-1), shape)

