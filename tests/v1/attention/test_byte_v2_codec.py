# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.byte_v2_codec import (
    BYTE_V2_PACKED_TILE_ELEMS,
    BYTE_V2_TILE_ELEMS,
    compress_byte_v2_tensor,
    decompress_byte_v2_tensor,
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


def test_byte_v2_codec_fast_path_round_trip():
    torch.manual_seed(0)
    data = (1.0 + 0.01 * torch.randn(32, 32)).to(torch.bfloat16)
    data[1::2] = -data[1::2]

    payload = compress_byte_v2_tensor(data)
    decompressed = decompress_byte_v2_tensor(payload)

    _assert_bf16_bits_equal(decompressed, data)
    assert payload.fallback_tiles == 0
    assert payload.base.shape == (4,)
    assert payload.fallback.shape == (4,)
    assert payload.low_bytes.shape == (4, BYTE_V2_TILE_ELEMS)
    assert payload.code_packed.shape == (4, BYTE_V2_PACKED_TILE_ELEMS)
    assert payload.logical_compressed_bytes == 4 + 1 + 4 * (256 + 128)


def test_byte_v2_codec_fallback_round_trip():
    exp = (torch.arange(BYTE_V2_TILE_ELEMS, dtype=torch.int32) % 32) + 40
    low = torch.arange(BYTE_V2_TILE_ELEMS, dtype=torch.int32) & 0x7F
    bits = (exp << 7) | low
    data = _bf16_from_u16(bits, (16, 16))

    payload = compress_byte_v2_tensor(data)
    decompressed = decompress_byte_v2_tensor(payload)

    _assert_bf16_bits_equal(decompressed, data)
    assert payload.fallback_tiles == 1
    assert int(payload.fallback[0].item()) == 1
    assert torch.count_nonzero(payload.low_bytes).item() == 0
    assert torch.count_nonzero(payload.code_packed).item() == 0


def test_byte_v2_codec_validates_input():
    with pytest.raises(ValueError, match="2D bfloat16"):
        compress_byte_v2_tensor(torch.zeros(16, 16, dtype=torch.float32))

    with pytest.raises(ValueError, match="multiples of 16"):
        compress_byte_v2_tensor(torch.zeros(16, 15, dtype=torch.bfloat16))


def test_byte_v2_codec_validates_decompression_shape():
    payload = compress_byte_v2_tensor(torch.ones(16, 16, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="multiple of 16"):
        decompress_byte_v2_tensor(payload, shape=(16, 15))

    with pytest.raises(ValueError, match="tile count"):
        decompress_byte_v2_tensor(payload, shape=(32, 16))

