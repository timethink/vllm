# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.byte_v2_layout import (
    BYTE_V2_PAGE_STATUS_COMPRESSED,
    BYTE_V2_PAGE_STATUS_OFFSET,
    ByteV2PageLayout,
    pack_byte_v2_kv_block_to_page,
)


def _reference_wmma_microbench(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(
        query.to(torch.float32), key.to(torch.float32).transpose(-1, -2)
    )
    probs = scores.to(torch.bfloat16).to(torch.float32)
    return torch.matmul(probs, value.to(torch.float32)).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_wmma_layout_microbench_matches_reference_cuda():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA microbench requires Ampere or newer")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(91)
    query = (
        0.125
        * torch.randn(
            2, 16, 128, device="cuda", dtype=torch.float32, generator=generator
        )
    ).to(torch.bfloat16)
    key = (
        0.125
        * torch.randn(
            2, 16, 128, device="cuda", dtype=torch.float32, generator=generator
        )
    ).to(torch.bfloat16)
    value = (
        0.125
        * torch.randn(
            2, 16, 128, device="cuda", dtype=torch.float32, generator=generator
        )
    ).to(torch.bfloat16)

    actual = ops.byte_v2_wmma_layout_microbench(
        query, key, value, variant=0, repeat_count=2
    )
    expected = _reference_wmma_microbench(query.cpu(), key.cpu(), value.cpu())

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_byte_v2_decode_page_wmma_microbench_matches_reference_cuda():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 WMMA microbench requires Ampere or newer")

    torch.manual_seed(92)
    layout = ByteV2PageLayout(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        raw_tail_bytes=0,
    )
    key = (1.0 + 0.01 * torch.randn(2, 16, 1, 128)).to(torch.bfloat16)
    value = (2.0 + 0.01 * torch.randn(2, 16, 1, 128)).to(torch.bfloat16)
    kv_cache = torch.zeros(2, layout.page_size_bytes, dtype=torch.uint8)
    for page_idx in range(2):
        pack_byte_v2_kv_block_to_page(
            key[page_idx], value[page_idx], layout, page=kv_cache[page_idx]
        )
        assert int(kv_cache[page_idx, BYTE_V2_PAGE_STATUS_OFFSET].item()) == (
            BYTE_V2_PAGE_STATUS_COMPRESSED
        )

    query = (0.125 * torch.randn(2, 16, 128)).to(torch.bfloat16)
    expected = _reference_wmma_microbench(
        query,
        key[:, :, 0, :],
        value[:, :, 0, :],
    )
    actual = ops.byte_v2_decode_page_wmma_microbench(
        query.cuda(),
        kv_cache.cuda(),
        num_kv_heads=1,
        kv_head=0,
        page_size_bytes=layout.page_size_bytes,
        repeat_count=2,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
