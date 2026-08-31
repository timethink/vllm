# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    ByteV2FullAttentionSpec,
    FullAttentionSpec,
)
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _BlockFirstBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs):
        return 0


def _zeroer(spec, kv_cache):
    group = AttentionGroup(
        backend=_BlockFirstBackend,
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    return KVBlockZeroer(
        device=kv_cache.device,
        pin_memory=False,
        attn_groups_iter=[group],
        kernel_block_sizes=[spec.block_size],
        cache_dtype="auto",
        static_forward_context={
            "layer": SimpleNamespace(kv_cache=kv_cache),
        },
    )


def test_byte_v2_block_zeroer_preserves_poisoned_payload():
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
    )
    kv_cache = torch.full(
        (3, spec.page_size_bytes),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    zeroer = _zeroer(spec, kv_cache)
    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    metadata_bytes = spec.page_metadata_size_bytes
    assert torch.count_nonzero(kv_cache[1, :metadata_bytes]) == 0
    assert torch.all(kv_cache[1, metadata_bytes:] == 0xA5)
    assert torch.all(kv_cache[0] == 0xA5)
    assert torch.all(kv_cache[2] == 0xA5)


def test_static_w16_block_zeroer_initializes_canonical_empty_header(monkeypatch):
    monkeypatch.setenv("BYTE_V2_STATIC_W16_CODEBOOK", "/unused/codebook.json")
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
    )
    kv_cache = torch.full(
        (3, spec.page_size_bytes),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    zeroer = _zeroer(spec, kv_cache)
    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    offset = spec.page_metadata_offset_bytes
    size = spec.page_metadata_size_bytes
    assert offset == 49_152
    assert size == 128
    assert torch.all(kv_cache[1, :offset] == 0xA5)
    header = kv_cache[1, offset : offset + size].view(torch.int32)
    assert torch.count_nonzero(header[:16]) == 0
    assert header[16] == 2
    assert torch.count_nonzero(header[17:]) == 0
    assert torch.all(kv_cache[1, offset + size :] == 0xA5)
    dummy_header = kv_cache[0, offset : offset + size].view(torch.int32)
    assert torch.all(kv_cache[0, :offset] == 0xA5)
    assert torch.count_nonzero(dummy_header[:16]) == 0
    assert dummy_header[16] == 2
    assert torch.count_nonzero(dummy_header[17:]) == 0
    assert torch.all(kv_cache[0, offset + size :] == 0xA5)
    assert torch.all(kv_cache[2] == 0xA5)


def test_full_attention_block_zeroer_still_clears_complete_page():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
    )
    kv_cache = torch.full(
        (3, 16, 1, 8),
        3.0,
        dtype=torch.float32,
        device="cuda",
    )

    _zeroer(spec, kv_cache).zero_block_ids([1])
    torch.accelerator.synchronize()

    assert torch.count_nonzero(kv_cache[1]) == 0
    assert torch.all(kv_cache[0] == 3.0)
    assert torch.all(kv_cache[2] == 3.0)
