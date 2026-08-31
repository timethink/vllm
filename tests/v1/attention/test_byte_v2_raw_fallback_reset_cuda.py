# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.byte_v2_ops import (
    byte_v2_batched_raw_fallback_reset_is_available,
    byte_v2_reset_raw_fallback_pages,
    byte_v2_reset_raw_fallback_pages_batched,
)


def _reset_state(device: torch.device):
    num_blocks = 64
    num_raw_slots = 16
    page_to_raw_slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
    page_to_raw_slot[torch.tensor([3, 7, 11], device=device)] = torch.tensor(
        [15, 14, 13], dtype=torch.int32, device=device
    )
    return SimpleNamespace(
        page_to_raw_slot=page_to_raw_slot,
        free_slots=torch.arange(num_raw_slots, dtype=torch.int32, device=device),
        free_count=torch.tensor([13], dtype=torch.int32, device=device),
        fatal=torch.zeros(1, dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
def test_batched_raw_fallback_reset_matches_per_layer_and_preserves_cached_page():
    if not byte_v2_batched_raw_fallback_reset_is_available():
        pytest.skip("Batched raw-fallback reset schema is not registered")

    device = torch.device("cuda")
    reference_states = [_reset_state(device) for _ in range(32)]
    batched_states = [_reset_state(device) for _ in range(32)]
    # Duplicate 7 exercises the atomicExch idempotency contract. Page 11 is
    # intentionally omitted: it represents a still-live prefix-cached page.
    physical_block_ids = torch.tensor([3, 7, 7], dtype=torch.int32, device=device)

    for state in reference_states:
        byte_v2_reset_raw_fallback_pages(
            state.page_to_raw_slot,
            state.free_slots,
            state.free_count,
            state.fatal,
            physical_block_ids,
        )
    byte_v2_reset_raw_fallback_pages_batched(
        [state.page_to_raw_slot for state in batched_states],
        [state.free_slots for state in batched_states],
        [state.free_count for state in batched_states],
        [state.fatal for state in batched_states],
        physical_block_ids,
    )
    torch.accelerator.synchronize()

    for reference, batched in zip(reference_states, batched_states):
        assert torch.equal(batched.page_to_raw_slot, reference.page_to_raw_slot)
        assert batched.page_to_raw_slot[3].item() == -1
        assert batched.page_to_raw_slot[7].item() == -1
        assert batched.page_to_raw_slot[11].item() == 13
        assert batched.free_count.item() == reference.free_count.item() == 15
        assert batched.fatal.item() == reference.fatal.item() == 0
        active_count = batched.free_count.item()
        assert torch.equal(
            batched.free_slots[:active_count].sort().values,
            reference.free_slots[:active_count].sort().values,
        )
