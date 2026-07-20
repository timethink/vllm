# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.attention import (
    bind_byte_v2_raw_staging_workspace,
)
from vllm.v1.attention.backends.byte_v2_attn import (
    ByteV2RawFallbackStore,
    ByteV2RawStagingWorkspaceSpec,
    plan_byte_v2_raw_staging_waves,
)
from vllm.v1.attention.backends.byte_v2_layout import (
    DEFAULT_BYTE_V2_TILE_POLICY,
    ByteV2RawStagingLayout,
)

pytestmark = pytest.mark.cpu_test


def _metadata(query_start_locs, num_actual_tokens=None):
    if num_actual_tokens is None:
        num_actual_tokens = query_start_locs[-1]
    return SimpleNamespace(
        num_actual_tokens=num_actual_tokens,
        query_start_loc_cpu=torch.tensor(query_start_locs, dtype=torch.int32),
    )


def test_byte_v2_raw_staging_workspace_exact_size_and_initial_state():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=512,
        num_staging_slots=128,
        slot_size_bytes=65_536,
        device=torch.device("cpu"),
    )

    workspace = spec.allocate()

    assert spec.nbytes == 65_536 * 128 + 4 * 512 + 8 * 128 + 8
    assert workspace.raw_staging.shape == (128, 65_536)
    assert workspace.block_to_staging_slot.tolist() == [-1] * 512
    assert workspace.staging_to_physical_block.tolist() == [-1] * 128
    assert not workspace.valid_rows.any()
    assert workspace.next_staging_slot.item() == 0
    assert workspace.overflow.item() == 0


def test_byte_v2_raw_fallback_uses_plan_and_rejects_cache_rebind():
    store = ByteV2RawFallbackStore(
        raw_layout=ByteV2RawStagingLayout(
            tile_policy=DEFAULT_BYTE_V2_TILE_POLICY,
            num_kv_heads=8,
        )
    )
    store.bind_plan(num_blocks=2, num_raw_slots=3)
    first_cache = torch.empty((2, 1), dtype=torch.uint8)
    second_cache = torch.empty((2, 1), dtype=torch.uint8)

    state = store.state(first_cache)

    assert state.raw_pages.shape[0] == 3
    with pytest.raises(RuntimeError, match="binding changed"):
        store.state(second_cache)


def test_byte_v2_page_aware_wave_packs_one_long_request():
    waves = plan_byte_v2_raw_staging_waves(
        1024,
        128,
        _metadata([0, 1024]),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 1024, 65)
    ]


def test_byte_v2_page_aware_wave_splits_16k_into_nine_waves():
    waves = plan_byte_v2_raw_staging_waves(
        16_384,
        128,
        _metadata([0, 16_384]),
    )

    assert len(waves) == 9
    assert waves[0].start == 0
    assert waves[-1].end == 16_384
    assert all(wave.max_unique_pages <= 128 for wave in waves)
    assert all(left.end == right.start for left, right in zip(waves, waves[1:]))


def test_byte_v2_page_aware_wave_bounds_many_one_token_requests():
    query_start_locs = list(range(130))

    waves = plan_byte_v2_raw_staging_waves(
        129,
        128,
        _metadata(query_start_locs),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 128, 128),
        (128, 129, 1),
    ]


def test_byte_v2_untrusted_metadata_uses_one_page_per_token_bound():
    waves = plan_byte_v2_raw_staging_waves(
        300,
        128,
        _metadata([1, 300]),
    )

    assert [(wave.start, wave.end, wave.max_unique_pages) for wave in waves] == [
        (0, 128, 128),
        (128, 256, 128),
        (256, 300, 44),
    ]


class _WorkspaceImpl:
    def __init__(self, spec):
        self.spec = spec
        self.workspace = None
        self.raw_plan = None
        self.raw_state_cache = None

    def raw_staging_workspace_spec(self, kv_cache, num_staging_slots):
        assert kv_cache.device == self.spec.device
        assert num_staging_slots == self.spec.num_staging_slots
        return self.spec

    def bind_raw_staging_workspace(self, workspace):
        self.workspace = workspace

    def bind_raw_fallback_plan(self, *, num_blocks, num_raw_slots):
        self.raw_plan = (num_blocks, num_raw_slots)

    def initialize_raw_fallback_state(self, kv_cache):
        self.raw_state_cache = kv_cache


def test_bind_byte_v2_raw_staging_workspace_shares_one_allocation():
    spec = ByteV2RawStagingWorkspaceSpec(
        num_blocks=4,
        num_staging_slots=2,
        slot_size_bytes=64,
        device=torch.device("cpu"),
    )
    impls = [_WorkspaceImpl(spec), _WorkspaceImpl(spec)]
    context = {
        str(index): SimpleNamespace(
            impl=impl,
            kv_cache=torch.empty((4, 1), dtype=torch.uint8),
        )
        for index, impl in enumerate(impls)
    }
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=2,
        byte_v2_raw_staging_workspace_bytes=spec.nbytes,
    )

    workspace = bind_byte_v2_raw_staging_workspace(
        context,
        config,
        use_ubatching=False,
    )

    assert workspace is not None
    assert impls[0].workspace is workspace
    assert impls[1].workspace is workspace
    assert impls[0].raw_plan == (4, 1)
    assert impls[1].raw_plan == (4, 1)
    assert impls[0].raw_state_cache is context["0"].kv_cache
    assert impls[1].raw_state_cache is context["1"].kv_cache


def test_bind_byte_v2_raw_staging_workspace_rejects_ubatching():
    config = SimpleNamespace(
        num_blocks=4,
        byte_v2_raw_fallback_slots=1,
        byte_v2_raw_staging_slots=128,
        byte_v2_raw_staging_workspace_bytes=1,
    )

    with pytest.raises(RuntimeError, match="single-lane"):
        bind_byte_v2_raw_staging_workspace(
            {},
            config,
            use_ubatching=True,
        )
