# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import byte_v2_e2e_profile, byte_v2_speculative_profile

PROFILE_MODULES = (byte_v2_e2e_profile, byte_v2_speculative_profile)
HYBRID_PROFILE_OPS = {
    "byte_v2_hydrate_raw_staging_from_hybrid_cache",
    "byte_v2_commit_raw_staging_to_hybrid_cache",
    "byte_v2_fa2_hybrid_paged_decode_attention",
    "byte_v2_reset_raw_fallback_pages",
    "byte_v2_test_force_promote_raw_staging_q1",
}


def _fake_llm(static_forward_context, model_runner=None):
    compilation_config = SimpleNamespace(static_forward_context=static_forward_context)
    vllm_config = SimpleNamespace(compilation_config=compilation_config)
    engine = SimpleNamespace(vllm_config=vllm_config)
    if model_runner is not None:
        engine.model_executor = SimpleNamespace(
            driver_worker=SimpleNamespace(model_runner=model_runner)
        )
    return SimpleNamespace(llm_engine=engine)


def _fake_layer(store):
    return SimpleNamespace(impl=SimpleNamespace(raw_fallback_store=store))


def _fake_store(
    page_to_raw_slot,
    *,
    slots,
    free_count,
    fatal=0,
    forced_raw_diagnostic=None,
):
    state = SimpleNamespace(
        raw_pages=torch.empty((slots, 16), dtype=torch.uint8),
        page_to_raw_slot=torch.tensor(page_to_raw_slot, dtype=torch.int32),
        free_count=torch.tensor([free_count], dtype=torch.int32),
        fatal=torch.tensor([fatal], dtype=torch.int32),
    )
    store = SimpleNamespace(current_state=state)
    if forced_raw_diagnostic is not None:
        diagnostic = torch.tensor(forced_raw_diagnostic, dtype=torch.int32)
        state.forced_raw_diagnostic = diagnostic
        store.forced_raw_diagnostic = True
        store.clear_forced_raw_diagnostic = diagnostic.zero_
        store.arm_forced_raw_promotion = lambda: diagnostic[0].fill_(1)
    return store


@pytest.mark.parametrize("module", PROFILE_MODULES)
def test_byte_v2_profile_known_ops_include_hybrid_runtime_ops(module):
    assert set(module._BYTE_V2_PROFILE_OP_NAMES) >= HYBRID_PROFILE_OPS


@pytest.mark.parametrize("module", PROFILE_MODULES)
def test_collect_hybrid_state_reports_disabled_feature(module):
    llm = _fake_llm(
        {
            "model.layers.0.self_attn": _fake_layer(None),
            "unrelated": SimpleNamespace(impl=SimpleNamespace()),
        }
    )

    result = module._collect_hybrid_raw_fallback_state(llm)

    assert result == {
        "enabled": False,
        "layer_count": 1,
        "enabled_layer_count": 0,
        "initialized_layer_count": 0,
        "fully_initialized": False,
        "raw_page_count": None,
        "free_count": None,
        "slot_count": None,
        "fatal": None,
        "layers": [
            {
                "name": "model.layers.0.self_attn",
                "store_enabled": False,
                "initialized": False,
                "raw_page_count": None,
                "free_count": None,
                "slot_count": None,
                "fatal": None,
            }
        ],
    }


@pytest.mark.parametrize("module", PROFILE_MODULES)
def test_collect_hybrid_state_aggregates_initialized_layers(module):
    llm = _fake_llm(
        {
            "model.layers.1.self_attn": _fake_layer(
                _fake_store([-1, -1], slots=2, free_count=2, fatal=1)
            ),
            "model.layers.0.self_attn": _fake_layer(
                _fake_store([-1, 0, -1, 2], slots=3, free_count=1)
            ),
        }
    )

    result = module._collect_hybrid_raw_fallback_state(llm)

    assert result["enabled"] is True
    assert result["layer_count"] == 2
    assert result["enabled_layer_count"] == 2
    assert result["initialized_layer_count"] == 2
    assert result["fully_initialized"] is True
    assert result["raw_page_count"] == 2
    assert result["free_count"] == 3
    assert result["slot_count"] == 5
    assert result["fatal"] == 1
    assert [layer["name"] for layer in result["layers"]] == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    assert result["layers"][0]["raw_page_count"] == 2
    assert result["layers"][0]["free_count"] == 1
    assert result["layers"][0]["slot_count"] == 3
    assert result["layers"][1]["fatal"] == 1


@pytest.mark.parametrize("module", PROFILE_MODULES)
def test_collect_hybrid_state_does_not_initialize_store(module):
    store = SimpleNamespace(current_state=None)
    llm = _fake_llm({"model.layers.0.self_attn": _fake_layer(store)})

    result = module._collect_hybrid_raw_fallback_state(llm)

    assert result["enabled"] is True
    assert result["initialized_layer_count"] == 0
    assert result["fully_initialized"] is False
    assert result["raw_page_count"] is None
    assert store.current_state is None


def test_collect_speculative_profile_kv_cache_plan():
    config = SimpleNamespace(
        num_blocks=512,
        kv_cache_tensors=[SimpleNamespace(size=100), SimpleNamespace(size=200)],
        byte_v2_raw_fallback_sidecar_bytes=30,
        byte_v2_raw_staging_workspace_bytes=40,
        num_byte_v2_layers=2,
        byte_v2_raw_fallback_slots=3,
        byte_v2_raw_staging_slots=128,
    )
    runner = SimpleNamespace(kv_cache_config=config)
    driver_worker = SimpleNamespace(model_runner=runner)
    executor = SimpleNamespace(driver_worker=driver_worker)
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(model_executor=executor),
    )

    result = byte_v2_speculative_profile._collect_kv_cache_plan(llm)

    assert result == {
        "available": True,
        "num_blocks": 512,
        "compact_tensor_bytes": 300,
        "raw_fallback_sidecar_bytes": 30,
        "raw_staging_workspace_bytes": 40,
        "total_planned_bytes": 370,
        "num_byte_v2_layers": 2,
        "raw_fallback_slots_per_layer": 3,
        "raw_staging_slots": 128,
    }


def test_collect_speculative_forced_raw_lifecycle_evidence():
    llm = _fake_llm(
        {
            "model.layers.0.self_attn": _fake_layer(
                _fake_store(
                    [-1, -1],
                    slots=2,
                    free_count=2,
                    forced_raw_diagnostic=[0, 2, 5],
                )
            ),
            "model.layers.1.self_attn": _fake_layer(
                _fake_store(
                    [-1],
                    slots=1,
                    free_count=1,
                    forced_raw_diagnostic=[0, 3, 7],
                )
            ),
        }
    )

    result = byte_v2_speculative_profile._collect_hybrid_raw_fallback_state(llm)

    diagnostic = result["forced_raw_lifecycle"]
    assert diagnostic == {
        "enabled": True,
        "enabled_layer_count": 2,
        "initialized_layer_count": 2,
        "fully_initialized": True,
        "armed_layer_count": 0,
        "promotion_count": 5,
        "mapped_page_visit_count": 12,
        "all_latches_clear": True,
        "request_reset_observed": True,
    }


def test_speculative_forced_raw_control_and_phase_proof():
    stores = [
        _fake_store(
            [-1],
            slots=1,
            free_count=1,
            forced_raw_diagnostic=[9, 4, 6],
        )
        for _ in range(2)
    ]
    llm = _fake_llm(
        {
            f"model.layers.{index}.self_attn": _fake_layer(store)
            for index, store in enumerate(stores)
        }
    )

    cleared = byte_v2_speculative_profile._control_forced_raw_diagnostic(llm, arm=False)
    assert cleared["forced_raw_lifecycle"]["promotion_count"] == 0
    armed = byte_v2_speculative_profile._control_forced_raw_diagnostic(llm, arm=True)
    assert armed["forced_raw_lifecycle"]["armed_layer_count"] == 2

    for store in stores:
        store.current_state.forced_raw_diagnostic.copy_(
            torch.tensor([0, 1, 3], dtype=torch.int32)
        )
    completed = byte_v2_speculative_profile._collect_hybrid_raw_fallback_state(llm)
    proof = byte_v2_speculative_profile._forced_raw_phase_result(
        armed,
        completed,
    )

    assert proof["layer_count"] == 2
    assert proof["promotion_count"] == 2
    assert proof["mapped_page_visit_count"] == 6
    assert proof["request_reset_observed"] is True
    assert proof["verified"] is True


def test_speculative_forced_raw_reset_uses_runner_lifecycle_hook():
    store = _fake_store(
        [0, -1],
        slots=1,
        free_count=0,
        forced_raw_diagnostic=[0, 1, 2],
    )
    reset_calls = []

    def zero_block_ids(block_ids):
        reset_calls.append(block_ids)
        store.current_state.page_to_raw_slot.fill_(-1)
        store.current_state.free_count.fill_(1)

    runner = SimpleNamespace(_zero_block_ids=zero_block_ids)
    llm = _fake_llm(
        {"model.layers.0.self_attn": _fake_layer(store)},
        model_runner=runner,
    )

    result = byte_v2_speculative_profile._reset_forced_raw_pages_through_runner(llm)

    assert result == {
        "source": "runner_zero_block_ids_after_request",
        "physical_block_ids": [0],
    }
    assert reset_calls == [[0]]
    assert store.current_state.page_to_raw_slot.tolist() == [-1, -1]
    assert store.current_state.free_count.item() == 1


def test_speculative_forced_raw_cli_is_explicit_and_diagnostic_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "byte_v2_speculative_profile.py",
            "--backend",
            "byte_v2",
            "--spec-tokens",
            "0",
            "--batch-size",
            "1",
            "--max-tokens",
            "3",
            "--diagnose-forced-raw-lifecycle",
        ],
    )

    args = byte_v2_speculative_profile.parse_args()

    assert args.diagnose_forced_raw_lifecycle is True
    assert args.disable_prefix_caching is True
    assert args.collect_hybrid_state is True
