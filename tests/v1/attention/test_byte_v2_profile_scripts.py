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
    "byte_v2_fa2_direct_paged_prefill_attention",
    "byte_v2_fa2_hybrid_paged_decode_attention",
    "byte_v2_fa2_raw_staging_prefill_attention",
    "byte_v2_reset_raw_fallback_pages",
    "byte_v2_test_force_promote_raw_staging_q1",
    "byte_v2_update_hybrid_cache_raw_tail_q1",
    "byte_v2_update_hybrid_cache_raw_staging_multi_token",
    "byte_v2_update_hybrid_cache_raw_staging_multi_token_retained",
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
        "block_size": 16,
        "capacity_tokens": 8192,
        "compact_tensor_bytes": 300,
        "raw_fallback_sidecar_bytes": 30,
        "raw_staging_workspace_bytes": 40,
        "total_planned_bytes": 370,
        "num_byte_v2_layers": 2,
        "raw_fallback_slots_per_layer": 3,
        "raw_staging_slots": 128,
    }


def test_speculative_profile_engine_limits_and_kv_demand():
    args = SimpleNamespace(
        context_lens=[4096],
        max_tokens=1024,
        engine_max_model_len=16384,
        max_num_batched_tokens=8192,
    )

    assert byte_v2_speculative_profile._resolve_engine_limits(args, 0) == (
        16384,
        8192,
    )
    demand = byte_v2_speculative_profile._workload_kv_demand(
        [[1] * 16, [2] * 32],
        [[3] * 17, [4]],
        {"block_size": 16, "num_blocks": 5},
    )
    assert demand == {
        "logical_prompt_tokens": 48,
        "logical_output_tokens": 18,
        "peak_resident_kv_tokens": 64,
        "peak_resident_kv_blocks": 4,
        "planned_kv_blocks": 5,
        "block_size": 16,
        "pressure_ratio": 0.8,
    }

    args.engine_max_model_len = 5000
    with pytest.raises(ValueError, match="smaller than the requested"):
        byte_v2_speculative_profile._resolve_engine_limits(args, 0)


def test_speculative_profile_scheduler_trace():
    preempt_calls = []

    def preempt(request, timestamp):
        preempt_calls.append((request.request_id, timestamp))

    scheduler = SimpleNamespace(
        _preempt_request=preempt,
        running=[
            SimpleNamespace(request_id="0", is_prefill_chunk=True),
            SimpleNamespace(request_id="1", is_prefill_chunk=False),
        ],
        waiting=[object(), object(), object()],
        skipped_waiting=[object()],
        kv_cache_manager=SimpleNamespace(usage=0.75),
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(
                engine_core=SimpleNamespace(scheduler=scheduler)
            )
        )
    )
    trace = byte_v2_speculative_profile.SchedulerTrace()
    trace.install(llm)
    trace.observe()
    request = SimpleNamespace(request_id="1", num_computed_tokens=123)
    scheduler._preempt_request(request, 4.0)
    trace.restore()

    assert preempt_calls == [("1", 4.0)]
    assert scheduler._preempt_request is preempt
    assert trace.result() == {
        "available": True,
        "peak_running_requests": 2,
        "peak_waiting_capacity_requests": 3,
        "peak_waiting_deferred_requests": 1,
        "peak_kv_cache_usage": 0.75,
        "steps_with_chunked_prefill": 1,
        "prefill_chunk_request_steps": 1,
        "preemption_count": 1,
        "discarded_kv_tokens": 123,
        "preempted_request_count": 1,
        "preempted_request_ids": ["1"],
    }


def test_speculative_profile_scheduler_counter_snapshot():
    llm = SimpleNamespace(
        get_metrics=lambda: [
            SimpleNamespace(name="vllm:num_preemptions", value=3),
            SimpleNamespace(name="vllm:prompt_tokens", value=100),
            SimpleNamespace(name="vllm:generation_tokens", value=20),
            SimpleNamespace(name="vllm:kv_cache_usage_perc", value=0.9),
            SimpleNamespace(name="vllm:spec_decode_num_drafts", value=4),
        ]
    )

    snapshot = byte_v2_speculative_profile._metric_snapshot(llm)

    assert snapshot == {
        "vllm:num_preemptions": 3,
        "vllm:prompt_tokens": 100,
        "vllm:generation_tokens": 20,
        "vllm:spec_decode_num_drafts": 4,
    }
    assert byte_v2_speculative_profile._scheduler_counter_result(snapshot) == {
        "generation_tokens": 20.0,
        "num_preemptions": 3.0,
        "prompt_tokens": 100.0,
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


def test_speculative_ignore_eos_cli_and_exact_length_validation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "byte_v2_speculative_profile.py",
            "--ignore-eos",
        ],
    )

    args = byte_v2_speculative_profile.parse_args()

    assert args.ignore_eos is True
    byte_v2_speculative_profile._validate_exact_decode_lengths(
        [[1, 2], [3, 4]],
        expected_tokens=2,
        label="test generation",
    )
    with pytest.raises(RuntimeError, match=r"test generation.*\[2, 1\]"):
        byte_v2_speculative_profile._validate_exact_decode_lengths(
            [[1, 2], [3]],
            expected_tokens=2,
            label="test generation",
        )


def test_speculative_memory_pressure_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "byte_v2_speculative_profile.py",
            "--engine-max-model-len",
            "16384",
            "--max-num-batched-tokens",
            "8192",
            "--kv-cache-memory-bytes",
            "20000000000",
            "--e2e-only",
        ],
    )

    args = byte_v2_speculative_profile.parse_args()

    assert args.engine_max_model_len == 16384
    assert args.max_num_batched_tokens == 8192
    assert args.kv_cache_memory_bytes == 20_000_000_000
    assert args.e2e_only is True


def test_speculative_prompt_hash_is_stable_and_order_sensitive():
    first = byte_v2_speculative_profile._prompt_token_ids_sha256([1, 2, 3])
    repeated = byte_v2_speculative_profile._prompt_token_ids_sha256([1, 2, 3])
    reordered = byte_v2_speculative_profile._prompt_token_ids_sha256([3, 2, 1])

    assert first == repeated
    assert first != reordered
    assert len(first) == 64


def test_speculative_summary_distinguishes_raw_fa2_token_match(capsys):
    def row(backend, spec_tokens, token_ids):
        return {
            "backend": backend,
            "spec_tokens": spec_tokens,
            "context_len": 1024,
            "batch_size": 2,
            "measured_seconds": 1.0,
            "output_tokens_per_second": 2.0,
            "performance_valid_for_tps": True,
            "spec_metrics": {
                "acceptance_rate": None,
                "mean_acceptance_length": 1.0,
            },
            "token_ids": token_ids,
        }

    results = [
        row("byte_v2", 0, [[1], [2]]),
        row("byte_v2", 4, [[1], [2]]),
        row("flash_attn", 0, [[1], [3]]),
        row("flash_attn", 4, [[1], [4]]),
    ]

    byte_v2_speculative_profile._annotate_reference_matches(results)
    byte_v2_speculative_profile._print_summary(results)

    assert results[0]["same_backend_nonspec_match"] is True
    assert results[0]["raw_fa2_reference_match"] is False
    assert results[0]["raw_fa2_mismatch_request_indices"] == [1]
    assert results[1]["same_backend_nonspec_match"] is True
    assert results[1]["raw_fa2_reference_match"] is False
    assert results[2]["same_backend_nonspec_match"] is True
    assert results[2]["raw_fa2_reference_match"] is True
    assert results[3]["same_backend_nonspec_match"] is False
    assert results[3]["raw_fa2_reference_match"] is False
    output = capsys.readouterr().out
    assert "same_backend_nonspec_match,raw_fa2_reference_match" in output
    assert "byte_v2,0,1024,2,1.000000,2.000,True,None,1.0,True,False,[1]" in output


def test_speculative_sample_logprobs_are_sorted_and_serializable():
    positions = [
        {
            9: SimpleNamespace(logprob=-0.25, rank=2),
            4: SimpleNamespace(logprob=-0.10, rank=1),
        },
        None,
    ]
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(logprobs=positions)])]

    result = byte_v2_speculative_profile._serialize_sample_logprobs(outputs)

    assert result == [
        [
            [
                {"token_id": 4, "logprob": -0.10, "rank": 1},
                {"token_id": 9, "logprob": -0.25, "rank": 2},
            ],
            [],
        ]
    ]
