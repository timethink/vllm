# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backends import byte_v2_ops
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1


@pytest.fixture(params=[GPUModelRunnerV1, GPUModelRunnerV2], ids=["v1", "v2"])
def runner(request):
    runner = object.__new__(request.param)
    runner.device = torch.device("cpu")
    return runner


def _set_zeroer(runner, zeroer) -> None:
    if isinstance(runner, GPUModelRunnerV2):
        runner.kv_block_zeroer = zeroer
    else:
        runner._kv_block_zeroer = zeroer


def _init_reset_buffers(runner) -> None:
    runner._init_raw_fallback_reset(pin_memory=False)


def test_raw_fallback_reset_precedes_compact_zero_and_reuses_ids(runner) -> None:
    events = []
    reset_ids = []

    def record_reset(name, block_ids):
        events.append(name)
        reset_ids.append(block_ids)

    first_reset = Mock(side_effect=lambda ids: record_reset("reset-0", ids))
    second_reset = Mock(side_effect=lambda ids: record_reset("reset-1", ids))
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(
                impl=SimpleNamespace(reset_raw_fallback_pages=first_reset)
            ),
            "layer.1": SimpleNamespace(
                impl=SimpleNamespace(reset_raw_fallback_pages=second_reset)
            ),
        }
    )
    compact_zeroer = Mock()
    compact_zeroer.zero_block_ids.side_effect = lambda _: events.append("compact")
    _set_zeroer(runner, compact_zeroer)
    _init_reset_buffers(runner)

    block_ids = [3, 7]
    runner._zero_block_ids(block_ids)

    assert events == ["reset-0", "reset-1", "compact"]
    assert len(reset_ids) == 2
    assert reset_ids[0] is reset_ids[1]
    assert reset_ids[0].dtype == torch.int32
    assert reset_ids[0].device == runner.device
    assert reset_ids[0].tolist() == block_ids
    compact_zeroer.zero_block_ids.assert_called_once_with(block_ids)


def test_raw_fallback_reset_batches_compatible_layers(runner, monkeypatch) -> None:
    events = []
    first_state = tuple(torch.empty(1, dtype=torch.int32) for _ in range(4))
    second_state = tuple(torch.empty(1, dtype=torch.int32) for _ in range(4))
    first_reset = Mock()
    second_reset = Mock()
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(
                impl=SimpleNamespace(
                    reset_raw_fallback_pages=first_reset,
                    raw_fallback_reset_tensors=Mock(return_value=first_state),
                )
            ),
            "layer.1": SimpleNamespace(
                impl=SimpleNamespace(
                    reset_raw_fallback_pages=second_reset,
                    raw_fallback_reset_tensors=Mock(return_value=second_state),
                )
            ),
        }
    )
    batched_reset = Mock(side_effect=lambda *args: events.append("batched"))
    monkeypatch.setattr(
        byte_v2_ops,
        "byte_v2_batched_raw_fallback_reset_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        byte_v2_ops,
        "byte_v2_reset_raw_fallback_pages_batched",
        batched_reset,
    )
    compact_zeroer = Mock()
    compact_zeroer.zero_block_ids.side_effect = lambda _: events.append("compact")
    _set_zeroer(runner, compact_zeroer)
    _init_reset_buffers(runner)

    runner._zero_block_ids([5])

    assert events == ["batched", "compact"]
    first_reset.assert_not_called()
    second_reset.assert_not_called()
    page_maps, free_stacks, free_counts, fatals, reset_ids = (
        batched_reset.call_args.args
    )
    assert page_maps == [first_state[0], second_state[0]]
    assert free_stacks == [first_state[1], second_state[1]]
    assert free_counts == [first_state[2], second_state[2]]
    assert fatals == [first_state[3], second_state[3]]
    assert reset_ids.dtype == torch.int32
    assert reset_ids.tolist() == [5]


def test_missing_raw_fallback_reset_does_not_affect_compact_zero(
    runner, monkeypatch
) -> None:
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "no_impl": object(),
            "no_reset_method": SimpleNamespace(impl=object()),
        }
    )
    compact_zeroer = Mock()
    _set_zeroer(runner, compact_zeroer)
    tensor_factory = Mock(side_effect=AssertionError("unexpected buffer allocation"))
    monkeypatch.setattr(torch, "empty", tensor_factory)
    _init_reset_buffers(runner)

    block_ids = [11]
    runner._zero_block_ids(block_ids)

    tensor_factory.assert_not_called()
    compact_zeroer.zero_block_ids.assert_called_once_with(block_ids)


def test_noop_raw_fallback_reset_does_not_affect_compact_zero(runner) -> None:
    noop_reset = Mock()
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(
                impl=SimpleNamespace(reset_raw_fallback_pages=noop_reset)
            )
        }
    )
    compact_zeroer = Mock()
    _set_zeroer(runner, compact_zeroer)
    _init_reset_buffers(runner)

    block_ids = [13]
    runner._zero_block_ids(block_ids)

    noop_reset.assert_called_once()
    compact_zeroer.zero_block_ids.assert_called_once_with(block_ids)


def test_raw_fallback_id_buffers_are_reused_and_expand(runner) -> None:
    reset = Mock()
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(
                impl=SimpleNamespace(reset_raw_fallback_pages=reset)
            )
        }
    )
    compact_zeroer = Mock()
    _set_zeroer(runner, compact_zeroer)
    _init_reset_buffers(runner)

    runner._zero_block_ids([1])
    initial_pinned = runner._raw_fallback_ids_pinned
    initial_gpu = runner._raw_fallback_ids_gpu

    runner._zero_block_ids([2, 3])
    assert runner._raw_fallback_ids_pinned is initial_pinned
    assert runner._raw_fallback_ids_gpu is initial_gpu

    expanded_ids = list(range(8193))
    runner._zero_block_ids(expanded_ids)
    assert runner._raw_fallback_id_cap >= len(expanded_ids)
    assert runner._raw_fallback_ids_pinned is not initial_pinned
    assert runner._raw_fallback_ids_gpu is not initial_gpu
    last_reset_ids = reset.call_args.args[0]
    assert last_reset_ids.dtype == torch.int32
    assert last_reset_ids.tolist() == expanded_ids


def test_gpu_v2_decoded_prefix_cache_key_is_stable_until_signature_changes():
    runner = object.__new__(GPUModelRunnerV2)
    batch = SimpleNamespace(req_ids=["a", "b"])

    first = runner._cascade_prefix_cache_key(batch, [[4096]])
    second = runner._cascade_prefix_cache_key(batch, [[4096]])
    changed_prefix = runner._cascade_prefix_cache_key(batch, [[4080]])
    batch.req_ids.append("c")
    changed_membership = runner._cascade_prefix_cache_key(batch, [[4080]])

    assert (first, second, changed_prefix, changed_membership) == (1, 1, 2, 3)


def test_gpu_v2_invalidates_each_decoded_prefix_cache_once():
    runner = object.__new__(GPUModelRunnerV2)
    first = SimpleNamespace(invalidate_decoded_prefix_cache=Mock())
    second = SimpleNamespace(invalidate_decoded_prefix_cache=Mock())
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(impl=first),
            "layer.0.alias": SimpleNamespace(impl=first),
            "layer.1": SimpleNamespace(impl=second),
        }
    )
    runner._byte_v2_cascade_prefix_signature = (("old",), ((4096,),))

    runner._invalidate_decoded_prefix_caches()

    assert runner._byte_v2_cascade_prefix_signature is None
    first.invalidate_decoded_prefix_cache.assert_called_once_with()
    second.invalidate_decoded_prefix_cache.assert_called_once_with()


def test_gpu_v1_decoded_prefix_cache_key_is_stable_until_signature_changes():
    runner = object.__new__(GPUModelRunnerV1)
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"])

    first = runner._cascade_prefix_cache_key([[4096]])
    second = runner._cascade_prefix_cache_key([[4096]])
    changed_prefix = runner._cascade_prefix_cache_key([[4080]])
    runner.input_batch.req_ids.append("c")
    changed_membership = runner._cascade_prefix_cache_key([[4080]])

    assert (first, second, changed_prefix, changed_membership) == (1, 1, 2, 3)


def test_gpu_v1_invalidates_each_decoded_prefix_cache_once():
    runner = object.__new__(GPUModelRunnerV1)
    first = SimpleNamespace(invalidate_decoded_prefix_cache=Mock())
    second = SimpleNamespace(invalidate_decoded_prefix_cache=Mock())
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "layer.0": SimpleNamespace(impl=first),
            "layer.0.alias": SimpleNamespace(impl=first),
            "layer.1": SimpleNamespace(impl=second),
        }
    )
    runner._byte_v2_cascade_prefix_signature = (("old",), ((4096,),))

    runner._invalidate_decoded_prefix_caches()

    assert runner._byte_v2_cascade_prefix_signature is None
    first.invalidate_decoded_prefix_cache.assert_called_once_with()
    second.invalidate_decoded_prefix_cache.assert_called_once_with()
