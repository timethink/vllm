# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import torch

from vllm.v1.worker.gpu.warmup import warmup_kernels


def _make_runner(kv_block_zeroer):
    return SimpleNamespace(
        num_speculative_steps=2,
        kv_cache_config=SimpleNamespace(
            num_blocks=13,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
            ],
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=3,
            max_num_batched_tokens=12,
        ),
        is_pooling_model=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 128),
        is_last_pp_rank=True,
        kv_connector=Mock(),
        kv_block_zeroer=kv_block_zeroer,
    )


def test_warmup_zeros_each_synthetic_block_before_use(monkeypatch):
    """Warmup must honor the same fresh-block contract as the scheduler."""
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    runner = _make_runner(kv_block_zeroer=Mock())
    scheduler_outputs: list[Any] = []
    sample_outputs: list[Any] = []

    warmup_kernels(
        runner,
        scheduler_outputs.append,
        sample_outputs.append,
    )

    assert len(scheduler_outputs) == 3
    prefill_output, decode_output, cleanup_output = scheduler_outputs
    assert prefill_output.new_block_ids_to_zero == [1, 2, 3]
    assert decode_output.new_block_ids_to_zero == [4, 5, 6]
    assert cleanup_output.new_block_ids_to_zero is None

    prefill_request_blocks = [
        block_id
        for request in prefill_output.scheduled_new_reqs
        for group in request.block_ids
        for block_id in group
    ]
    decode_request_blocks = [
        block_id
        for request_groups in decode_output.scheduled_cached_reqs.new_block_ids
        if request_groups is not None
        for group in request_groups
        for block_id in group
    ]
    assert prefill_request_blocks == prefill_output.new_block_ids_to_zero
    assert decode_request_blocks == decode_output.new_block_ids_to_zero

    all_new_blocks = prefill_request_blocks + decode_request_blocks
    assert all_new_blocks == list(range(1, 7))
    assert len(all_new_blocks) == len(set(all_new_blocks))
    assert min(all_new_blocks) > 0
    assert max(all_new_blocks) < runner.kv_cache_config.num_blocks
    assert runner.kv_connector.set_disabled.call_args_list == [
        call(True),
        call(False),
    ]
    assert len(sample_outputs) == 2


def test_warmup_does_not_request_zeroing_without_zeroer(monkeypatch):
    """Raw KV caches must keep the pre-zeroing path disabled."""
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    runner = _make_runner(kv_block_zeroer=None)
    scheduler_outputs: list[Any] = []

    warmup_kernels(runner, scheduler_outputs.append, Mock())

    assert len(scheduler_outputs) == 3
    prefill_output, decode_output, cleanup_output = scheduler_outputs
    assert prefill_output.new_block_ids_to_zero is None
    assert decode_output.new_block_ids_to_zero is None
    assert cleanup_output.new_block_ids_to_zero is None
