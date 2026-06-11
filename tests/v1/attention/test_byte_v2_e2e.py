# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("BYTE_V2_RUN_E2E") != "1",
    reason="Set BYTE_V2_RUN_E2E=1 to run the Byte-v2 e2e smoke test",
)


def _get_model_path() -> Path:
    model = Path(
        os.environ.get(
            "BYTE_V2_E2E_MODEL",
            "/mnt/sda1/yxz/new_idea/models/pythia-14m",
        )
    )
    if not model.exists():
        pytest.skip(f"Byte-v2 e2e model path does not exist: {model}")
    return model


def _generate_token_ids(kv_cache_dtype: str) -> tuple[list[int], list[int]]:
    from vllm import LLM, SamplingParams

    model = _get_model_path()
    prompt = " ".join(["hello"] * 24)
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.2,
        trust_remote_code=False,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(
            max_tokens=2,
            min_tokens=2,
            ignore_eos=True,
            temperature=0.0,
        ),
    )
    prompt_token_ids = outputs[0].prompt_token_ids
    output_token_ids = outputs[0].outputs[0].token_ids
    del llm
    torch.cuda.synchronize()
    return prompt_token_ids, output_token_ids


def test_byte_v2_llm_generate_e2e_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    prompt_token_ids, output_token_ids = _generate_token_ids("byte_v2")

    assert len(prompt_token_ids) >= 16
    assert len(output_token_ids) == 2


def test_byte_v2_matches_raw_vllm_e2e_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    raw_prompt_ids, raw_output_ids = _generate_token_ids("auto")
    byte_prompt_ids, byte_output_ids = _generate_token_ids("byte_v2")

    assert byte_prompt_ids == raw_prompt_ids
    assert byte_output_ids == raw_output_ids


def test_byte_v2_prefix_cache_hit_e2e_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from vllm import LLM, SamplingParams

    model = _get_model_path()
    prompt = {"prompt_token_ids": [101] * 33}
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        kv_cache_dtype="byte_v2",
        block_size=16,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=float(
            os.environ.get("BYTE_V2_E2E_GPU_MEMORY_UTILIZATION", "0.2")
        ),
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        temperature=0.0,
    )

    first = llm.generate([prompt], sampling, use_tqdm=False)
    second = llm.generate([prompt], sampling, use_tqdm=False)

    assert first[0].num_cached_tokens == 0
    assert second[0].num_cached_tokens is not None
    assert second[0].num_cached_tokens >= 32

    del llm
    torch.cuda.synchronize()
