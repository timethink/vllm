# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def test_byte_v2_raw_tail_preprocess_error_is_request_scoped(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that an unsupported resumable request does not kill EngineCore."""

    if current_platform.is_rocm() or current_platform.is_xpu():
        pytest.skip(
            "Skipped on ROCm/XPU: this test only works with 'fork', "
            "but ROCm/XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Exercise the production guard only for the sentinel request while using
    # the existing multiprocess preprocessing error-response path.
    def trigger_raw_tail_guard(self, request: EngineCoreRequest):
        if request.prompt_token_ids and request.prompt_token_ids[0] == 333:
            request.resumable = True
            self.scheduler.byte_v2_raw_mutable_tail_q1 = True
            try:
                return original_preprocess(self, request)
            finally:
                self.scheduler.byte_v2_raw_mutable_tail_q1 = False
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore,
        "preprocess_add_request",
        trigger_raw_tail_guard,
    )

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # The unsupported request receives a request-scoped error.
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))  # type: ignore
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")
