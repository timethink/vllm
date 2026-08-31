# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def test_cudagraph_dispatch_can_exclude_full_graphs():
    manager = object.__new__(CudaGraphManager)
    manager._graphs_captured = True
    manager._candidates = [[] for _ in range(9)]
    full = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
    )
    piecewise = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=8,
        num_reqs=None,
    )
    manager._candidates[8] = [full, piecewise]

    assert manager.dispatch(8, 8, 1) is full
    assert manager.dispatch(8, 8, 1, max_mode=CUDAGraphMode.PIECEWISE) is piecewise
