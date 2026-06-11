# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import CacheConfig
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_byte_v2_kv_cache,
    is_compressed_kv_cache,
    is_quantized_kv_cache,
)


def test_byte_v2_cache_dtype_config():
    cache_config = CacheConfig(cache_dtype="byte_v2")

    assert cache_config.cache_dtype == "byte_v2"
    assert cache_config.block_size == CacheConfig.DEFAULT_BLOCK_SIZE
    assert cache_config.enable_prefix_caching


def test_byte_v2_cache_dtype_keeps_prefix_caching_enabled():
    cache_config = CacheConfig(
        cache_dtype="byte_v2",
        enable_prefix_caching=True,
    )

    assert cache_config.enable_prefix_caching


def test_auto_cache_dtype_keeps_prefix_caching_enabled():
    cache_config = CacheConfig(cache_dtype="auto")

    assert cache_config.enable_prefix_caching


def test_byte_v2_cache_dtype_helpers():
    assert STR_DTYPE_TO_TORCH_DTYPE["byte_v2"] is torch.uint8
    assert is_byte_v2_kv_cache("byte_v2")
    assert is_compressed_kv_cache("byte_v2")
    assert not is_quantized_kv_cache("byte_v2")
