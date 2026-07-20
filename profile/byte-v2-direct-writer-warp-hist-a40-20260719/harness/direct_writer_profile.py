# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the ByteV2 V5 direct cache writer on a 4K prompt."""

from __future__ import annotations

import argparse

import torch

from vllm.v1.attention.backends.byte_v2_layout import ByteV2PageLayoutV5
from vllm.v1.attention.backends.byte_v2_ops import byte_v2_reshape_and_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.tokens % 16:
        raise ValueError("--tokens must be a positive multiple of 16")

    torch.manual_seed(20260719)
    layout = ByteV2PageLayoutV5()
    key = torch.randn((args.tokens, 8, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.arange(args.tokens, dtype=torch.int64, device="cuda")
    kv_cache = torch.zeros(
        (args.tokens // 16, layout.page_size_bytes),
        dtype=torch.uint8,
        device="cuda",
    )

    def launch() -> None:
        byte_v2_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            codec_token_block=16,
            codec_dim_block=16,
            alloc_block_tokens=16,
        )

    for _ in range(args.warmup):
        launch()
    torch.accelerator.synchronize()
    launch()
    torch.accelerator.synchronize()


if __name__ == "__main__":
    main()
