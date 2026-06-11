# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.byte_v2_attn import ByteV2MetadataBuilder
from vllm.v1.kv_cache_interface import ByteV2FullAttentionSpec


def _make_builder() -> ByteV2MetadataBuilder:
    spec = ByteV2FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        head_size_v=16,
        dtype=torch.uint8,
    )
    return ByteV2MetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["dummy"],
        vllm_config=SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        device=torch.device("cpu"),
    )


def test_byte_v2_metadata_builder_all_decode():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[17, 33], query_lens=[1, 1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    builder = _make_builder()

    metadata = builder.build(0, common)

    assert metadata.seq_lens is common.seq_lens
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.block_table is common.block_table_tensor
    assert metadata.query_start_loc is common.query_start_loc
    assert metadata.query_start_loc_cpu is common.query_start_loc_cpu
    assert metadata.seq_lens_cpu is common.seq_lens_cpu_upper_bound
    assert metadata.num_actual_tokens == 2
    assert metadata.max_query_len == 1
    assert metadata.max_seq_len == 33
    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 2
    assert metadata.num_prefills == 0
    assert metadata.num_prefill_tokens == 0
    assert metadata.block_size == 16
    assert metadata.page_size_bytes == builder.page_size_bytes
    assert metadata.causal


def test_byte_v2_metadata_builder_mixed_decode_prefill():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[17, 33, 8], query_lens=[1, 1, 8]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )

    metadata = _make_builder().build(0, common)

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 2
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 8
    assert metadata.max_query_len == 8


def test_byte_v2_metadata_builder_ignores_cascade_prefix_len():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[17], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )

    builder = _make_builder()
    metadata = builder.build(16, common)

    assert metadata.block_table is common.block_table_tensor
    assert metadata.seq_lens is common.seq_lens
    assert not builder.use_cascade_attention(
        common_prefix_len=16,
        query_lens=torch.tensor([1]).numpy(),
        num_query_heads=2,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=1,
        dcp_world_size=1,
    )


def test_byte_v2_metadata_update_block_table():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[17], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    builder = _make_builder()
    metadata = builder.build(0, common)
    new_block_table = torch.full_like(common.block_table_tensor, 7)
    new_slot_mapping = torch.full_like(common.slot_mapping, 11)

    updated = builder.update_block_table(
        metadata,
        blk_table=new_block_table,
        slot_mapping=new_slot_mapping,
    )

    assert updated is not metadata
    assert updated.block_table is new_block_table
    assert updated.slot_mapping is new_slot_mapping
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    assert updated.seq_lens is metadata.seq_lens


def test_byte_v2_metadata_cudagraph_capture_sets_small_seq_lens():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[17, 33], query_lens=[1, 1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    builder = _make_builder()

    metadata = builder.build_for_cudagraph_capture(common)

    assert torch.equal(metadata.seq_lens, torch.ones_like(common.seq_lens))
    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 2
    assert builder.get_cudagraph_support(
        builder.vllm_config, builder.kv_cache_spec
    ) == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
