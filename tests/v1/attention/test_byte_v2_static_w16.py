# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import byte_v2_attn as byte_v2_attn_module
from vllm.v1.attention.backends import byte_v2_ops
from vllm.v1.attention.backends.byte_v2_attn import ByteV2AttentionImpl
from vllm.v1.attention.backends.byte_v2_static_w16 import (
    ByteV2StaticW16Codebook,
    byte_v2_static_w16_codebook_from_env,
    byte_v2_static_w16_retain_cascade_q16_requested,
    load_byte_v2_static_w16_codebook,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    wire = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256(wire)


def _write_codebook(root: Path) -> tuple[Path, Path, dict]:
    model = root / "model"
    model.mkdir()
    config = b'{"num_hidden_layers":2}\n'
    index = b'{"metadata":{},"weight_map":{}}\n'
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index)
    bases = [[115, 110], [116, 112]]
    payload = {
        "schema_version": 1,
        "kind": "bytev2_static_w16_e2e_prototype",
        "model": str(model),
        "model_config_sha256": _sha256(config),
        "model_index_sha256": _sha256(index),
        "bases": bases,
        "bases_sha256": _canonical_sha256(bases),
    }
    path = root / "codebook.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, model, payload


def test_load_byte_v2_static_w16_codebook(tmp_path: Path):
    path, model, payload = _write_codebook(tmp_path)
    codebook = load_byte_v2_static_w16_codebook(str(path), str(model), 2)

    assert codebook.bases == ((115, 110), (116, 112))
    assert codebook.bases_sha256 == payload["bases_sha256"]
    assert codebook.file_sha256 == _sha256(path.read_bytes())
    assert codebook.bases_for_layer(1) == (116, 112)
    with pytest.raises(RuntimeError, match="outside the frozen table"):
        codebook.bases_for_layer(2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(kind="wrong"), "codebook kind"),
        (lambda payload: payload.update(bases_sha256="0" * 64), "bases_sha256"),
        (lambda payload: payload["bases"].pop(), "exactly one"),
        (lambda payload: payload["bases"][1].__setitem__(1, 241), r"\[0, 240\]"),
    ],
)
def test_static_w16_codebook_rejects_invalid_bases(
    tmp_path: Path,
    mutation,
    message: str,
):
    path, model, payload = _write_codebook(tmp_path)
    mutation(payload)
    if message != "bases_sha256":
        payload["bases_sha256"] = _canonical_sha256(payload["bases"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_byte_v2_static_w16_codebook(str(path), str(model), 2)


def test_static_w16_codebook_env_is_explicit(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BYTE_V2_STATIC_W16_CODEBOOK", raising=False)
    assert (
        byte_v2_static_w16_codebook_from_env(
            expected_model=str(tmp_path),
            expected_num_layers=2,
        )
        is None
    )

    monkeypatch.setenv("BYTE_V2_STATIC_W16_CODEBOOK", "")
    with pytest.raises(RuntimeError, match="must not be empty"):
        byte_v2_static_w16_codebook_from_env(
            expected_model=str(tmp_path),
            expected_num_layers=2,
        )


def test_static_w16_cascade_q16_retention_env_is_explicit(monkeypatch):
    env_name = "BYTE_V2_STATIC_W16_RETAIN_CASCADE_Q16"
    monkeypatch.delenv(env_name, raising=False)
    assert not byte_v2_static_w16_retain_cascade_q16_requested()
    monkeypatch.setenv(env_name, "1")
    assert byte_v2_static_w16_retain_cascade_q16_requested()
    monkeypatch.setenv(env_name, "0")
    assert not byte_v2_static_w16_retain_cascade_q16_requested()
    monkeypatch.setenv(env_name, "true")
    with pytest.raises(ValueError, match=env_name):
        byte_v2_static_w16_retain_cascade_q16_requested()


def test_static_w16_cascade_q16_retention_proof_fails_closed():
    num_requests = 4
    common_prefix_len = 4_080
    metadata = SimpleNamespace(
        num_actual_tokens=num_requests * 16,
        max_query_len=16,
        query_start_loc_cpu=torch.arange(
            0,
            (num_requests + 1) * 16,
            16,
            dtype=torch.int32,
        ),
        seq_lens_cpu_upper_bound=torch.full(
            (num_requests,),
            common_prefix_len + 16,
            dtype=torch.int32,
        ),
        is_prefilling=torch.ones(num_requests, dtype=torch.bool),
        use_cascade=True,
        common_prefix_len=common_prefix_len,
    )

    assert byte_v2_attn_module._trusted_static_w16_cascade_q16_pages(
        metadata,
        num_requests * 16,
        alloc_block_tokens=16,
    )
    metadata.seq_lens_cpu_upper_bound[0] += 16
    assert not byte_v2_attn_module._trusted_static_w16_cascade_q16_pages(
        metadata,
        num_requests * 16,
        alloc_block_tokens=16,
    )
    metadata.seq_lens_cpu_upper_bound[0] -= 16
    metadata.is_prefilling[0] = False
    assert not byte_v2_attn_module._trusted_static_w16_cascade_q16_pages(
        metadata,
        num_requests * 16,
        alloc_block_tokens=16,
    )


def test_static_w16_layer_name_configures_writer():
    codebook = ByteV2StaticW16Codebook(
        path="/codebook.json",
        file_sha256="f" * 64,
        bases_sha256="b" * 64,
        model="/model",
        model_config_sha256="c" * 64,
        model_index_sha256="i" * 64,
        bases=((115, 110), (116, 112)),
    )
    configured: list[tuple[int, int]] = []
    impl = ByteV2AttentionImpl.__new__(ByteV2AttentionImpl)
    impl.static_w16_codebook = codebook
    impl.static_w16_layer_index = None
    impl.raw_staging_manager = SimpleNamespace(
        configure_static_w16_bases=configured.append
    )

    impl._configure_static_w16_writer(
        SimpleNamespace(layer_name="model.layers.1.self_attn")
    )

    assert impl.static_w16_layer_index == 1
    assert configured == [(116, 112)]
    with pytest.raises(RuntimeError, match="changed layer identity"):
        impl._configure_static_w16_writer(
            SimpleNamespace(layer_name="model.layers.0.self_attn")
        )


def test_static_w16_accepts_prefix_caching(monkeypatch):
    codebook = ByteV2StaticW16Codebook(
        path="/codebook.json",
        file_sha256="f" * 64,
        bases_sha256="b" * 64,
        model="/model",
        model_config_sha256="c" * 64,
        model_index_sha256="i" * 64,
        bases=((115, 110), (116, 112)),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/model",
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
    )
    monkeypatch.setenv("BYTE_V2_STATIC_W16_CODEBOOK", codebook.path)
    monkeypatch.setenv("BYTE_V2_FA2_HYBRID_RAW_FALLBACK", "1")
    monkeypatch.delenv("BYTE_V2_STATIC_W8_CODEBOOK", raising=False)
    monkeypatch.delenv("BYTE_V2_TEST_FORCE_RAW_PROMOTION", raising=False)
    monkeypatch.setattr(
        byte_v2_attn_module,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_static_w16_codebook_from_env",
        lambda **_: codebook,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_static_w16_runtime_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        byte_v2_attn_module,
        "byte_v2_hybrid_cache_update_is_available",
        lambda: True,
    )

    impl = ByteV2AttentionImpl(
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
    )

    assert impl.static_w16_codebook is codebook
    assert impl.decode_fa2_available


def test_static_w16_runtime_requires_canonical_fa2_reader(monkeypatch):
    writer_ops = {
        "byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache",
        "byte_v2_static_w16_commit_raw_staging_to_hybrid_cache",
        "byte_v2_static_w16_update_hybrid_cache_raw_tail_q1",
    }
    monkeypatch.setattr(
        byte_v2_ops,
        "_find_op",
        lambda namespace, name: (
            object() if namespace == "_C_cache_ops" and name in writer_ops else None
        ),
    )
    monkeypatch.setattr(
        byte_v2_ops,
        "_find_fa2_op",
        lambda name: object() if name == "static_w16_varlen_fwd" else None,
    )
    assert not byte_v2_ops.byte_v2_static_w16_runtime_is_available()

    monkeypatch.setattr(
        byte_v2_ops,
        "_find_fa2_op",
        lambda name: object() if name == "static_w16_canonical_varlen_fwd" else None,
    )
    assert byte_v2_ops.byte_v2_static_w16_runtime_is_available()


def test_static_w16_fa2_direct_fragment_dispatch_contract():
    repo_root = Path(__file__).resolve().parents[3]
    loader_source = (
        repo_root / "csrc/libtorch_stable/byte_v2/byte_v2_fa2_loader.cuh"
    ).read_text(encoding="utf-8")
    fa2_patch_source = (
        repo_root / "cmake/patches/vllm_flash_attn_byte_v2.patch"
    ).read_text(encoding="utf-8")

    assert "static_assert(DirectFragmentK);" not in loader_source
    assert "DirectFragmentK || DependentFalse<Params>::value" in loader_source
    assert "static constexpr bool Enabled = Loader::DirectFragmentK;" in (
        fa2_patch_source
    )
    assert (
        "ExternalKvLoaderDirectFragmentKConfig<ExternalKvLoader>::Enabled"
        in fa2_patch_source
    )

    direct_call = "ExternalKvLoader::direct_fragment_k_gemm("
    call_offsets = [
        offset
        for offset in range(len(fa2_patch_source))
        if fa2_patch_source.startswith(direct_call, offset)
    ]
    assert len(call_offsets) == 2
    for call_offset in call_offsets:
        branch_offset = fa2_patch_source.rfind(
            "if constexpr (Direct_fragment_k) {", 0, call_offset
        )
        fallback_offset = fa2_patch_source.find(
            "} else if (active_query_warp) {", call_offset
        )
        assert 0 < call_offset - branch_offset < 200
        assert 0 < fallback_offset - call_offset < 600
        assert (
            "FLASH_NAMESPACE::gemm("
            in fa2_patch_source[fallback_offset : fallback_offset + 500]
        )


def test_static_w16_attention_routes_to_canonical_reader(monkeypatch):
    selected_ops = []
    calls = []

    def fake_require_fa2_op(name):
        selected_ops.append(name)

        def fake_op(*args):
            calls.append(args)

        return fake_op

    monkeypatch.setattr(byte_v2_ops, "_require_fa2_op", fake_require_fa2_op)
    monkeypatch.delenv(
        "BYTE_V2_STATIC_W16_FA2_PROFILE_SEQ_RANGE",
        raising=False,
    )
    tensors = [object() for _ in range(8)]
    byte_v2_ops.byte_v2_static_w16_fa2_paged_attention(
        *tensors,
        scale=0.125,
        max_query_len=1,
        max_seq_len=4096,
        causal=True,
    )

    assert selected_ops == ["static_w16_canonical_varlen_fwd"]
    assert len(calls) == 1
    output, query, cache, raw, page_map, starts, _, seq_lens = tensors
    assert calls[0][:8] == (
        query,
        cache,
        raw,
        page_map,
        output,
        starts,
        starts,
        seq_lens,
    )
    assert calls[0][-2] == 0


def test_static_w16_attention_exposes_output_and_lse(monkeypatch):
    output = object()
    lse = object()

    def fake_op(*args):
        assert args[4] is None
        return [output, lse]

    monkeypatch.setattr(byte_v2_ops, "_require_fa2_op", lambda _: fake_op)
    monkeypatch.delenv(
        "BYTE_V2_STATIC_W16_FA2_PROFILE_SEQ_RANGE",
        raising=False,
    )
    tensors = [object() for _ in range(7)]

    result = byte_v2_ops.byte_v2_static_w16_fa2_paged_attention_with_lse(
        *tensors,
        scale=0.125,
        max_query_len=8,
        max_seq_len=256,
        causal=False,
    )

    assert result == (output, lse)


def test_static_w16_profile_split_trims_only_empty_splits():
    common = {
        "device_name": "NVIDIA A40",
        "num_sms": 84,
        "query_shape": (1, 32, 128),
        "query_is_bf16": True,
        "query_is_cuda": True,
        "block_table_shape": (1, 8192),
        "max_query_len": 1,
        "max_seq_len": 131072,
        "causal": True,
        "preserve_mixed_dispatch": False,
    }

    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (4097, 4159),
            **common,
        )
        == 17
    )
    for seq_range, expected in (
        ((5121, 5183), 14),
        ((6145, 6207), 17),
        ((7169, 7231), 19),
        ((8193, 8255), 17),
    ):
        assert (
            byte_v2_ops._static_w16_fa2_profile_split_for_shape(
                seq_range,
                **common,
            )
            == expected
        )
    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (16385, 16447),
            **common,
        )
        == 19
    )
    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (65537, 65599),
            **common,
        )
        == 0
    )

    # Outside the A40 batch-one split20 override, mirror Raw-FA2's general
    # split heuristic before trimming. These ranges preserve its N=128
    # blocks-per-split value for every runtime length.
    for batch_size, seq_range, expected in (
        (2, (6913, 7168), 8),
        (4, (1921, 2048), 4),
        (8, (1921, 2048), 4),
        (16, (1921, 2048), 4),
    ):
        batched = common | {
            "query_shape": (batch_size, 32, 128),
            "block_table_shape": (batch_size, 8192),
        }
        assert (
            byte_v2_ops._static_w16_fa2_profile_split_for_shape(
                seq_range,
                **batched,
            )
            == expected
        )

    batch_17 = common | {
        "query_shape": (17, 32, 128),
        "block_table_shape": (17, 8192),
    }
    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (1921, 2048),
            **batch_17,
        )
        == 0
    )

    for override in (
        {"device_name": "NVIDIA A100-SXM4-80GB"},
        {"num_sms": 108},
        {"query_shape": (2, 32, 128)},
        {"query_is_bf16": False},
        {"block_table_shape": (2, 8192)},
        {"max_query_len": 2},
        {"causal": False},
        {"preserve_mixed_dispatch": True},
    ):
        candidate = common | override
        assert (
            byte_v2_ops._static_w16_fa2_profile_split_for_shape(
                (4097, 4159),
                **candidate,
            )
            == 0
        )

    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (4097, 4159),
            **(common | {"max_seq_len": 16384}),
        )
        == 17
    )

    for seq_range in ((2433, 2561), (4865, 5120)):
        assert (
            byte_v2_ops._static_w16_fa2_profile_split_for_shape(
                seq_range,
                **common,
            )
            == 0
        )
    assert (
        byte_v2_ops._static_w16_fa2_profile_split_for_shape(
            (8192, 8192),
            **common,
        )
        == 16
    )


@pytest.mark.parametrize(
    ("batch_size", "num_n_blocks", "expected"),
    [
        (1, 32, 16),
        (1, 128, 19),
        (1, 1024, 18),
        (2, 1024, 9),
        (4, 1024, 5),
        (16, 1024, 5),
        (17, 1024, 1),
    ],
)
def test_static_w16_fa2_split_heuristic_matches_fa2(
    batch_size: int,
    num_n_blocks: int,
    expected: int,
):
    assert (
        byte_v2_ops._fa2_num_splits_heuristic(
            batch_size * 8,
            168,
            num_n_blocks,
            128,
        )
        == expected
    )


@pytest.mark.parametrize("value", ["", "4097", "x:4159", "4160:4159"])
def test_static_w16_profile_split_rejects_invalid_range(monkeypatch, value: str):
    monkeypatch.setenv("BYTE_V2_STATIC_W16_FA2_PROFILE_SEQ_RANGE", value)
    with pytest.raises(RuntimeError, match="PROFILE_SEQ_RANGE"):
        byte_v2_ops._static_w16_fa2_profile_seq_range()
