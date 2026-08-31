# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.byte_v2_attn import ByteV2AttentionImpl
from vllm.v1.attention.backends.byte_v2_static_w8 import (
    ByteV2StaticW8Codebook,
    byte_v2_static_w8_codebook_from_env,
    load_byte_v2_static_w8_codebook,
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


def _write_codebook(
    root: Path,
    *,
    num_layers: int = 2,
    model: Path | None = None,
) -> tuple[Path, Path, dict]:
    model = model or root / "model"
    model.mkdir()
    config = b'{"num_hidden_layers":2}\n'
    index = b'{"metadata":{},"weight_map":{}}\n'
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index)
    tables = {
        str(layer): {
            "K": {"static_w8_high7_base": 58},
            "V": {"static_w8_high7_base": 55 + layer},
        }
        for layer in range(num_layers)
    }
    payload = {
        "schema_version": 1,
        "kind": "bytev2_static_w8_production_ablation",
        "model": str(model),
        "model_config_sha256": _sha256(config),
        "model_index_sha256": _sha256(index),
        "tables": tables,
        "tables_sha256": _canonical_sha256(tables),
    }
    path = root / "codebook.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, model, payload


def test_load_byte_v2_static_w8_codebook(tmp_path: Path):
    path, model, payload = _write_codebook(tmp_path)
    codebook = load_byte_v2_static_w8_codebook(str(path), str(model), 2)

    assert codebook.bases == ((58, 55), (58, 56))
    assert codebook.tables_sha256 == payload["tables_sha256"]
    assert codebook.file_sha256 == _sha256(path.read_bytes())
    assert codebook.bases_for_layer(1) == (58, 56)
    with pytest.raises(RuntimeError, match="outside the frozen table"):
        codebook.bases_for_layer(2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(kind="wrong"), "codebook kind"),
        (lambda payload: payload.update(tables_sha256="0" * 64), "tables_sha256"),
        (lambda payload: payload["tables"].pop("1"), "complete table"),
        (
            lambda payload: payload["tables"]["1"]["V"].update(
                static_w8_high7_base=121
            ),
            "base must be an integer",
        ),
    ],
)
def test_static_w8_codebook_rejects_invalid_tables(
    tmp_path: Path,
    mutation,
    message: str,
):
    path, model, payload = _write_codebook(tmp_path)
    mutation(payload)
    if message != "tables_sha256":
        payload["tables_sha256"] = _canonical_sha256(payload["tables"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_byte_v2_static_w8_codebook(str(path), str(model), 2)


def test_static_w8_codebook_rejects_model_mismatch(tmp_path: Path):
    path, _, _ = _write_codebook(tmp_path)
    other_model = tmp_path / "other-model"
    other_model.mkdir()

    with pytest.raises(RuntimeError, match="model mismatch"):
        load_byte_v2_static_w8_codebook(str(path), str(other_model), 2)


def test_static_w8_codebook_env_is_explicit(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BYTE_V2_STATIC_W8_CODEBOOK", raising=False)
    assert (
        byte_v2_static_w8_codebook_from_env(
            expected_model=str(tmp_path),
            expected_num_layers=2,
        )
        is None
    )

    monkeypatch.setenv("BYTE_V2_STATIC_W8_CODEBOOK", "")
    with pytest.raises(RuntimeError, match="must not be empty"):
        byte_v2_static_w8_codebook_from_env(
            expected_model=str(tmp_path),
            expected_num_layers=2,
        )


def test_static_w8_layer_name_configures_production_writer():
    codebook = ByteV2StaticW8Codebook(
        path="/codebook.json",
        file_sha256="f" * 64,
        tables_sha256="t" * 64,
        model="/model",
        model_config_sha256="c" * 64,
        model_index_sha256="i" * 64,
        bases=((58, 55), (58, 56)),
    )
    configured: list[tuple[int, int]] = []
    impl = ByteV2AttentionImpl.__new__(ByteV2AttentionImpl)
    impl.static_w8_codebook = codebook
    impl.static_w8_layer_index = None
    impl.raw_staging_manager = SimpleNamespace(
        configure_static_w8_bases=configured.append
    )

    impl._configure_static_w8_writer(
        SimpleNamespace(layer_name="model.layers.1.self_attn")
    )

    assert impl.static_w8_layer_index == 1
    assert configured == [(58, 56)]
    with pytest.raises(RuntimeError, match="changed layer identity"):
        impl._configure_static_w8_writer(
            SimpleNamespace(layer_name="model.layers.0.self_attn")
        )
