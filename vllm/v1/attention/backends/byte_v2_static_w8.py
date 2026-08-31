# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed loader for experimental ByteV2 Static-W8 writer tables."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

_STATIC_W8_ENV = "BYTE_V2_STATIC_W8_CODEBOOK"
_CODEBOOK_KIND = "bytev2_static_w8_production_ablation"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_model_identity(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    return model


@dataclass(frozen=True)
class ByteV2StaticW8Codebook:
    """Validated, immutable per-layer K/V bases for one model checkpoint."""

    path: str
    file_sha256: str
    tables_sha256: str
    model: str
    model_config_sha256: str
    model_index_sha256: str
    bases: tuple[tuple[int, int], ...]

    def bases_for_layer(self, layer_index: int) -> tuple[int, int]:
        """Return the frozen K/V bases for a global transformer layer."""
        if layer_index < 0 or layer_index >= len(self.bases):
            raise RuntimeError(
                "ByteV2 Static-W8 layer index is outside the frozen table: "
                f"{layer_index} not in [0, {len(self.bases)})"
            )
        return self.bases[layer_index]


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"ByteV2 Static-W8 codebook requires non-empty {name}")
    return value


def _validate_model_file_digest(
    *,
    model_root: Path,
    filename: str,
    expected_sha256: str,
) -> None:
    path = model_root / filename
    if not path.is_file():
        raise RuntimeError(
            f"ByteV2 Static-W8 model fingerprint file is missing: {path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"ByteV2 Static-W8 {filename} digest mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


@cache
def load_byte_v2_static_w8_codebook(
    path_text: str,
    expected_model: str,
    expected_num_layers: int,
) -> ByteV2StaticW8Codebook:
    """Load and validate one explicitly selected Static-W8 codebook.

    The result is cached before CUDA graph capture. An explicitly selected but
    invalid table raises instead of silently running Dynamic-W8.
    """
    path = Path(path_text).expanduser().resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot load ByteV2 Static-W8 codebook {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("ByteV2 Static-W8 codebook root must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeError("ByteV2 Static-W8 codebook schema_version must be 1")
    if payload.get("kind") != _CODEBOOK_KIND:
        raise RuntimeError(f"ByteV2 Static-W8 codebook kind must be {_CODEBOOK_KIND!r}")

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise RuntimeError("ByteV2 Static-W8 codebook tables must be an object")
    tables_sha256 = _sha256_bytes(_canonical_json_bytes(tables))
    if payload.get("tables_sha256") != tables_sha256:
        raise RuntimeError("ByteV2 Static-W8 codebook tables_sha256 mismatch")
    if expected_num_layers <= 0:
        raise RuntimeError("ByteV2 Static-W8 expected layer count must be positive")
    expected_keys = {str(index) for index in range(expected_num_layers)}
    if set(tables) != expected_keys:
        missing = sorted(expected_keys - set(tables), key=int)
        extra = sorted(set(tables) - expected_keys)
        raise RuntimeError(
            "ByteV2 Static-W8 requires one complete table per model layer; "
            f"missing={missing}, extra={extra}"
        )

    bases: list[tuple[int, int]] = []
    for layer_index in range(expected_num_layers):
        layer = tables[str(layer_index)]
        if not isinstance(layer, dict) or set(layer) != {"K", "V"}:
            raise RuntimeError(
                "ByteV2 Static-W8 layer table must contain exactly K and V: "
                f"layer={layer_index}"
            )
        layer_bases: list[int] = []
        for side in ("K", "V"):
            side_table = layer[side]
            if not isinstance(side_table, dict):
                raise RuntimeError(
                    f"ByteV2 Static-W8 layer {layer_index} {side} must be an object"
                )
            base = side_table.get("static_w8_high7_base")
            if (
                isinstance(base, bool)
                or not isinstance(base, int)
                or not 0 <= base <= 120
            ):
                raise RuntimeError(
                    "ByteV2 Static-W8 base must be an integer in [0, 120]: "
                    f"layer={layer_index}, side={side}, base={base!r}"
                )
            layer_bases.append(base)
        bases.append((layer_bases[0], layer_bases[1]))

    model = _required_text(payload, "model")
    if _canonical_model_identity(model) != _canonical_model_identity(expected_model):
        raise RuntimeError(
            "ByteV2 Static-W8 codebook model mismatch: "
            f"table={model!r}, engine={expected_model!r}"
        )
    model_root = Path(expected_model).expanduser()
    if not model_root.is_dir():
        raise RuntimeError(
            "ByteV2 Static-W8 production ablation requires a local model "
            f"directory, got {expected_model!r}"
        )
    model_root = model_root.resolve()
    model_config_sha256 = _required_text(payload, "model_config_sha256")
    model_index_sha256 = _required_text(payload, "model_index_sha256")
    _validate_model_file_digest(
        model_root=model_root,
        filename="config.json",
        expected_sha256=model_config_sha256,
    )
    _validate_model_file_digest(
        model_root=model_root,
        filename="model.safetensors.index.json",
        expected_sha256=model_index_sha256,
    )
    return ByteV2StaticW8Codebook(
        path=str(path),
        file_sha256=_sha256_bytes(raw),
        tables_sha256=tables_sha256,
        model=str(model_root),
        model_config_sha256=model_config_sha256,
        model_index_sha256=model_index_sha256,
        bases=tuple(bases),
    )


def byte_v2_static_w8_codebook_from_env(
    *,
    expected_model: str,
    expected_num_layers: int,
) -> ByteV2StaticW8Codebook | None:
    """Return the explicitly configured codebook, or ``None`` for Dynamic-W8."""
    path = os.environ.get(_STATIC_W8_ENV)
    if path is None:
        return None
    if not path.strip():
        raise RuntimeError(f"{_STATIC_W8_ENV} must not be empty")
    return load_byte_v2_static_w8_codebook(
        path,
        expected_model,
        expected_num_layers,
    )
