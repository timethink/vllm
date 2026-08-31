# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed loader for the experimental ByteV2 Static-W16 format."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

STATIC_W16_PAGE_BYTES = 49_792
STATIC_W16_HEADER_OFFSET_BYTES = 49_152
STATIC_W16_HEADER_BYTES = 128
STATIC_W16_NUM_RANGE_DESCRIPTORS = 16
STATIC_W16_EMPTY_RANGE_DESCRIPTOR = 0
STATIC_W16_STATUS_OFFSET_BYTES = STATIC_W16_HEADER_OFFSET_BYTES + 64
STATIC_W16_CANONICAL_METADATA_STATUS = 2
STATIC_W16_CANONICAL_RAW_FALLBACK_STATUS = STATIC_W16_CANONICAL_METADATA_STATUS | 1

_STATIC_W16_ENV = "BYTE_V2_STATIC_W16_CODEBOOK"
_STATIC_W16_RETAIN_CASCADE_Q16_ENV = "BYTE_V2_STATIC_W16_RETAIN_CASCADE_Q16"
_CODEBOOK_KIND = "bytev2_static_w16_e2e_prototype"


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
    return str(path.resolve()) if path.exists() else model


def byte_v2_static_w16_requested() -> bool:
    """Return whether the experimental Static-W16 format was selected."""
    value = os.environ.get(_STATIC_W16_ENV)
    if value is None:
        return False
    if not value.strip():
        raise RuntimeError(f"{_STATIC_W16_ENV} must not be empty")
    return True


def byte_v2_static_w16_retain_cascade_q16_requested() -> bool:
    """Return whether one post-prefix Q16 page should remain raw."""
    value = os.environ.get(_STATIC_W16_RETAIN_CASCADE_Q16_ENV)
    if value is None:
        return False
    if value not in ("0", "1"):
        raise ValueError(
            f"{_STATIC_W16_RETAIN_CASCADE_Q16_ENV} must be unset, 0, or 1; "
            f"got {value!r}"
        )
    return value == "1"


@dataclass(frozen=True)
class ByteV2StaticW16Codebook:
    """Validated immutable per-layer K/V exponent-window bases."""

    path: str
    file_sha256: str
    bases_sha256: str
    model: str
    model_config_sha256: str
    model_index_sha256: str
    bases: tuple[tuple[int, int], ...]

    def bases_for_layer(self, layer_index: int) -> tuple[int, int]:
        """Return the frozen K/V bases for one transformer layer."""
        if layer_index < 0 or layer_index >= len(self.bases):
            raise RuntimeError(
                "ByteV2 Static-W16 layer index is outside the frozen table: "
                f"{layer_index} not in [0, {len(self.bases)})"
            )
        return self.bases[layer_index]


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"ByteV2 Static-W16 codebook requires non-empty {name}")
    return value


def _validate_model_file_digest(
    *,
    model_root: Path,
    filename: str,
    expected_sha256: str,
) -> None:
    path = model_root / filename
    if not path.is_file():
        raise RuntimeError(f"ByteV2 Static-W16 model fingerprint is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"ByteV2 Static-W16 {filename} digest mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


@cache
def load_byte_v2_static_w16_codebook(
    path_text: str,
    expected_model: str,
    expected_num_layers: int,
) -> ByteV2StaticW16Codebook:
    """Load and attest one explicitly selected Static-W16 codebook."""
    path = Path(path_text).expanduser().resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot load ByteV2 Static-W16 codebook {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("ByteV2 Static-W16 codebook root must be an object")
    if payload.get("schema_version") != 1:
        raise RuntimeError("ByteV2 Static-W16 codebook schema_version must be 1")
    if payload.get("kind") != _CODEBOOK_KIND:
        raise RuntimeError(
            f"ByteV2 Static-W16 codebook kind must be {_CODEBOOK_KIND!r}"
        )
    if expected_num_layers <= 0:
        raise RuntimeError("ByteV2 Static-W16 expected layer count must be positive")

    raw_bases = payload.get("bases")
    if not isinstance(raw_bases, list) or len(raw_bases) != expected_num_layers:
        raise RuntimeError(
            "ByteV2 Static-W16 requires exactly one [K, V] base pair per layer"
        )
    bases: list[tuple[int, int]] = []
    for layer_index, pair in enumerate(raw_bases):
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError(
                "ByteV2 Static-W16 base entries must be [K, V] pairs: "
                f"layer={layer_index}"
            )
        if any(
            isinstance(base, bool) or not isinstance(base, int) or not 0 <= base <= 240
            for base in pair
        ):
            raise RuntimeError(
                "ByteV2 Static-W16 bases must be integers in [0, 240]: "
                f"layer={layer_index}, bases={pair!r}"
            )
        bases.append((pair[0], pair[1]))
    bases_sha256 = _sha256_bytes(_canonical_json_bytes(raw_bases))
    if payload.get("bases_sha256") != bases_sha256:
        raise RuntimeError("ByteV2 Static-W16 codebook bases_sha256 mismatch")

    model = _required_text(payload, "model")
    if _canonical_model_identity(model) != _canonical_model_identity(expected_model):
        raise RuntimeError(
            "ByteV2 Static-W16 codebook model mismatch: "
            f"table={model!r}, engine={expected_model!r}"
        )
    model_root = Path(expected_model).expanduser()
    if not model_root.is_dir():
        raise RuntimeError(
            "ByteV2 Static-W16 requires a local model directory, got "
            f"{expected_model!r}"
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
    return ByteV2StaticW16Codebook(
        path=str(path),
        file_sha256=_sha256_bytes(raw),
        bases_sha256=bases_sha256,
        model=str(model_root),
        model_config_sha256=model_config_sha256,
        model_index_sha256=model_index_sha256,
        bases=tuple(bases),
    )


def byte_v2_static_w16_codebook_from_env(
    *,
    expected_model: str,
    expected_num_layers: int,
) -> ByteV2StaticW16Codebook | None:
    """Return the selected Static-W16 table, or ``None`` when disabled."""
    path = os.environ.get(_STATIC_W16_ENV)
    if path is None:
        return None
    if not path.strip():
        raise RuntimeError(f"{_STATIC_W16_ENV} must not be empty")
    return load_byte_v2_static_w16_codebook(
        path,
        expected_model,
        expected_num_layers,
    )
