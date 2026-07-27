# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force a common long-Q1 FA2 split count in an isolated profile worker.

This module is loaded only when its directory is prepended to ``PYTHONPATH``.
The default value of ``BYTE_V2_PROFILE_FORCE_LONG_Q1_SPLITS`` is zero, which
keeps both backends on the original FA2 heuristic while retaining identical
Python-side instrumentation overhead.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _enabled_worker() -> bool:
    return (
        "--worker" in sys.argv and "BYTE_V2_PROFILE_FORCE_LONG_Q1_SPLITS" in os.environ
    )


if _enabled_worker():
    import torch

    from vllm.v1.attention.backends import byte_v2_ops, flash_attn
    from vllm.vllm_flash_attn import flash_attn_interface

    _num_splits = int(os.environ["BYTE_V2_PROFILE_FORCE_LONG_Q1_SPLITS"])
    if _num_splits < 0 or _num_splits > 128:
        raise ValueError("BYTE_V2_PROFILE_FORCE_LONG_Q1_SPLITS must be in [0, 128]")

    _min_seq_len = int(
        os.environ.get("BYTE_V2_PROFILE_FORCE_LONG_Q1_MIN_SEQ_LEN", "32768")
    )
    if _min_seq_len <= 0:
        raise ValueError("BYTE_V2_PROFILE_FORCE_LONG_Q1_MIN_SEQ_LEN must be positive")

    _counts: Counter[str] = Counter()
    _original_flash_attn_varlen_func = flash_attn.flash_attn_varlen_func
    _original_require_fa2_op = byte_v2_ops._require_fa2_op

    def _run_forced_raw_fa2(kwargs: dict[str, Any]) -> Any:
        q, k, v = [
            flash_attn_interface.maybe_contiguous(kwargs[name])
            for name in ("q", "k", "v")
        ]
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs.get("cu_seqlens_k")
        dropout_p = float(kwargs.get("dropout_p", 0.0))
        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        window_size = kwargs.get("window_size")
        window_left, window_right = (
            (-1, -1)
            if window_size is None
            else (int(window_size[0]), int(window_size[1]))
        )
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            kwargs.get("out"),
            cu_seqlens_q,
            (torch.empty_like(cu_seqlens_q) if cu_seqlens_k is None else cu_seqlens_k),
            kwargs.get("seqused_k"),
            None,
            kwargs.get("block_table"),
            kwargs.get("alibi_slopes"),
            int(kwargs["max_seqlen_q"]),
            int(kwargs["max_seqlen_k"]),
            dropout_p,
            float(softmax_scale),
            False,
            bool(kwargs.get("causal", False)),
            window_left,
            window_right,
            float(kwargs.get("softcap", 0.0)),
            bool(kwargs.get("return_softmax_lse", False)) and dropout_p > 0.0,
            _num_splits,
            None,
        )
        if kwargs.get("return_softmax_lse", False):
            return out, softmax_lse
        return out

    def _profile_flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:
        _counts["raw_calls"] += 1
        is_long_q1 = (
            not args
            and kwargs.get("max_seqlen_q") == 1
            and int(kwargs.get("max_seqlen_k", 0)) >= _min_seq_len
            and kwargs.get("block_table") is not None
            and kwargs.get("causal") is True
            and kwargs.get("fa_version") == 2
        )
        if is_long_q1:
            _counts["raw_long_q1_calls"] += 1
            for name in (
                "scheduler_metadata",
                "q_descale",
                "k_descale",
                "v_descale",
                "s_aux",
                "mask_mod",
                "aux_tensors",
            ):
                if kwargs.get(name) is not None:
                    _counts[f"raw_long_q1_nonnull_{name}"] += 1
        if is_long_q1 and _num_splits:
            _counts["raw_forced_calls"] += 1
            return _run_forced_raw_fa2(kwargs)
        return _original_flash_attn_varlen_func(*args, **kwargs)

    def _profile_require_fa2_op(name: str) -> Any:
        op = _original_require_fa2_op(name)
        if name != "byte_v2_hybrid_varlen_fwd":
            return op

        def _profile_hybrid_op(*args: Any, **kwargs: Any) -> Any:
            _counts["byte_calls"] += 1
            positional = list(args)
            is_long_q1 = (
                len(positional) >= 24
                and positional[12] == 1
                and int(positional[13]) >= _min_seq_len
                and positional[10] is not None
                and positional[17] is True
            )
            if is_long_q1 and _num_splits:
                positional[22] = _num_splits
                _counts["byte_forced_calls"] += 1
            return op(*positional, **kwargs)

        return _profile_hybrid_op

    flash_attn.flash_attn_varlen_func = _profile_flash_attn_varlen_func
    byte_v2_ops._require_fa2_op = _profile_require_fa2_op

    def _write_trace() -> None:
        trace_path_value = os.environ.get("BYTE_V2_PROFILE_SPLIT_TRACE_PATH")
        if trace_path_value is None:
            return
        trace_path = Path(trace_path_value)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "argv": sys.argv,
            "counts": dict(_counts),
            "min_seq_len": _min_seq_len,
            "num_splits": _num_splits,
            "pid": os.getpid(),
        }
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, sort_keys=True) + "\n")

    atexit.register(_write_trace)
