#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <byte_v2|flash_attn> <0..128 splits> <max-tokens> <output-jsonl>" >&2
  exit 2
fi

backend=$1
num_splits=$2
max_tokens=$3
output_jsonl=$4
trace_jsonl="${output_jsonl%.jsonl}.trace.jsonl"

if [[ "$backend" != "byte_v2" && "$backend" != "flash_attn" ]]; then
  echo "backend must be byte_v2 or flash_attn" >&2
  exit 2
fi
if ((num_splits < 0 || num_splits > 128)); then
  echo "splits must be in [0, 128]" >&2
  exit 2
fi
if ((max_tokens < 2)); then
  echo "max-tokens must be at least 2" >&2
  exit 2
fi
if [[ -e "$output_jsonl" || -e "$trace_jsonl" ]]; then
  echo "refusing to overwrite existing result or trace" >&2
  exit 2
fi

repo_root=/mnt/sdb/yxz/ByteV2/vllm
harness_dir="$repo_root/profile/bytev2-v6-256-fa2-a40-20260727/harness"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONPATH="$harness_dir${PYTHONPATH:+:$PYTHONPATH}"

export BYTE_V2_PROFILE_FORCE_LONG_Q1_SPLITS="$num_splits"
export BYTE_V2_PROFILE_FORCE_LONG_Q1_MIN_SEQ_LEN=32768
export BYTE_V2_PROFILE_SPLIT_TRACE_PATH="$trace_jsonl"

export BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1
export BYTE_V2_FA2_DIRECT_PREFILL=1
export BYTE_V2_DECODE_KERNEL=fa2
export BYTE_V2_DECODE_RAW_FALLBACK=0
export BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT=1
export BYTE_V2_HYBRID_FULL_WAVE_WRITER_CANDIDATE=1
export BYTE_V2_HYBRID_COOPERATIVE_WRITER_LIMIT=144
export BYTE_V2_NATIVE_RAW_STAGING_UPDATE=1
export BYTE_V2_FUSED_SINGLE_TOKEN_STAGING=1
export BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=0
export BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=0
export BYTE_V2_FUSED_COMMIT_METADATA_CLEAR=1
export BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM=1
export BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE=0
export BYTE_V2_FA2_RAW_STAGING_SLOTS=4096
unset BYTE_V2_FA2_CACHED_PREFILL_HYDRATE_TO_RAW

cd "$repo_root"
exec .venv/bin/python scripts/byte_v2_speculative_profile.py \
  --backend "$backend" \
  --context-lens 65536 \
  --spec-tokens 0 \
  --batch-size 1 \
  --max-tokens "$max_tokens" \
  --ignore-eos \
  --e2e-only \
  --collect-hybrid-state \
  --disable-prefix-caching \
  --no-enforce-eager \
  --compile-size-specialization \
  --engine-max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 20000000000 \
  --output-jsonl "$output_jsonl"
