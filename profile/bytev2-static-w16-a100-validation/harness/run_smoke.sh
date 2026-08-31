#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd "${experiment_dir}/../.." && pwd)
python_bin=${repo_dir}/.venv/bin/python
driver=${experiment_dir}/harness/benchmark.py
fa2_so=${BYTEV2_FA2_SO:-}
gpu_id=${BYTEV2_GPU_ID:-0}
output_dir=${1:-${experiment_dir}/results/smoke}

if [[ ! -x ${python_bin} || ! -f ${driver} ]]; then
  echo "repository Python or benchmark driver is missing" >&2
  exit 1
fi
if [[ -z ${fa2_so} || ! -f ${fa2_so} ]]; then
  echo "set BYTEV2_FA2_SO to the native sm_80 extension" >&2
  exit 2
fi
if [[ ! ${gpu_id} =~ ^[0-9]+$ ]]; then
  echo "BYTEV2_GPU_ID must be one physical GPU index" >&2
  exit 2
fi

device_args=()
if [[ ${BYTEV2_ALLOW_NON_A100:-0} == 1 ]]; then
  device_args+=(--allow-non-a100)
fi

inputs=(synthetic-compact synthetic-escapes)
for input in "${inputs[@]}"; do
  output=${output_dir}/${input}.json
  extra_args=()
  if [[ ${input} == synthetic-escapes ]]; then
    extra_args+=(--permute-pages)
  fi
  env \
    -u BYTE_V2_FA2_PROFILE_Q16_GQA_SPLIT \
    -u BYTE_V2_FA2_PROFILE_RAW_KV_SMEM_ALIAS \
    -u BYTE_V2_FA2_PROFILE_STATIC_W16_ACTIVE_QUERY_WARP \
    -u BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_7P1C \
    -u BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_OVERLAP \
    -u BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_TRACE \
    -u BYTE_V2_FA2_PROFILE_STATIC_W16_SMEM_PADDING_BYTES \
    -u BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${python_bin}" "${driver}" \
    --fa2-so "${fa2_so}" \
    --seq-len 1024 \
    --num-splits 8 \
    --warmup 2 \
    --iterations 4 \
    --calls-per-sample 5 \
    --repetition 1 \
    --input "${input}" \
    --expected-raw-fallback-pages 0 \
    --output "${output}" \
    "${device_args[@]}" \
    "${extra_args[@]}"
done

printf 'smoke results: %s\n' "${output_dir}"
