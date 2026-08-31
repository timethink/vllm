#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd "${experiment_dir}/../.." && pwd)
python_bin=${repo_dir}/.venv/bin/python
driver=${experiment_dir}/harness/benchmark.py
fa2_so=${BYTEV2_FA2_SO:-}
gpu_id=${BYTEV2_GPU_ID:-0}
ncu_bin=${BYTEV2_NCU_BIN:-$(command -v ncu || true)}
output_dir=${1:-${experiment_dir}/ncu/formal}
sequence_length=${BYTEV2_NCU_SEQ:-8192}
split_count=${BYTEV2_NCU_SPLIT:-20}

if [[ ! -x ${python_bin} || ! -f ${driver} ]]; then
  echo "repository Python or benchmark driver is missing" >&2
  exit 1
fi
if [[ -z ${fa2_so} || ! -f ${fa2_so} ]]; then
  echo "set BYTEV2_FA2_SO to the native sm_80 extension" >&2
  exit 2
fi
if [[ -z ${ncu_bin} || ! -x ${ncu_bin} ]]; then
  echo "set BYTEV2_NCU_BIN to the Nsight Compute executable" >&2
  exit 2
fi
if [[ ! ${gpu_id} =~ ^[0-9]+$ ]]; then
  echo "BYTEV2_GPU_ID must be one physical GPU index" >&2
  exit 2
fi
if [[ ! ${sequence_length} =~ ^[1-9][0-9]*$ ]] || \
  [[ ! ${split_count} =~ ^[1-9][0-9]*$ ]]; then
  echo "BYTEV2_NCU_SEQ and BYTEV2_NCU_SPLIT must be positive integers" >&2
  exit 2
fi

device_args=()
if [[ ${BYTEV2_ALLOW_NON_A100:-0} == 1 ]]; then
  device_args+=(--allow-non-a100)
fi

for backend in raw w16; do
  report_base=${output_dir}/${backend}
  driver_json=${output_dir}/${backend}_driver.json
  if [[ -e ${report_base}.ncu-rep || -e ${driver_json} ]]; then
    echo "refusing to overwrite ${report_base} artifacts" >&2
    exit 1
  fi
  kernel='regex:.*flash_fwd_splitkv_kernel<.*'
  if [[ ${backend} == w16 ]]; then
    kernel='regex:.*flash_fwd_splitkv_byte_v2_kernel<.*'
  fi
  mkdir -p "${output_dir}"
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
    "${ncu_bin}" \
    --target-processes all \
    --profile-from-start off \
    --kernel-name-base demangled \
    --kernel-name "${kernel}" \
    --launch-count 1 \
    --cache-control none \
    --clock-control none \
    --set full \
    --export "${report_base}" \
    "${python_bin}" "${driver}" \
    --fa2-so "${fa2_so}" \
    --seq-len "${sequence_length}" \
    --num-splits "${split_count}" \
    --warmup 2 \
    --iterations 1 \
    --calls-per-sample 1 \
    --repetition 1 \
    --input synthetic-compact \
    --expected-raw-fallback-pages 0 \
    --profile-backend "${backend}" \
    --output "${driver_json}" \
    "${device_args[@]}"
  test -s "${report_base}.ncu-rep"
  test -s "${driver_json}"
done

printf 'NCU reports: %s\n' "${output_dir}"
