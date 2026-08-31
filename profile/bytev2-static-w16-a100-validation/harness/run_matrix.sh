#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd "${experiment_dir}/../.." && pwd)
python_bin=${repo_dir}/.venv/bin/python
driver=${experiment_dir}/harness/benchmark.py
fa2_so=${BYTEV2_FA2_SO:-}
gpu_id=${BYTEV2_GPU_ID:-0}
output_dir=${1:-${experiment_dir}/results/formal}
input_kind=${BYTEV2_INPUT:-synthetic-compact}
repetitions=${BYTEV2_REPETITIONS:-3}
warmup=${BYTEV2_WARMUP:-12}
iterations=${BYTEV2_ITERATIONS:-60}
calls_per_sample=${BYTEV2_CALLS_PER_SAMPLE:-20}
read -r -a sequence_lengths <<< "${BYTEV2_SEQS:-8192 16384 32768 65536 131072}"
read -r -a split_counts <<< "${BYTEV2_SPLITS:-8 12 16 20 24 32}"

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
for value in "${repetitions}" "${warmup}" "${iterations}" \
  "${calls_per_sample}" "${sequence_lengths[@]}" "${split_counts[@]}"; do
  if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
    echo "matrix counts must all be positive integers; got ${value}" >&2
    exit 2
  fi
done
if [[ ${input_kind} != synthetic-compact && ${input_kind} != capture ]]; then
  echo "BYTEV2_INPUT must be synthetic-compact or capture" >&2
  exit 2
fi

device_args=()
if [[ ${BYTEV2_ALLOW_NON_A100:-0} == 1 ]]; then
  device_args+=(--allow-non-a100)
fi
input_args=(--input "${input_kind}")
if [[ ${input_kind} == capture ]]; then
  if [[ -z ${BYTEV2_CAPTURE:-} || ! -f ${BYTEV2_CAPTURE} ]]; then
    echo "BYTEV2_INPUT=capture requires an existing BYTEV2_CAPTURE" >&2
    exit 2
  fi
  input_args+=(--capture "${BYTEV2_CAPTURE}")
  if [[ -n ${BYTEV2_CAPTURE_LAYER:-} ]]; then
    input_args+=(--capture-layer "${BYTEV2_CAPTURE_LAYER}")
  fi
fi

num_splits=${#split_counts[@]}
for ((repetition = 1; repetition <= repetitions; repetition++)); do
  offset=$(((repetition - 1) % num_splits))
  for sequence_length in "${sequence_lengths[@]}"; do
    for ((index = 0; index < num_splits; index++)); do
      split=${split_counts[$(((index + offset) % num_splits))]}
      output=${output_dir}/s${sequence_length}/split${split}/r${repetition}.json
      printf 'run sequence=%s split=%s repetition=%s\n' \
        "${sequence_length}" "${split}" "${repetition}"
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
        --seq-len "${sequence_length}" \
        --num-splits "${split}" \
        --warmup "${warmup}" \
        --iterations "${iterations}" \
        --calls-per-sample "${calls_per_sample}" \
        --repetition "${repetition}" \
        --expected-raw-fallback-pages 0 \
        --output "${output}" \
        "${input_args[@]}" \
        "${device_args[@]}"
    done
  done
done

printf 'formal matrix results: %s\n' "${output_dir}"
