#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd "${experiment_dir}/../.." && pwd)
python_bin=${repo_dir}/.venv/bin/python
results_dir=${1:-${experiment_dir}/results/formal}
output_dir=${2:-${experiment_dir}/analysis/a100}
repetitions=${BYTEV2_REPETITIONS:-3}

if [[ ! -x ${python_bin} ]]; then
  echo "missing repository Python: ${python_bin}" >&2
  exit 1
fi

extra_args=()
if [[ ${BYTEV2_ALLOW_NON_A100:-0} == 1 ]]; then
  extra_args+=(--allow-non-a100)
fi
if [[ ${BYTEV2_ANALYZE_FORCE:-0} == 1 ]]; then
  extra_args+=(--force)
fi

"${python_bin}" "${experiment_dir}/analysis/analyze.py" \
  --results "${results_dir}" \
  --output-json "${output_dir}/SUMMARY.json" \
  --output-md "${output_dir}/RESULTS.md" \
  --expected-repetitions "${repetitions}" \
  "${extra_args[@]}"
