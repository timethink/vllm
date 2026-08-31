#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd "${experiment_dir}/../.." && pwd)
python_bin=${repo_dir}/.venv/bin/python
ninja_bin=${repo_dir}/.venv/bin/ninja
build_dir=${1:-${experiment_dir}/results/build-sm80}
build_jobs=${BYTEV2_BUILD_JOBS:-8}

if [[ ! -x ${python_bin} ]]; then
  echo "missing ${python_bin}; create the repository uv environment first" >&2
  exit 1
fi
if [[ ! -x ${ninja_bin} ]]; then
  echo "missing ${ninja_bin}; install Ninja in the repository uv environment" >&2
  exit 1
fi
if [[ ! ${build_jobs} =~ ^[1-9][0-9]*$ ]]; then
  echo "BYTEV2_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
if [[ -n ${VLLM_FLASH_ATTN_SRC_DIR:-} ]]; then
  echo "unset VLLM_FLASH_ATTN_SRC_DIR so the managed ByteV2 patch is used" >&2
  exit 2
fi

# PyTorch's CMake package derives its CUDA flags from TORCH_CUDA_ARCH_LIST and
# otherwise auto-detects the build-host GPU.  CMAKE_CUDA_ARCHITECTURES alone is
# ignored there, which would silently produce sm_86 when cross-building on A40.
TORCH_CUDA_ARCH_LIST=8.0 PATH="${repo_dir}/.venv/bin:${PATH}" cmake \
  -S "${repo_dir}" \
  -B "${build_dir}" \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${ninja_bin}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX="${build_dir}/install" \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="${python_bin}" \
  -DPython_EXECUTABLE="${python_bin}"

TORCH_CUDA_ARCH_LIST=8.0 PATH="${repo_dir}/.venv/bin:${PATH}" cmake \
  --build "${build_dir}" \
  --target _vllm_fa2_C \
  --parallel "${build_jobs}"

fa2_so=${build_dir}/vllm-flash-attn/_vllm_fa2_C.abi3.so
if [[ ! -s ${fa2_so} ]]; then
  echo "build finished without the expected extension: ${fa2_so}" >&2
  exit 1
fi
if command -v cuobjdump >/dev/null 2>&1; then
  image_list=$(cuobjdump --list-elf "${fa2_so}")
  image_found=0
  if command -v rg >/dev/null 2>&1; then
    if rg -q 'sm_80' <<< "${image_list}"; then
      image_found=1
    fi
  elif grep -q 'sm_80' <<< "${image_list}"; then
    image_found=1
  fi
  if [[ ${image_found} != 1 ]]; then
    echo "the extension does not contain an sm_80 image: ${fa2_so}" >&2
    exit 1
  fi
else
  echo "warning: cuobjdump is unavailable; sm_80 image was not audited" >&2
fi

sha256sum "${fa2_so}"
printf 'export BYTEV2_FA2_SO=%q\n' "${fa2_so}"
