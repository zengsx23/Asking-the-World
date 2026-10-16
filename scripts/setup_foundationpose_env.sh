#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:?usage: scripts/setup_foundationpose_env.sh <sam3d-environment-prefix>}"
MAMBA_BIN="${MAMBA_BIN:-micromamba}"
FOUNDATIONPOSE_ROOT="${PROJECT_ROOT}/third_party/FoundationPose"
WEIGHT_ROOT="${FOUNDATIONPOSE_ROOT}/weights"

if [[ ! -d "${FOUNDATIONPOSE_ROOT}" ]]; then
  echo "Required submodule is missing: third_party/FoundationPose" >&2
  exit 1
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "SAM3D environment is missing: ${ENV_PREFIX}" >&2
  exit 1
fi

# Keep the pinned upstream source reproducible, while letting reference sessions
# retain CUDA allocator blocks between refinement calls. Safe to rerun setup.
refine_patch="${PROJECT_ROOT}/patches/foundationpose-refine-cache.patch"
if patch --batch --dry-run --silent --forward -p1 -d "${FOUNDATIONPOSE_ROOT}" < "${refine_patch}"; then
  patch --batch --forward -p1 -d "${FOUNDATIONPOSE_ROOT}" < "${refine_patch}"
elif ! patch --batch --dry-run --silent --reverse -p1 -d "${FOUNDATIONPOSE_ROOT}" < "${refine_patch}"; then
  echo "FoundationPose cache patch does not match the checked-out source." >&2
  exit 1
fi
required_weights=(
  "2023-10-28-18-33-37/config.yml"
  "2023-10-28-18-33-37/model_best.pth"
  "2024-01-11-20-02-45/config.yml"
  "2024-01-11-20-02-45/model_best.pth"
)
for relative_path in "${required_weights[@]}"; do
  if [[ ! -f "${WEIGHT_ROOT}/${relative_path}" ]]; then
    echo "FoundationPose weights are incomplete under ${WEIGHT_ROOT}." >&2
    echo "Download them manually from the official Google Drive folder linked in docs/installation.md." >&2
    exit 1
  fi
done

run_python() {
  "${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" python "$@"
}

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.cache/tmp}"
mkdir -p "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TMPDIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CPLUS_INCLUDE_PATH="${ENV_PREFIX}/include/eigen3${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

"${MAMBA_BIN}" install --yes --prefix "${ENV_PREFIX}" --channel conda-forge \
  boost-cpp cmake eigen ninja

run_python -m pip install \
  kornia==0.7.2 \
  numba==0.60.0 \
  pybind11==2.12.0 \
  pyglet==1.5.28 \
  pypng==0.20220715.0 \
  pyrender==0.1.45 \
  pysdf==0.1.9 \
  ruamel.yaml==0.18.6 \
  simplejson==3.19.2 \
  transformations==2024.6.1 \
  webdataset==0.2.86

if ! run_python -c 'import nvdiffrast.torch'; then
  run_python -m pip install --no-build-isolation \
    "https://github.com/NVlabs/nvdiffrast/archive/253ac4fcea7de5f396371124af597e6cc957bfae.tar.gz"
fi

PYBIND11_CMAKE_DIR="$(run_python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  cmake -S "${FOUNDATIONPOSE_ROOT}/mycpp" -B "${FOUNDATIONPOSE_ROOT}/mycpp/build" \
  "-Dpybind11_DIR=${PYBIND11_CMAKE_DIR}"
"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  cmake --build "${FOUNDATIONPOSE_ROOT}/mycpp/build" --parallel "${MAX_JOBS}"
run_python -m pip install --no-build-isolation -e \
  "${FOUNDATIONPOSE_ROOT}/bundlesdf/mycuda"

PYTHONPATH="${FOUNDATIONPOSE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  run_python -c 'import estimater, nvdiffrast.torch; print("FoundationPose setup verified")'
