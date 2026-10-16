#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:?usage: scripts/setup_sam3d_objects_env.sh <environment-prefix>}"
MAMBA_BIN="${MAMBA_BIN:-micromamba}"

if [[ ! -d "${PROJECT_ROOT}/third_party/sam-3d-objects" ]]; then
  echo "Required submodule is missing: third_party/sam-3d-objects" >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${MAMBA_BIN}" create --yes --prefix "${ENV_PREFIX}" python=3.11 pip
fi

# Open3D's Linux wheel still links against the X11/GL runtime even for
# headless point-cloud processing. Keep those shared libraries in the isolated
# Conda environment.
"${MAMBA_BIN}" install --yes --prefix "${ENV_PREFIX}" --channel conda-forge \
  glib libgl xorg-libx11
"${MAMBA_BIN}" env config vars set --prefix "${ENV_PREFIX}" \
  "LD_LIBRARY_PATH=${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "PYTHONUTF8=1"

run_python() {
  "${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" python "$@"
}

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.cache/tmp}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TMPDIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

run_python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
if ! run_python -c 'import utils3d'; then
  run_python -m pip install \
    "https://github.com/EasternJournalist/utils3d/archive/3913c65d81e05e47b9f367250cf8c0f7462a0900.tar.gz"
fi
if ! run_python -c 'import moge'; then
  run_python -m pip install --no-deps \
    "https://github.com/microsoft/MoGe/archive/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b.tar.gz"
fi
run_python -m pip install -r "${PROJECT_ROOT}/requirements/sam3d_objects_runtime.txt"
run_python -m pip install -e "${PROJECT_ROOT}/third_party/sam-3d-objects" --no-deps

if ! run_python -c 'import pytorch3d'; then
  run_python -m pip install --no-build-isolation \
    "https://github.com/facebookresearch/pytorch3d/archive/75ebeeaea0908c5527e7b1e305fbc7681382db47.tar.gz"
fi
if ! run_python -c 'import flash_attn'; then
  run_python -m pip install flash_attn==2.8.3 --no-build-isolation
fi
if ! run_python -c 'import gsplat'; then
  # The official 1.5.3 wheel includes the glm submodule omitted by GitHub tarballs.
  run_python -m pip install gsplat==1.5.3
fi
grep -v '^gsplat @' "${PROJECT_ROOT}/third_party/sam-3d-objects/requirements.inference.txt" | \
  PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
  run_python -m pip install -r /dev/stdin

"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  "${PROJECT_ROOT}/third_party/sam-3d-objects/patching/hydra"

run_python -c \
  'import hydra, kaolin, pytorch3d, torch; print(torch.__version__, torch.version.cuda, hydra.__version__, kaolin.__version__)'
