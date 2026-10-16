# Installation

Use Linux with an NVIDIA GPU, a compatible driver, Conda or Micromamba, a C++ compiler, and a CUDA toolkit for native extensions. The main environment uses Python 3.12 and PyTorch 2.10.0 with CUDA 12.8 wheels. SAM 3D Objects and FoundationPose use a separate Python 3.11 / PyTorch 2.5.1 environment; do not merge their dependencies.

Run the following from the repository root. Keep the two environment names adjacent: the runtime discovers `physmind-v2-sam3d` beside the main environment by default.

```bash
git submodule update --init --recursive
conda create -n physmind-v2 python=3.12 pip -y
conda activate physmind-v2
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/main.txt
python -m pip install --no-deps -e third_party/sam3
```

Install the isolated reconstruction environment. Set `CUDA_HOME` to your CUDA toolkit and `TORCH_CUDA_ARCH_LIST` to your GPU architecture (for example, `8.0` for A100). The setup scripts accept `MAMBA_BIN=conda` as well as Micromamba.

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0
export MAMBA_BIN=conda
SAM3D_ENV="$(dirname "$CONDA_PREFIX")/physmind-v2-sam3d"
bash scripts/setup_sam3d_objects_env.sh "$SAM3D_ENV"
```

## Pretrained weights

Accept the upstream access conditions for [SAM3](https://huggingface.co/facebook/sam3) and [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects), then authenticate with `hf auth login`.

```bash
python scripts/prepare_sam3.py
python scripts/prepare_sam3d_objects.py
python scripts/prepare_track4world.py
```

These scripts use pinned revisions and verify the downloaded assets. Keep the same `HF_HOME` when running inference; SAM3 resolves its checkpoint from the local Hugging Face cache. Track4World and SAM 3D assets are placed under `data/models`.

Download the [official FoundationPose weights](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) into `third_party/FoundationPose/weights`. Both `2023-10-28-18-33-37` and `2024-01-11-20-02-45` must contain `config.yml` and `model_best.pth`. Then run:

```bash
bash scripts/setup_foundationpose_env.sh "$SAM3D_ENV"
```

The setup applies the included FoundationPose refinement-cache patch. Native CUDA extensions may compile on first use; retain their cache between runs.

## API configuration

Copy `.env.example` to `.env`, fill in `OPEN_ROUTER_KEY`, and export the same key for the independent request client:

```bash
cp .env.example .env
# Edit .env locally; never commit API keys.
set -a
source .env
set +a
python scripts/run_atw.py --help
```

The default configuration uses Gemini-3-Flash through OpenRouter. Model settings and prompts are part of the implementation. `--help` checks the entry point only; a full scene run also requires model weights, CUDA dependencies, video files and API access.
