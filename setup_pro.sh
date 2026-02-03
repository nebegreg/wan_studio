#!/usr/bin/env bash
set -euo pipefail

# Wan Studio - Setup PRO
# Linux / RTX 4090 friendly. Creates a reproducible venv, installs deps, downloads tools,
# and (optionally) pre-downloads the default model.

# Tip: for a fully reproducible install with lockfile, use ./setup_uv.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY_BIN="${PY_BIN:-python3}"
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="${PY_BIN:-python3.12}"
fi

VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/5] Creating venv: $VENV_DIR ($PY_BIN)"
"$PY_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading packaging toolchain"
pip install -U pip setuptools wheel

# NOTE: Torch install is intentionally not forced because GPU builds vary (cu118/cu121/cu128, NGC index, etc.).
# If you don't have torch with CUDA, install it first then re-run this script.

DIFFUSERS_COMMIT="${DIFFUSERS_COMMIT:-8600b4c10d67b0ce200f664204358747bd53c775}"

echo "[3/5] Installing Python dependencies"
# Base deps
pip install -r requirements.txt
# Pin diffusers to a known commit (more stable for Wan2.2 behavior)
pip uninstall -y diffusers >/dev/null 2>&1 || true
pip install -U "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}"
# Ensure multi-LoRA control
pip install -U accelerate transformers peft

echo "[4/5] Installing tools (RIFE / Real-ESRGAN)"
mkdir -p tools
bash get_tools.sh ./tools

# Reduce CUDA fragmentation on long renders (Torch 2.9+ uses PYTORCH_ALLOC_CONF)
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${PYTORCH_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Optional: model pre-download
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
export MODEL_ID
# Hugging Face cache location (optional override)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  echo "[5/5] Pre-downloading model: $MODEL_ID"
  python - <<PY
from huggingface_hub import snapshot_download
import os
model_id = os.environ.get("MODEL_ID")
hf_home = os.environ.get("HF_HOME")
print("Downloading:", model_id)
print("Cache dir:", hf_home)
if not model_id:
    raise RuntimeError("MODEL_ID is empty. Set MODEL_ID env var or edit setup_pro.sh")
path = snapshot_download(repo_id=model_id, cache_dir=hf_home, resume_download=True)
print("OK:", path)
PY
else
  echo "[5/5] Model download skipped (SKIP_MODEL_DOWNLOAD=1)"
fi

echo
python audit.py
python rife_diag.py || true
python mistral_diag.py || true

cat <<'TXT'

✅ Setup complete.

Run the app:
  source .venv/bin/activate
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  # (optional) export MISTRAL_API_KEY=...
  python app.py

TXT
