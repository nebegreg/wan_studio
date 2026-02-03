#!/usr/bin/env bash
set -euo pipefail

# Wan Studio - Setup (uv)
# - Creates a venv with uv
# - Generates a lock file (requirements.lock.txt) and installs deps reproducibly
# - Downloads tools (RIFE / Real-ESRGAN)
# - Optionally pre-downloads the default model

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY_BIN="${PY_BIN:-python3}"
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="${PY_BIN:-python3.12}"
fi

VENV_DIR="${VENV_DIR:-.venv}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "[0/5] Installing uv (user-local)"
  "$PY_BIN" -m pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

if ! ensure_uv; then
  echo "❌ uv not found after installation. Ensure $HOME/.local/bin is in PATH."
  exit 1
fi

echo "[1/5] Creating venv: $VENV_DIR ($PY_BIN)"
uv venv --python "$PY_BIN" "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Checking torch/CUDA"
python - <<'PY'
try:
    import torch
    ok = bool(torch.cuda.is_available())
    print("torch:", torch.__version__, "cuda:", ok)
    if ok:
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch: not installed (or failed to import):", e)
    print("NOTE: Install a CUDA-enabled torch build before rendering.")
PY

echo "[3/5] Building lock + syncing deps"
# You can set DIFFUSERS_COMMIT to change the pinned ref.
DIFFUSERS_COMMIT="${DIFFUSERS_COMMIT:-8600b4c10d67b0ce200f664204358747bd53c775}"

# Patch requirements.in dynamically if user overrides commit
tmp_in=".requirements.uv.in"
sed "s/@[0-9a-f]\{40\}/@${DIFFUSERS_COMMIT}/" requirements.in > "$tmp_in"

uv pip compile "$tmp_in" -o requirements.lock.txt
uv pip sync requirements.lock.txt
rm -f "$tmp_in"

echo "[4/5] Installing tools (RIFE / Real-ESRGAN)"
mkdir -p tools
bash get_tools.sh ./tools

# Reduce CUDA fragmentation on long renders (Torch 2.9+ prefers PYTORCH_ALLOC_CONF)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODEL_ID HF_HOME

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  echo "[5/5] Pre-downloading model: $MODEL_ID"
  python - <<'PY'
from huggingface_hub import snapshot_download
import os

model_id = os.environ.get("MODEL_ID")
hf_home = os.environ.get("HF_HOME")
if not model_id:
    raise RuntimeError("MODEL_ID is empty. Set MODEL_ID env var or edit setup_uv.sh")

print("Downloading:", model_id)
print("Cache dir:", hf_home)
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
