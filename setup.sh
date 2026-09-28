#!/usr/bin/env bash
# Idempotent: safe to rerun on every new pod; only missing pieces are installed or downloaded.
# Works on CPU pods too (installs and downloads only), so a cheap pod can prepare the volume before training.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
export TMPDIR=$S1_HOME/tmp   # pip unpacks multi-GB wheels; a 5 GB container disk is not enough
mkdir -p "$TMPDIR"
if [ ! -f "$S1_HOME/venv/.installed" ]; then   # marker written only after every install succeeded
  # Some images ship an old python3 (Ubuntu 20.04: 3.8); transformers 5.x needs 3.10+.
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)
  [ -n "$PY" ] || { echo "Need Python 3.10+ on this image"; exit 1; }
  "$PY" -m venv --clear "$S1_HOME/venv"   # on the volume: packages survive new pods; --clear drops a half install
  source "$S1_HOME/venv/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
  # causal-conv1d compiles CUDA code; if it fails, training still runs on the slower PyTorch fallback.
  pip install causal-conv1d --no-build-isolation || echo "WARNING: causal-conv1d not installed (slower fallback)"
  touch "$S1_HOME/venv/.installed"
fi
source "$S1_HOME/venv/bin/activate"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"):  # 0.8B is for the 5-minute environment check
    print("ready", snapshot_download(repo))
PY
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU pod)')"
echo "SETUP_DONE"
