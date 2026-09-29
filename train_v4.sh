#!/usr/bin/env bash
# S1 v4 on one H800/H100 (e.g. a mainland-China cloud box): continue S1 v3 on the v4 mix, then calibrate.
# Run it yourself in a terminal (tmux recommended): HF_TOKEN=... bash train_v4.sh. Rerunning resumes from the checkpoint.
# MIX: v4s (default, 59k rows, ~13 h) or v4 (108k rows, ~26 h). MAX_STEPS=N: a pilot that stops after N optimizer steps
# (16 rows each; ~300 steps is about 1.5 h) into runs/s1-<MIX>-pilot.
# Needs Python 3.10 (the causal-conv1d wheel is cp310) and ~120 GB of disk under $S1_HOME.
# Mainland China: Hugging Face goes through HF_ENDPOINT (default hf-mirror.com); set it to https://huggingface.co elsewhere.
set -euo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?set HF_TOKEN (read access to nigelleong0703/taiji-data)}"
export S1_HOME=${S1_HOME:-/root/autodl-tmp/s1} HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=$S1_HOME/hf DATA_REPO=${DATA_REPO:-nigelleong0703/taiji-data} TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
mkdir -p "$S1_HOME"/{data,images,runs,logs}
# The rows name their screenshots under /workspace/s1/images (where they were built): point that path here.
[ -e /workspace/s1 ] || { mkdir -p /workspace && ln -s "$S1_HOME" /workspace/s1; }

if [ ! -f "$S1_HOME/venv/bin/activate" ]; then
  python3.10 -m venv "$S1_HOME/venv"
  "$S1_HOME/venv/bin/pip" install -q torch==2.14.0+cu126 torchvision==0.29.0+cu126 --extra-index-url "$TORCH_INDEX"
  "$S1_HOME/venv/bin/pip" install -q transformers==5.17.0 peft==0.21.0 safetensors==0.8.0 pillow==12.3.0 \
    "huggingface_hub>=0.34" flash-linear-attention
fi
source "$S1_HOME/venv/bin/activate"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
home, repo = os.environ["S1_HOME"], os.environ["DATA_REPO"]
mix = os.environ.get("MIX", "v4s")
for f in (f"data/{mix}_train.jsonl", f"data/{mix}_val.jsonl", "images/weblinx.tar", "images/mind2web.tar", "images/valen.tar"):
    hf_hub_download(repo, f, repo_type="dataset", local_dir=home)
snapshot_download(repo, repo_type="dataset", allow_patterns=["runs/s1-v3-final/*"], local_dir=home)
wheel = hf_hub_download("nigelleong0703/Taiji-2B", "wheels/causal_conv1d-1.7.0-cp310-cp310-linux_x86_64.whl", local_dir=home)
snapshot_download("Qwen/Qwen3.5-2B")
print("downloads ready", wheel)
PY
for t in weblinx mind2web valen; do [ -d "$S1_HOME/images/$t" ] || tar -xf "$S1_HOME/images/$t.tar" -C "$S1_HOME/images"; done
python -c "import causal_conv1d" 2>/dev/null || pip install -q --no-deps "$S1_HOME"/wheels/causal_conv1d-*.whl
python -c "import causal_conv1d; print('causal_conv1d OK')"

MIX=${MIX:-v4s} MAX_STEPS=${MAX_STEPS:-0}
R=$S1_HOME/runs/s1-$MIX$([ "$MAX_STEPS" -gt 0 ] && echo -pilot || true)
# Continue from v3 at half v3's learning rate; everything else as v3 (train_v3.sh).
[ -f "$R/final/adapter_config.json" ] || python train.py --base Qwen/Qwen3.5-2B --init "$S1_HOME/runs/s1-v3-final" \
  --train "$S1_HOME/data/${MIX}_train.jsonl" --val "$S1_HOME/data/${MIX}_val.jsonl" --out "$R" --resume --head yesno \
  --lr 5e-5 --brier 0.1 --label-smoothing 0.05 --text-weight 3 --batch-tokens 49152 --eval-every 100 --save-every 100 \
  --max-steps "$MAX_STEPS" 2>&1 | tee -a "$S1_HOME/logs/train_$(basename "$R").log"
python calibrate.py --base Qwen/Qwen3.5-2B --adapter "$R/final" --val "$S1_HOME/data/${MIX}_val.jsonl" --limit 3000 \
  --text-limit 200 2>&1 | tee "$S1_HOME/logs/calibrate_$(basename "$R").txt"
echo "TRAIN_DONE: $R/final (adapter, head.pt, temperature.json, calibration_report.json)"
