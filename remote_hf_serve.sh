#!/usr/bin/env bash
# Start command for a cheap GPU box, any datacenter, no volume: code from CODE_B64 (a tar.gz of the serving files),
# the adapter from a private Hugging Face repo (HF_REPO, HF_TOKEN), then check_cache.py decides --shared-prefix, then
# serve.py on port 8000 (S1_API_KEY). Logs on port 8888; SERVE_READY [--shared-prefix] or <STEP>_FAILED.
mkdir -p /workspace/out /workspace/code && cd /workspace
(setsid python3 -m http.server 8888 --directory /workspace/out > /dev/null 2>&1 &)
exec > /workspace/out/serve.log 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz -C /workspace/code || fail CODE_FAILED
[ -d /root/venv ] || { python3 -m venv /root/venv && . /root/venv/bin/activate \
  && pip install -q -r /workspace/code/requirements.txt; } || fail INSTALL_FAILED
. /root/venv/bin/activate
export HF_HOME=/root/hf
set +x  # HF_TOKEN stays out of the log
hf download "$HF_REPO" --local-dir /workspace/adapter --token "$HF_TOKEN" > /dev/null || fail DOWNLOAD_FAILED
set -x
# Prebuilt by build_wheel.sh for this torch (2.14+cu126); without it the model runs the slower PyTorch fallback.
pip install -q --no-deps /workspace/adapter/wheels/causal_conv1d-*.whl \
  && python -c "import causal_conv1d; print('causal_conv1d OK')" || echo "WARNING: causal-conv1d not installed"
BASE=${BASE:-Qwen/Qwen3.5-2B}
cd /workspace/code
python -c "from PIL import Image; Image.new('RGB', (640, 400), (200, 40, 40)).save('/workspace/red.png')"
python check_cache.py $BASE /workspace/adapter /workspace/red.png > /workspace/out/check_cache.txt 2>&1
SP=""; grep -q "OK: --shared-prefix is safe" /workspace/out/check_cache.txt && SP="--shared-prefix"
set +x  # S1_API_KEY stays out of the log
(setsid python serve.py --base $BASE --adapter /workspace/adapter --port 8000 $SP > /workspace/out/server.log 2>&1 &)
for _ in $(seq 120); do curl -sf localhost:8000/health > /dev/null && break; sleep 5; done
curl -sf localhost:8000/health || fail SERVE_FAILED
echo "SERVE_READY $SP"
# Serving memory while requests arrive (peak = the largest line): how much VRAM a local host needs.
while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /workspace/out/gpu_mem.txt; sleep 5; done
