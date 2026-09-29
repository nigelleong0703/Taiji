#!/usr/bin/env bash
# GPU pod: where does training time go? Short train.py runs on 3,000 real v4s rows (with images) in a few configurations;
# each log line reports tokens/s, waiting-for-data vs GPU vs optimizer seconds, and the padding share.
# CODE_B64: the training files. HF_TOKEN, DATA_REPO, STEPS (default 25), CONFIGS ("name|train.py flags;..."). Logs on port 8888.
mkdir -p /workspace/s1/logs /root/code && cd /root/code
L=/workspace/s1/logs/probe.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz || fail CODE_FAILED
export S1_HOME=/workspace/s1 HF_HOME=/workspace/s1/hf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q -r requirements.txt || fail INSTALL_FAILED
set +x
python - <<'PY' || fail DOWNLOAD_FAILED
import os
from huggingface_hub import hf_hub_download, snapshot_download
repo, home = os.environ["DATA_REPO"], "/workspace/s1"
for f in ("data/v4s_train.jsonl", "images/weblinx.tar", "images/mind2web.tar", "images/valen.tar"):
    hf_hub_download(repo, f, repo_type="dataset", local_dir=home, token=os.environ["HF_TOKEN"])
snapshot_download(repo, repo_type="dataset", allow_patterns=["runs/s1-v3-final/*"], local_dir=home, token=os.environ["HF_TOKEN"])
hf_hub_download("nigelleong0703/Taiji-2B", "wheels/causal_conv1d-1.7.0-cp310-cp310-linux_x86_64.whl", local_dir=home)
snapshot_download("Qwen/Qwen3.5-2B")
PY
set -x
for t in weblinx mind2web valen; do tar -xf $S1_HOME/images/$t.tar -C $S1_HOME/images; done
pip install -q --no-deps $S1_HOME/wheels/causal_conv1d-*.whl && python -c "import causal_conv1d; print('causal_conv1d OK')"
head -n 3000 $S1_HOME/data/v4s_train.jsonl > $S1_HOME/data/probe_train.jsonl
IFS=';' read -ra RUNS <<< "$CONFIGS"
for run in "${RUNS[@]}"; do
  name=${run%%|*}; flags=${run#*|}
  timeout 1800 python train.py --base Qwen/Qwen3.5-2B --init $S1_HOME/runs/s1-v3-final --train $S1_HOME/data/probe_train.jsonl \
    --out $S1_HOME/runs/probe-$name --head yesno --lr 5e-5 --brier 0.1 --label-smoothing 0.05 --text-weight 3 \
    --max-steps ${STEPS:-25} --log-every 5 --save-every 100000 $flags > $S1_HOME/logs/probe_$name.log 2>&1 \
    || echo "RUN_FAILED $name"
  grep -E '"step"|Error|error|out of memory' $S1_HOME/logs/probe_$name.log | tail -3 | cut -c1-400
done
echo PROBE_DONE
sleep infinity
