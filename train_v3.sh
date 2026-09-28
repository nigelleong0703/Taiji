#!/usr/bin/env bash
# The third run, one unattended job (pod start command, no terminal): Qwen3.5-2B with the Open-Jev method
# (--head yesno, Brier) on web rows + Open-Jev browser/general rows -> calibrate -> stop the pod.
# Rerunning it (a restarted pod) resumes from the last checkpoint. Progress: port 8888, $S1_HOME/logs/train_lora.log.
cd "$(dirname "$0")"
source env.sh
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true  # worker processes pass many tensors
set -x
cat COMMIT 2>/dev/null
D=$S1_HOME/data R=$S1_HOME/runs/s1-v3 L=$S1_HOME/logs BASE=Qwen/Qwen3.5-2B
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
timeout 1800 bash build_kernels.sh || echo "WARNING: causal-conv1d not installed (much slower fallback)"
python -c "import causal_conv1d; print('causal_conv1d OK')" || true
# Batch setting: gradient checkpointing on, 49K tokens per batch (--no-grad-ckpt ran out of memory in the 0.8B check;
# a 15-step probe measured kernel warm-up, not speed, since every new sequence length compiles kernels). Override by
# writing other train.py flags to $R/speed.txt before the run.
mkdir -p $R
[ -s $R/speed.txt ] || echo "--batch-tokens 49152" > $R/speed.txt
if [ ! -f $R/final/adapter_config.json ]; then
  python train.py --base $BASE --train $D/v3_train.jsonl --val $D/v3_val.jsonl --out $R --resume --head yesno \
    --brier 0.1 --label-smoothing 0.05 --text-weight 3 $(cat $R/speed.txt) --eval-every 200 --save-every 200 \
    || echo TRAIN_FAILED
fi
[ -f $R/final/adapter_config.json ] && python calibrate.py --base $BASE --adapter $R/final --val $D/v3_val.jsonl \
  --limit 3000 --text-limit 200 > $S1_HOME/logs/calibrate_v3.txt 2>&1 && echo CALIBRATE_DONE
echo TRAIN_V3_DONE
set +x
runpodctl stop pod "$RUNPOD_POD_ID" || echo "could not stop the pod itself: stop it from the console"
sleep infinity
