#!/usr/bin/env bash
# CPU node with the s1-data network volume at /workspace: list what the volume holds, then add the next round's data
# (Valen UI screens) under $S1_HOME. Code arrives in CODE_B64. Log on port 8888; ends with DATA_ROUND_DONE or <STEP>_FAILED.
mkdir -p /workspace/s1/logs /root/code && cd /root/code
L=/workspace/s1/logs/data_round.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz || fail CODE_FAILED
export S1_HOME=/workspace/s1 SCRATCH=/workspace/s1/scratch
df -h /workspace | tail -1
du -sh $S1_HOME/* 2>/dev/null | sort -h
ls -la $S1_HOME/data | head -60
du -sh $S1_HOME/images/* 2>/dev/null
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q "huggingface_hub>=0.34" || fail INSTALL_FAILED
python prepare_valen.py --out $S1_HOME/data --images $S1_HOME/images/valen --cache $SCRATCH/valen || fail VALEN_FAILED
wc -l $S1_HOME/data/valen_ui_*.jsonl; du -sh $S1_HOME/images/valen
df -h /workspace | tail -1
# HF_DATA_REPO: also store this round in a private HF dataset (rows as JSONL, images as one tar per source, paths
# relative to $S1_HOME, so `tar -x -C $S1_HOME/images` restores the paths the rows name). HF_TOKEN stays out of the log.
if [ -n "$HF_DATA_REPO" ]; then
  mkdir -p $SCRATCH/hf/data $SCRATCH/hf/images && cp $S1_HOME/data/valen_ui_*.jsonl $SCRATCH/hf/data/ || fail STAGE_FAILED
  tar -cf $SCRATCH/hf/images/valen.tar -C $S1_HOME/images valen || fail TAR_FAILED
  set +x
  python - <<'PY' || fail UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_DATA_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=os.environ["SCRATCH"] + "/hf",
                  commit_message="Valen UI screens: rows + images")
print("uploaded to", repo, sorted(f for f in api.list_repo_files(repo, repo_type="dataset")))
PY
  set -x
fi
echo DATA_ROUND_DONE
sleep infinity
