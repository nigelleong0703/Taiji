#!/usr/bin/env bash
# Pod with the s1-data volume at /workspace: copy the earlier rounds (rows, images, the v3 run) into the private HF
# dataset $HF_DATA_REPO next to the new round. Images go as one tar per source (paths relative to $S1_HOME/images).
# Stops before uploading when the total passes 90 GB (HF private storage). Log on port 8888; BACKUP_DONE or <STEP>_FAILED.
mkdir -p /workspace/s1/logs && L=/workspace/s1/logs/backup.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
S1_HOME=/workspace/s1 OUT=/workspace/s1/scratch/hf-backup  # on the volume: the tars need room
df -h /workspace | tail -1; du -sh $S1_HOME/* 2>/dev/null | sort -h; du -sh $S1_HOME/images/* 2>/dev/null
TOTAL=$(du -sc $S1_HOME/data/*.jsonl $S1_HOME/images $S1_HOME/runs/s1-v3/final 2>/dev/null | tail -1 | cut -f1)
[ "$TOTAL" -lt $((90 * 1024 * 1024)) ] || fail TOO_BIG_FOR_HF
mkdir -p $OUT/data $OUT/images $OUT/runs && cp $S1_HOME/data/*.jsonl $OUT/data/ || fail STAGE_FAILED
[ -d $S1_HOME/runs/s1-v3/final ] && cp -r $S1_HOME/runs/s1-v3/final $OUT/runs/s1-v3-final
for d in $S1_HOME/images/*/; do s=$(basename $d); [ "$s" = valen ] || [ -s $OUT/images/$s.tar ] || tar -cf $OUT/images/$s.tar -C $S1_HOME/images $s || fail TAR_FAILED; done
ls -la $OUT/data $OUT/images
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q "huggingface_hub>=0.34" || fail INSTALL_FAILED
set +x
python - <<'PY' || fail UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_DATA_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path="/workspace/s1/scratch/hf-backup",
                  commit_message="Earlier rounds: rows, images, v3 run")
print("uploaded:", len(api.list_repo_files(repo, repo_type="dataset")), "files")
PY
echo BACKUP_DONE
sleep infinity
