#!/usr/bin/env bash
# Pod job: convert NNetNav-live and AgentTrek (prepare_axtree.py) and add the rows to the private HF dataset.
# CODE_B64: prepare_axtree.py + agent_format.py. HF_TOKEN, HF_DATA_REPO. Log on port 8888; AXTREE_DONE or <STEP>_FAILED.
mkdir -p /workspace/s1/logs /workspace/s1/data /root/code && cd /root/code
L=/workspace/s1/logs/axtree.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz || fail CODE_FAILED
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q pyarrow "huggingface_hub>=0.34" || fail INSTALL_FAILED
export HF_HOME=/workspace/s1/hf
for s in nnetnav agenttrek; do python prepare_axtree.py --source $s --out /workspace/s1/data || fail "${s^^}_FAILED"; done
wc -l /workspace/s1/data/*.jsonl
set +x
python - <<'PY' || fail UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_DATA_REPO"]
for s in ("nnetnav", "agenttrek"):
    for split in ("train", "val"):
        api.upload_file(path_or_fileobj=f"/workspace/s1/data/{s}_{split}.jsonl", path_in_repo=f"data/{s}_{split}.jsonl",
                        repo_id=repo, repo_type="dataset", commit_message=f"{s} rows (prepare_axtree.py)")
print("uploaded", [f for f in api.list_repo_files(repo, repo_type="dataset") if "nnetnav" in f or "agenttrek" in f])
PY
echo AXTREE_DONE
sleep infinity
