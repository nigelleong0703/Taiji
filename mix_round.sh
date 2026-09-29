#!/usr/bin/env bash
# CPU pod: build the v4 mix from the private HF dataset, estimate its size, check train.py --init against the v3 run,
# upload v4_{train,val}.jsonl. CODE_B64: mix.py s1.py train.py check_init.py requirements.txt. HF_TOKEN, HF_DATA_REPO,
# MIX (the mix.py --train list). Log on port 8888; MIX_DONE or <STEP>_FAILED.
mkdir -p /workspace/s1/logs /root/code && cd /root/code
L=/workspace/s1/logs/mix.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz || fail CODE_FAILED
export S1_HOME=/workspace/s1 HF_HOME=/workspace/s1/hf
python3 -m venv /root/venv && . /root/venv/bin/activate \
  && pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
  && pip install -q transformers==5.17.0 peft==0.21.0 safetensors pillow "huggingface_hub>=0.34" || fail INSTALL_FAILED
SOURCES=$(echo $MIX | tr ' ' '\n' | cut -d= -f1)
set +x
python - $SOURCES <<'PY' || fail DOWNLOAD_FAILED
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
repo = os.environ["HF_DATA_REPO"]
for s in sys.argv[1:]:
    for split in ("train", "val"):
        hf_hub_download(repo, f"data/{s}_{split}.jsonl", repo_type="dataset", local_dir="/workspace/s1", token=os.environ["HF_TOKEN"])
snapshot_download(repo, repo_type="dataset", allow_patterns=["runs/s1-v3-final/*"], local_dir="/workspace/s1", token=os.environ["HF_TOKEN"])
print("downloaded")
PY
set -x
python mix.py --name ${NAME:-v4} --train $MIX --op-floor 0.08 --estimate || fail MIX_FAILED
python check_init.py Qwen/Qwen3.5-2B /workspace/s1/runs/s1-v3-final || fail INIT_FAILED
set +x
python - <<'PY' || fail UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
name = os.environ.get("NAME", "v4")
for split in ("train", "val"):
    api.upload_file(path_or_fileobj=f"/workspace/s1/data/{name}_{split}.jsonl", path_in_repo=f"data/{name}_{split}.jsonl",
                    repo_id=os.environ["HF_DATA_REPO"], repo_type="dataset", commit_message=f"{name} mix: {os.environ['MIX']}")
print("uploaded", name, "mix")
PY
echo MIX_DONE
sleep infinity
