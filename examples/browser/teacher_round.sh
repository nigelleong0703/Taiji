#!/usr/bin/env bash
# Pod job: teacher runs in headless Chrome (no desktop), then rows into the private HF dataset.
# CODE_B64: teacher.py, agent_eval.py, prepare.py + harness/ (jev-ultrafast). ENV_B64: the agent env (TYPESAFE_*, TEXT_MODEL_*,
# S2_MODEL_*; S2 also judges InSTA runs). HF_TOKEN, HF_DATA_REPO. WORKERS (default 4), LIMIT (tasks per worker, 0 = all),
# INSTA (InSTA-150k tasks added to the hand-checked ones), ONLY (comma-separated task ids: rerun just these).
# Log on port 8888; ends with TEACHER_DONE or <STEP>_FAILED.
mkdir -p /workspace/s1/logs /root/code && cd /root/code
L=/workspace/s1/logs/teacher.log
(setsid python3 -m http.server 8888 --directory /workspace/s1/logs > /dev/null 2>&1 &)
exec > $L 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
echo "$CODE_B64" | base64 -d | tar --no-same-owner -xz || fail CODE_FAILED
set +x; echo "$ENV_B64" | base64 -d > /root/agent.env; chmod 600 /root/agent.env; set -x
export DEBIAN_FRONTEND=noninteractive S1_HOME=/workspace/s1 OUT=/workspace/s1/teacher
curl -fsSLo /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
  && apt-get update -qq && apt-get install -y -qq /tmp/chrome.deb > /dev/null || fail CHROME_FAILED
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null && export PATH=$HOME/.local/bin:$PATH || fail UV_FAILED
cd harness && uv sync -q || fail SYNC_FAILED
W=${WORKERS:-4} UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
for i in $(seq 0 $((W - 1))); do
  # Restarted whenever it exits: a crashed Chrome otherwise fails every later task of its worker.
  (setsid bash -c "while true; do google-chrome --headless=new --no-sandbox --remote-debugging-port=$((9300 + i)) \
     --user-data-dir=/tmp/chrome$i --window-size=1120,780 --user-agent='$UA' --no-first-run about:blank; sleep 2; done" \
     > /tmp/chrome$i.log 2>&1 &)
done
sleep 5
for i in $(seq 0 $((W - 1))); do
  BU_NAME=w$i BU_CDP_URL=http://127.0.0.1:$((9300 + i)) uv run --with pyarrow --with huggingface_hub --env-file /root/agent.env python ../teacher.py \
    --out $OUT --shard $i/$W --limit ${LIMIT:-0} --insta ${INSTA:-0} ${ONLY:+--only $ONLY} > $S1_HOME/logs/teacher_w$i.log 2>&1 &
done
wait
cat $S1_HOME/logs/teacher_w*.log | grep '^{"id"' > $S1_HOME/logs/teacher_runs.jsonl
python3 - <<'PY'
import json, collections
runs = [json.loads(l) for l in open("/workspace/s1/logs/teacher_runs.jsonl")]
by = collections.defaultdict(lambda: [0, 0, 0])
for r in runs:
    k = r["id"].split("-")[0]; by[k][0] += 1; by[k][1] += r["success"]; by[k][2] += r["captcha"]
print("TEACHER_SUMMARY", {k: {"runs": v[0], "passed": v[1], "captcha": v[2]} for k, v in by.items()})
cal = [r["calibration"] for r in runs if r.get("calibration") and r["calibration"]["judge"] is not None]
print("JUDGE_VS_CHECK", {"compared": len(cal), "agree": sum(c["judge"] == c["check"] for c in cal),
      "judge_pass_check_fail": sum(c["judge"] and not c["check"] for c in cal)})
PY
uv run python ../prepare.py $OUT --out $OUT/rows --only-success || fail PREPARE_FAILED
mkdir -p /root/hf/data /root/hf/images /root/hf/raw && cp $OUT/rows/train.jsonl /root/hf/data/teacher_train.jsonl \
  && cp $OUT/rows/val.jsonl /root/hf/data/teacher_val.jsonl && cp $S1_HOME/logs/teacher_runs.jsonl /root/hf/raw/ \
  && tar -cf /root/hf/images/teacher.tar -C $S1_HOME teacher/shots && tar -czf /root/hf/raw/teacher_traces.tar.gz -C $S1_HOME teacher/traces \
  || fail STAGE_FAILED
wc -l /root/hf/data/*.jsonl; du -sh /root/hf/*
set +x
python3 -m venv /root/venv && /root/venv/bin/pip install -q "huggingface_hub>=0.34" && /root/venv/bin/python - <<'PY' || fail UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(os.environ["HF_DATA_REPO"], repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=os.environ["HF_DATA_REPO"], repo_type="dataset", folder_path="/root/hf",
                  commit_message="Teacher runs: rows (passing runs only), screenshots, raw traces")
print("uploaded", sorted(api.list_repo_files(os.environ["HF_DATA_REPO"], repo_type="dataset")))
PY
echo TEACHER_DONE
sleep infinity
