#!/usr/bin/env bash
# CPU job for the third run's data: Open-Jev rows, then the v3 mix (web rows as in the "next" mix, GUI-360 left out,
# + all Open-Jev browser rows + 10K Open-Jev general rows). Log: $S1_HOME/logs/data_v3.log; ends with DATA_V3_DONE.
cd "$(dirname "$0")"
source env.sh
ulimit -n 65536 2>/dev/null || true
set -x
cat COMMIT 2>/dev/null
D=$S1_HOME/data
[ -f $D/openjev_browser_val.jsonl ] || python prepare_openjev.py --out $D --cache $S1_HOME/scratch/openjev || exit 1
python mix.py --name v3 --train weblinx=12000 mind2web=9000 tools=3000 general=3000 synthetic=1500 weblinx_done=1420 \
  select=2952 openjev_browser=13814 openjev=10000 --op-floor 0.08 --estimate || exit 1
echo DATA_V3_DONE
