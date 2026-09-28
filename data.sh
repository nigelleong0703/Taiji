#!/usr/bin/env bash
# Download every public source and build data/train.jsonl and data/val.jsonl. Run from this folder: bash data.sh
# Research use only: WebLINX is CC BY-NC-SA, Yelp is non-commercial; see each prepare_*.py docstring.
# Idempotent: every download and conversion is skipped when its output already exists on the volume.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
RAW=$S1_HOME/raw DATA=$S1_HOME/data
TOUCAN_TRAJECTORIES=${TOUCAN_TRAJECTORIES:-1500}   # ~15 rows each: ~23K rows, about the size of WebLINX (27.5K)
GENERAL_PER_SOURCE=${GENERAL_PER_SOURCE:-3000}     # 7 sources
GUI360_TRAJECTORIES=${GUI360_TRAJECTORIES:-2000}   # of 13,750 desktop trajectories, ~9 rows each

# WebLINX-full: real pages, boxes and screenshots per step (hours: ~40 MB per demo, 1,069 demos, rate limited).
# prepare_weblinx.py (the small chat version) is the fallback when that download is not possible.
SCRATCH=${SCRATCH:-/root/s1-scratch}
[ -f $DATA/weblinx_val.jsonl ] || python prepare_weblinx_full.py --out $DATA --images $S1_HOME/images/weblinx --cache $SCRATCH/weblinx

python -c "from huggingface_hub import snapshot_download as d; d('Agent-Ark/Toucan-1.5M', repo_type='dataset', allow_patterns=['SFT/*'], local_dir='$RAW/toucan')"
# Delete $DATA/tools_val.jsonl (or general_val.jsonl) to rebuild after changing the sizes above.
[ -f $DATA/tools_val.jsonl ] || python prepare_tools.py $RAW/toucan/SFT/*.parquet --out $DATA --max-trajectories "$TOUCAN_TRAJECTORIES"
[ -f $DATA/general_val.jsonl ] || python prepare_general.py --out $DATA --cache $RAW/general --per-source "$GENERAL_PER_SOURCE"

# Screenshot sources: shards and trajectory files go to local scratch, only cropped screenshots reach the volume.
[ -f $DATA/mind2web_val.jsonl ] || python prepare_mind2web.py --out $DATA --images $S1_HOME/images/mind2web --cache $SCRATCH/mind2web
[ -f $DATA/gui360_val.jsonl ] || python prepare_gui360.py --out $DATA --images $S1_HOME/images/gui360 --cache $SCRATCH/gui360 \
  --train-trajectories "$GUI360_TRAJECTORIES"

# WAIT / BLOCKED rows built from real pages (prepare_synthetic.py); no public source records those moments.
for split in train val; do
  n=$([ $split = train ] && echo 1500 || echo 150)
  [ -f $DATA/synthetic_$split.jsonl ] || python prepare_synthetic.py --rows $DATA/weblinx_$split.jsonl \
    $DATA/mind2web_$split.jsonl --out $DATA/synthetic_$split.jsonl --per-label $n
  # SELECT on real native dropdowns (the recordings have a few hundred): an operation row plus a select_target row.
  [ -f $DATA/select_$split.jsonl ] || python prepare_synthetic.py --rows $DATA/weblinx_$split.jsonl \
    $DATA/mind2web_$split.jsonl --out $DATA/select_$split.jsonl --per-label $n --kinds SELECT
done

# DONE at the end of every WebLINX sub-task (the chat's report-then-next-request boundaries), not only the last.
[ -f $DATA/weblinx_done_val.jsonl ] || python prepare_done.py --data $DATA --cache $SCRATCH/done

# Everything in one file (training mixes with source tags come from mix.py instead).
for split in train val; do
  cat $DATA/weblinx_$split.jsonl $DATA/tools_$split.jsonl $DATA/general_$split.jsonl \
    $DATA/mind2web_$split.jsonl $DATA/gui360_$split.jsonl $DATA/synthetic_$split.jsonl \
    $DATA/select_$split.jsonl $DATA/weblinx_done_$split.jsonl | shuf --random-source=<(yes) > $DATA/$split.jsonl
done
wc -l $DATA/*_train.jsonl $DATA/train.jsonl $DATA/val.jsonl
