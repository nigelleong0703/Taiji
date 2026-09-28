"""Drop GUI-360 DONE rows made from a finishing step that still acts (converted before prepare_gui360.py checked it).

Such a step (status OVERALL_FINISH with a click/type/drag, ~35% of trajectories) shows the screen before its last
action, so DONE there is wrong. Only the trajectories' JSONL files are downloaded (row "host" = trajectory id).
python fix_gui360_done.py --data $S1_HOME/data --cache $SCRATCH/g360   (a copy of each file is kept as *.predone)
"""

import argparse
import json
import shutil
from pathlib import Path

import hub
from prepare_gui360 import REPO


def acting_finish(path):
    steps = sorted((json.loads(line) for line in open(path)), key=lambda r: r["step_id"])
    finish = [s for s in steps if s["step"]["status"] == "OVERALL_FINISH"]
    return bool(finish) and bool(finish[-1]["step"]["action"].get("function"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cache", required=True)
    args = p.parse_args()
    for split, name in (("train", "train"), ("test", "val")):
        path = Path(args.data) / f"gui360_{name}.jsonl"
        hosts = set()
        with open(path) as f:
            for line in f:
                if '"label": "DONE"' in line:
                    hosts.add(json.loads(line)["host"])
        entries = {e["path"].rsplit("/", 1)[-1][:-len(".jsonl")]: e for e in hub.tree(REPO, f"{split}/data")
                   if e["path"].endswith(".jsonl") and "/success/" in e["path"]}
        wanted = [entries[h] for h in sorted(hosts) if h in entries]
        got = {}
        for i in range(0, len(wanted), 200):
            got.update(hub.download(REPO, wanted[i:i + 200], args.cache))
        bad = {e["path"].rsplit("/", 1)[-1][:-len(".jsonl")] for e in wanted
               if got.get(e["path"]) and acting_finish(got[e["path"]])}
        shutil.copy(path, path.with_suffix(".predone"))
        dropped, tmp = 0, path.with_suffix(".tmp")
        with open(path.with_suffix(".predone")) as f, open(tmp, "w") as out:
            for line in f:
                if '"label": "DONE"' in line:
                    row = json.loads(line)
                    if row["label"] == "DONE" and row["host"] in bad:
                        dropped += 1
                        continue
                out.write(line)
        tmp.rename(path)
        print(json.dumps({"file": path.name, "done_trajectories": len(hosts), "found": len(wanted),
                          "acting_finish": len(bad), "done_rows_dropped": dropped}), flush=True)


if __name__ == "__main__":
    main()
