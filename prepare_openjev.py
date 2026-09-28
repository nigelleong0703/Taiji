"""Open-Jev (ZefanCai/Open-Jev, CC0-1.0) -> S1 rows, soft labels kept.

The browser-drone-expansion-v1 config holds release-v2 (customer support, workflow controls, games, painting geometry,
reasoning; what Open-Jev-2B/9B trained on) plus drone and browser-control rows. Mostly synthetic, controlled tasks:
short states (median ~930 characters), so they are cheap to train on.

  choice   options "KEY: description" -> criteria {KEY: description} (so browser operation rows keep Jev's keys and
           count as "operation" rows); other options -> criteria {"1": text, ...}
  noul     options ["no", "yes"] -> criteria {"true": "yes", "false": "no"}
  score    options -> ordered levels
Writes openjev_{train,val}.jsonl (every source but browser-control-v1) and openjev_browser_{train,val}.jsonl.
python prepare_openjev.py --out $S1_HOME/data --cache $SCRATCH/openjev
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/ZefanCai/Open-Jev/resolve/main/data/"
       "browser-drone-expansion-v1-redistributable/{}-00000-of-00001.parquet")
KEYED = re.compile(r"^([A-Za-z0-9_.\-]{1,40}): (.+)$", re.S)


def question_of(kind, text, options):
    """Open-Jev (kind, question, options) -> (S1 question, option keys in Open-Jev's order)."""
    if kind == "noul":
        assert [o.lower() for o in options] == ["no", "yes"], options
        return {"type": "noul", "instructions": text, "criteria": {"true": "yes", "false": "no"}}, ["false", "true"]
    if kind == "score":
        return {"type": "score", "instructions": text, "criteria": list(options)}, [str(i) for i in range(len(options))]
    parts = [KEYED.match(o) for o in options]
    if all(parts) and len({p.group(1) for p in parts}) == len(parts):
        criteria = {p.group(1): p.group(2) for p in parts}
    else:
        criteria = {str(i + 1): o for i, o in enumerate(options)}
    return {"type": "choice", "instructions": text, "criteria": criteria}, list(criteria)


def convert(record):
    question, keys = question_of(record["kind"], record["question"], list(record["options"]))
    target = [float(t) for t in record["target"]]
    source = "openjev_browser" if record["source"] == "browser-control-v1" else "openjev"
    return source, {"state": json.loads(record["state_json"]), "question": question, "source": source,
                    "target": {k: t for k, t in zip(keys, target, strict=True)}, "host": record["group_id"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    args = p.parse_args()
    import pyarrow.parquet as pq

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("validation", "val")):
        local = Path(args.cache) / f"{split}.parquet"
        if not local.exists():
            urllib.request.urlretrieve(URL.format(split), local)
        files = {s: open(Path(args.out) / f"{s}_{name}.jsonl", "w") for s in ("openjev", "openjev_browser")}
        counts = dict.fromkeys(files, 0)
        for record in pq.read_table(local).to_pylist():
            source, row = convert(record)
            files[source].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[source] += 1
        for f in files.values():
            f.close()
        print(json.dumps({"split": name, **counts}), flush=True)


if __name__ == "__main__":
    # The three Open-Jev kinds map onto S1 questions with the target in the same order as the options.
    q, k = question_of("choice", "Next?", ["CLICK: Click one element", "DONE: The goal is visible"])
    assert q["criteria"] == {"CLICK": "Click one element", "DONE": "The goal is visible"} and k == ["CLICK", "DONE"]
    q, k = question_of("choice", "Which?", ["Paris", "Rome: the capital"])
    assert k == ["1", "2"] and q["criteria"]["2"] == "Rome: the capital"
    assert question_of("noul", "True?", ["no", "yes"])[1] == ["false", "true"]
    assert question_of("score", "How bad?", ["low", "high"])[0]["criteria"] == ["low", "high"]
    main()
