"""A training mix with every row tagged by its source, plus a matching validation mix.

python mix.py --name next --train weblinx=12000 mind2web=9000 gui360=4000 tools=3000 general=3000 synthetic=3000
Reads $S1_HOME/data/<source>_train.jsonl and <source>_val.jsonl; writes <name>_train.jsonl and <name>_val.jsonl there.
"source" lets train.py and calibrate.py report accuracy per source. --estimate prints tokens and hours (tokenizer).
"""

import argparse
import collections
import json
import os
import random
from pathlib import Path

from s1 import kind, read_rows, row_at

Ref = collections.namedtuple("Ref", "source path offset kind label")


def sample(path, n, source, rnd):
    """n random rows as refs, reading only those; rows are re-read one at a time when the mix is written (parsed
    rows carry full page states: a 30k-row mix held in memory got a small pod killed)."""
    offsets = read_rows(path)
    rnd.shuffle(offsets)
    refs = []
    for offset in offsets[:n]:
        row = row_at(path, offset)
        label = row.get("label")
        if label is None and isinstance(row.get("target"), dict):  # soft labels (Open-Jev): the most probable option
            label = max(row["target"], key=row["target"].get)
        refs.append(Ref(source, path, offset, kind(row), label))
    return refs


def balance(rows, floor, rnd, cap=4):
    """Extra copies of rare operation labels (DONE, TYPE_TEXT, SCROLL_*, SELECT, ...) until each is at least `floor`
    of the operation rows, at most `cap` times its own count: CLICK is ~75% of recorded steps otherwise."""
    ops = [r for r in rows if r.kind == "operation"]
    by_label = collections.defaultdict(list)
    for r in ops:
        if r.source != "synthetic":  # WAIT/BLOCKED are sized by their own --train count, never repeated
            by_label[r.label].append(r)
    extra = []
    for group in by_label.values():
        want = min(int(floor * len(ops)), cap * len(group)) - len(group)
        extra += [rnd.choice(group) for _ in range(max(0, want))]
    return extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train", nargs="+", required=True, help="source=rows, e.g. weblinx=12000")
    p.add_argument("--val-per-source", type=int, default=400)
    p.add_argument("--data", default=os.path.join(os.environ.get("S1_HOME", "/workspace/s1"), "data"))
    p.add_argument("--estimate", action="store_true", help="count tokens on a sample (loads the tokenizer)")
    p.add_argument("--op-floor", type=float, default=0.0,
                   help="train only: repeat rare operation labels up to this share of operation rows (e.g. 0.08)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rnd, data = random.Random(args.seed), Path(args.data)
    sizes = dict((s, int(n)) for s, n in (item.split("=") for item in args.train))
    for split, per in (("train", None), ("val", args.val_per_source)):
        rows = []
        for source, n in sizes.items():
            path = data / f"{source}_{split}.jsonl"
            if path.exists():
                rows += sample(path, per or n, source, rnd)
        if split == "train" and args.op_floor:
            rows += balance(rows, args.op_floor, rnd)
        rnd.shuffle(rows)
        out = data / f"{args.name}_{split}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({"source": r.source, **row_at(r.path, r.offset)}, ensure_ascii=False) + "\n")
        counts = collections.Counter((r.source, r.kind) for r in rows)
        ops = collections.Counter(r.label for r in rows if r.kind == "operation")
        shares = {k: f"{v} ({v / max(sum(ops.values()), 1):.0%})" for k, v in ops.most_common()}
        print(json.dumps({"split": split, "rows": len(rows), "operation_labels": shares,
                          "by_source_kind": {f"{s}/{k}": v for (s, k), v in sorted(counts.items())}}))
        if split == "train" and args.estimate:
            from transformers import AutoTokenizer

            from s1 import as_text, plain
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
            picked = [row_at(r.path, r.offset) for r in rows[:: max(1, len(rows) // 1500)]]
            mean = sum(min(8192, len(plain(tok, as_text(r["context"]) + r["target"] if r.get("task") == "text"
                                           else as_text(r["state"]) + as_text(r["question"])))
                           + (400 if r.get("image") else 0)) for r in picked) / len(picked)
            total = mean * len(rows)
            print(f"~{total / 1e6:.0f}M tokens (mean {mean:.0f}/row); LoRA at 3,150 tokens/s on one H100: "
                  f"~{total / 3150 / 3600:.1f} h")


if __name__ == "__main__":
    main()
