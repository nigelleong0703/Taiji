"""Clean converted web rows in place (the originals are kept as <file>.raw.jsonl). No model, no GPU.

1. Click-then-type: people click a field and then type into it; the recordings make that two steps, but the agent's
   TYPE_TEXT focuses the field itself. A CLICK step whose target is an editable field, followed (same task) by a
   TYPE_TEXT step into the field with the same name, is dropped; the TYPE_TEXT step already teaches the right move.
2. WebLINX history: a typed step is logged as "textInput" when the field had no text; it gets the field's name from
   the task's own text rows (the field a value was typed into).

python clean_web.py $S1_HOME/data/{weblinx,mind2web,gui360}_{train,val}.jsonl [--dry-run]
Rows keep their order: a step is an operation row plus the target and text rows that follow it.
"""

import argparse
import collections
import json
import re
from pathlib import Path

from s1 import kind


def option_name(row):
    """The visible name of a target row's chosen option, without its index and the agent's "Open " prefix."""
    element = row["question"]["criteria"][row["label"]]["element"]
    return re.sub(r"^\[[^\]]+\] ", "", element).removeprefix("Open ").strip()


def editable(row):
    index = str(row["label"]).split(":")[0]
    return any(e["index"] == index and "TYPE_TEXT" in e["operations"] for e in row["state"]["elements"])


def steps_of(rows):
    """Group a row stream: an operation row opens a step; target/text rows of the same host join it."""
    step = None
    for row in rows:
        k = kind(row)
        if step is None or k == "operation" or step["host"] != row.get("host") or k not in {"target", "text"}:
            if step:
                yield step
            step = {"host": row.get("host"), "rows": [], "op": row["label"] if k == "operation" else None}
        step["rows"].append(row)
    if step:
        yield step


def target_of(step):
    return next((r for r in step["rows"] if kind(r) == "target"), None)


def clean(rows, stats):
    """A stream of cleaned rows; holds at most two steps, so multi-GB files fit a small pod."""
    fields = collections.defaultdict(dict)  # host -> typed value -> field name (WebLINX history repair)
    previous = None
    for step in steps_of(rows):
        if previous is not None:
            yield from finish(previous, step, fields, stats)
        previous = step
    if previous is not None:
        yield from finish(previous, None, fields, stats)


def finish(step, nxt, fields, stats):
    target, next_target = target_of(step), nxt and target_of(nxt)
    if (step["op"] == "CLICK" and target and editable(target) and nxt and nxt["host"] == step["host"]
            and nxt["op"] == "TYPE_TEXT" and next_target and option_name(next_target) == option_name(target)):
        stats["click_before_type_dropped"] += 1
        return
    for r in step["rows"]:
        if kind(r) == "text":
            fields[step["host"]][(r["target"] or "").strip()[:200]] = r["context"]["field"].get("label")
    for r in step["rows"]:
        state = r.get("state")
        for a in state.get("recent_actions", []) if isinstance(state, dict) else []:
            if a.get("action") == "textInput":
                name = fields[step["host"]].get((a.get("text") or "").strip()[:200])
                if name:
                    a["action"] = name
                    stats["history_textInput_named"] += 1
                else:
                    stats["history_textInput_left"] += 1
        yield r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--dry-run", action="store_true", help="report only")
    args = p.parse_args()
    for path in map(Path, args.files):
        raw = path.with_suffix(".raw.jsonl")
        source = raw if raw.exists() else path  # rerunning cleans the original again, never twice
        stats, before, after = collections.Counter(), collections.Counter(), collections.Counter()
        tmp = path.with_suffix(".tmp")

        def rows(source=source, before=before):
            with open(source) as f:
                for line in f:
                    row = json.loads(line)
                    if kind(row) == "operation":
                        before[row["label"]] += 1
                    yield row

        with open(tmp, "w") as out:
            for r in clean(rows(), stats):
                if kind(r) == "operation":
                    after[r["label"]] += 1
                if not args.dry_run:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"file": path.name, **stats, "ops_before": dict(before.most_common()),
                          "ops_after": dict(after.most_common())}), flush=True)
        if args.dry_run:
            tmp.unlink()
        else:
            if not raw.exists():
                path.rename(raw)
            tmp.rename(path)

if __name__ == "__main__":
    main()
