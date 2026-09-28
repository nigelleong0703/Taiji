"""Toucan (Agent-Ark/Toucan-1.5M, Apache-2.0, real MCP tool executions) -> S1 tool-calling rows.

Input: the dataset's parquet files (or JSONL with the same fields: tools, messages). Output rows use the
shapes from prepare.py and are built by tools.py, the same code serve.py uses at inference time.

Every tool_call becomes a decision point, and so does every final assistant reply (label: none). The
assistant's lead-in text right before a call is dropped from the context: it names the tool that is about
to be called, and at inference time no such text exists yet.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from tools import NONE, function, parse_call, render, rows_for_call


def records(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=["tools", "messages"], batch_size=256):
            yield from batch.to_pylist()
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def as_list(value):
    return json.loads(value) if isinstance(value, str) else value


def decisions(messages):
    for i, message in enumerate(messages):
        role = message.get("role")
        final_reply = role == "assistant" and message.get("content") and (
            i + 1 == len(messages) or messages[i + 1].get("role") != "tool_call")
        if role != "tool_call" and not final_reply:
            continue
        context = messages[:i]
        while context and context[-1].get("role") == "assistant" and not context[-1].get("tool_calls"):
            context = context[:-1]  # lead-in text would leak the decision
        if not any(m.get("role") == "user" for m in context):
            continue
        name, arguments = parse_call(message["content"]) if role == "tool_call" else (NONE, {})
        yield context, name, arguments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Toucan parquet or JSONL files")
    p.add_argument("--out", required=True, help="folder for tools_train.jsonl and tools_val.jsonl")
    p.add_argument("--max-trajectories", type=int, default=0, help="0 = all")
    p.add_argument("--val-fraction", type=float, default=0.05, help="share of tool sets held out")
    p.add_argument("--keep-no-call", type=float, default=0.1,
                   help="share of trajectories with no tool call to keep (Toucan's 'irrelevant' subset is most "
                        "of some files; keeping all of it teaches the model to answer none too often)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"trajectories": 0, "train": 0, "val": 0, "skipped": 0, "no_call_dropped": 0}
    with open(out / "tools_train.jsonl", "w") as train, open(out / "tools_val.jsonl", "w") as val:
        for path in map(Path, args.inputs):
            for record in records(path):
                if args.max_trajectories and counts["trajectories"] >= args.max_trajectories:
                    break
                try:
                    tools, messages = as_list(record["tools"]), as_list(record["messages"])
                    if not any(m.get("role") == "tool_call" for m in messages) and \
                            random.Random(json.dumps(messages)[:2000]).random() >= args.keep_no_call:
                        counts["no_call_dropped"] += 1
                        continue
                    names = {function(t)["name"] for t in tools}
                    # Split by tool set, so validation measures tools the model never trained on.
                    group = hashlib.sha1(json.dumps(sorted(names)).encode()).hexdigest()
                    held_out = random.Random(group).random() < args.val_fraction
                    rows = []
                    for context, called, arguments in decisions(messages):
                        if called != NONE and called not in names:
                            raise ValueError("call to an unlisted tool")
                        rows += rows_for_call(render(context), tools, called, arguments)
                except (ValueError, KeyError, TypeError, SyntaxError, StopIteration):
                    counts["skipped"] += 1
                    continue
                counts["trajectories"] += 1
                sink, key = (val, "val") if held_out else (train, "train")
                for row in rows:
                    row["host"] = group[:12]
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[key] += len(rows)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
