"""Agent traces (Agent.snapshot() dumps, e.g. examples/flights.py state.json) -> S1 training rows.

Two row shapes share one JSONL; public datasets should be converted to the same shapes:
  choice: {"state": {...}, "question": {"type": "choice", "criteria": {key: ...}, "instructions": {...}},
           "label": "<criteria key>", "image": "<optional screenshot path>", "host": "<site>"}
  text:   {"task": "text", "context": <agent_format.field_context(...)>, "target": "Zurich", "host": "<site>"}
"host" only drives the train/val split, so no site appears on both sides.

Choice labels are the recorded TypeSafe choices (distillation); only the operation head and the head of the
chosen operation are kept. Text targets are what was actually typed.
"""

import argparse
import json
import random
from pathlib import Path
from urllib.parse import urlparse


def rows_from_trace(trace):
    decisions = trace.get("decisions", [])
    for step in trace.get("history", []):
        if not step.get("text"):
            continue
        # The decision that chose this field: the latest one before execution with the same choice.
        decision = next((d for d in reversed(decisions)
                         if d.get("choice") == step["choice"] and d.get("elapsed_ms", 0) <= step["executed_ms"]), None)
        if decision and isinstance(decision.get("request"), dict):
            state = decision["request"]["state"]
            action = next((c for q in decision["request"]["questions"].values() for c in q["criteria"].values()
                           if isinstance(c, dict) and c.get("element", "").endswith(step["action"])), {})
            # As agent_format.field_context builds it: the other fillable fields and their values sit beside the field.
            others = [{"label": e["label"], "value": e.get("value", "")} for e in state.get("elements", [])
                      if "TYPE_TEXT" in e.get("operations", []) and not step["action"].endswith(e["label"])]
            context = {
                "goal": trace["goal"],
                "field": {"label": step["action"], "role": action.get("role"), "value": action.get("current_value")},
                "form_fields": others[:12],
                "page": {"title": state["page"]["title"], "text": state["page"]["text"][:6000]},
                "recent_actions": [{"action": a.get("action"), "text": a.get("text")} for a in
                                   trace["history"][max(0, step["step"] - 7) : step["step"] - 1]],
            }
            host = urlparse(state["page"]["url"]).hostname or "local"
            yield {"task": "text", "context": context, "target": step["text"], "host": host}
    for decision in decisions:
        request, answers = decision.get("request"), decision.get("raw_answers")
        if not isinstance(request, dict) or not isinstance(answers, dict) or "questions" not in request:
            continue  # not a TypeSafe decision (e.g. a pure-LLM baseline)
        host = urlparse(request["state"]["page"]["url"]).hostname or "local"
        heads = ["operation", decision["operation"].lower() + "_target"]
        for name in heads:
            question, answer = request["questions"].get(name), answers.get(name)
            if question and answer and answer.get("choice") in question["criteria"]:
                row = {"state": request["state"], "question": question, "label": answer["choice"], "host": host}
                if decision.get("image"):  # teacher.py: the screenshot this decision saw
                    row["image"] = decision["image"]
                yield row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", help="state.json files or folders containing them")
    p.add_argument("--out", required=True, help="folder for train.jsonl and val.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.1, help="share of hosts held out")
    p.add_argument("--only-done", action="store_true", help="skip traces whose status is not done")
    p.add_argument("--only-success", action="store_true", help="skip traces whose independent check failed (teacher.py)")
    args = p.parse_args()

    files = [f for t in map(Path, args.traces) for f in ([t] if t.is_file() else [*t.rglob("state.json"), *t.rglob("traces/*.json")])]
    rows = []
    for f in files:
        trace = json.loads(f.read_text())
        if args.only_done and trace.get("status") != "done":
            continue
        if args.only_success and not trace.get("success"):
            continue
        rows.extend(rows_from_trace(trace))
    hosts = sorted({r["host"] for r in rows})
    random.Random(0).shuffle(hosts)
    held_out = set(hosts[: max(1, round(len(hosts) * args.val_fraction))]) if len(hosts) > 1 else set()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as train, open(out / "val.jsonl", "w") as val:
        for row in rows:
            (val if row["host"] in held_out else train).write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(files)} traces -> {len(rows)} rows; held-out hosts: {sorted(held_out)}")


if __name__ == "__main__":
    main()
