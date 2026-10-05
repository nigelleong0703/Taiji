"""Native tool calling vs the discriminative read-out, on the same teacher-labelled states.

The base model gets each state as a chat turn plus tool schemas (click, type_text, press_key, scroll,
wait, read_page, look, done, ask_s2) through its own chat template, generates one call, and the call
is mapped back to (dispatch, operation, target) and scored against the teacher labels that
`tools/eval_s1.py` uses. Output tool calls are parsed from the raw text (Qwen3.5's
<tool_call><function=...><parameter=...> format), so no server-side tool parser is involved.

  python tools/native_eval.py .state/datasets/desktop-recovery-2026-10-04.jsonl \
      --url http://<host>:8011/v1 --model /path/to/Qwen3.5-2B --out .state/eval/native.json
"""
import argparse
import collections
import concurrent.futures
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_s1 import groups  # noqa: E402

KEYS = {"enter": "KEY_ENTER", "escape": "KEY_ESCAPE", "tab": "KEY_TAB", "down": "KEY_DOWN", "up": "KEY_UP"}
SYSTEM = """You are S1, a fast executor. S2 (a planner) delegated the phase below; you advance it one tool call at a time.
Call exactly one tool. Act only on elements listed in the observation, by their index.
- Call done when the phase's done condition is already visibly satisfied in the observation.
- Call ask_s2 when the page is unrelated, a needed control or capability is missing, the phase conflicts with the user's request, or repeating failed attempts would not help.
- An element noted 'already tried Nx; page did not change' did not work; prefer another way.
- read_page re-reads stale state; look takes a screenshot.

{goal}

Original user request (authoritative): {context}"""
CALL = re.compile(r"<function=([\w-]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.S)


def tool(name, description, **params):
    required = [key for key, value in params.items() if not value.pop("optional", False)]
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": params, "required": required}}}


def tools_for(operations, elements):
    indices = [e["index"] for e in elements]
    out = []
    if "CLICK" in operations:
        out.append(tool("click", "Click an element.", element={"type": "string", "enum": indices}))
    if "TYPE_TEXT" in operations:
        out.append(tool("type_text", "Replace the text of an editable element.",
                        element={"type": "string", "enum": indices}, text={"type": "string"}))
    if "SELECT" in operations:
        out.append(tool("select", "Choose a dropdown option.", element={"type": "string", "enum": indices}))
    keys = [key for key, op in KEYS.items() if op in operations]
    if keys:
        out.append(tool("press_key", "Press a key on the focused control.", key={"type": "string", "enum": keys}))
    directions = [d for d in ("down", "up") if f"SCROLL_{d.upper()}" in operations]
    if directions:
        out.append(tool("scroll", "Scroll the page or window.", direction={"type": "string", "enum": directions}))
    if "WAIT" in operations:
        out.append(tool("wait", "Wait for the page to update."))
    out += [tool("read_page", "Read a fresh observation without changing anything."),
            tool("look", "Take a screenshot of the current page or window."),
            tool("done", "The phase done condition is visibly satisfied.", summary={"type": "string"}),
            tool("ask_s2", "Hand control back to S2 for planning or recovery.", reason={"type": "string"})]
    return out


def notes(rows):
    """Failed-attempt notes that the discriminative read-out shows beside its target options."""
    found = {}
    for row in rows:
        if row["meta"].get("head", "").endswith("_target"):
            for key, value in row["question"]["criteria"].items():
                match = re.search(r"\((already tried \d+x; page did not change)\)", str(value))
                if match:
                    found[key] = match.group(1)
    return found


def request_for(rows):
    by_head = {row["meta"]["head"]: row for row in rows}
    state = rows[0]["state"]
    operation = by_head["operation"]["question"]
    attempts = notes(rows)
    elements = [{k: e[k] for k in ("index", "label", "role", "value") if e.get(k) not in (None, "")}
                | ({"note": attempts[e["index"]]} if e["index"] in attempts else {}) for e in state["elements"]]
    observation = {"page": state["page"], "elements": elements,
                   "recent_actions": [{k: h.get(k) for k in ("action", "kind", "text", "page_changed")}
                                      for h in state.get("recent_actions", [])[-10:]]}
    goal = operation["instructions"]["goal"]
    system = SYSTEM.format(goal=goal, context=state.get("task_context", goal))
    gold = {head: str(row["label"]) for head, row in by_head.items()}
    return {"system": system, "observation": observation, "tools": tools_for(operation["criteria"], elements), "gold": gold}


def parse(text):
    match = CALL.search(text or "")
    if not match:
        return None
    return match.group(1), {k: v.strip() for k, v in PARAM.findall(match.group(2))}


def decision(call):
    """(dispatch, operation, target) from one generated call."""
    if call is None:
        return None, None, None
    name, args = call
    if name == "done":
        return "DONE", "DONE", None
    if name == "ask_s2":
        return "ASK_S2", "ASK_S2", None
    operation = {"click": "CLICK", "type_text": "TYPE_TEXT", "select": "SELECT", "wait": "WAIT", "read_page": "READ",
                 "look": "LOOK", "press_key": KEYS.get(args.get("key", ""), "KEY"),
                 "scroll": "SCROLL_" + args.get("direction", "down").upper()}.get(name, name)
    return "LOCAL", operation, args.get("element")


def ask(item, args):
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": item["system"]},
                         {"role": "user", "content": json.dumps(item["observation"], ensure_ascii=False)}],
            "tools": item["tools"], "tool_choice": "none",
            "chat_template_kwargs": {"enable_thinking": args.thinking}}
    started = time.perf_counter()
    request = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    text = result["choices"][0]["message"].get("content") or ""
    return text, (time.perf_counter() - started) * 1000, result.get("usage", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", nargs="+")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.rows for line in open(path) if line.strip()]
    items = [request_for(group) for group in groups(rows) if any(r["meta"].get("head") == "operation" for r in group)]
    items = items[:args.limit] if args.limit else items
    print(f"{len(items)} states", file=sys.stderr)

    tally = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    latencies, completion, records = [], [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for item, (text, ms, usage) in zip(items, pool.map(lambda i: ask(i, args), items)):
            dispatch, operation, target = decision(parse(text))
            gold = item["gold"]
            latencies.append(ms)
            completion.append(usage.get("completion_tokens", 0))
            checks = {"parsed": dispatch is not None, "dispatch": dispatch == gold.get("dispatch"),
                      "operation": operation == gold.get("operation")}
            if "click_target" in gold:
                checks["click_target"] = operation == "CLICK" and target == gold["click_target"]
            for name, ok in checks.items():
                tally[name][0] += ok
                tally[name][1] += 1
            tally[f"dispatch/{gold.get('dispatch')}"][0] += checks["dispatch"]
            tally[f"dispatch/{gold.get('dispatch')}"][1] += 1
            confusion[f"{gold.get('operation')}->{operation}"] += 1
            records.append({"gold": gold, "dispatch": dispatch, "operation": operation, "target": target, "text": text[-400:]})
    summary = {"model": args.model, "thinking": args.thinking, "states": len(items),
               "accuracy": {k: {"acc": round(h / n, 3), "n": n} for k, (h, n) in sorted(tally.items())},
               "operation_confusion": dict(confusion.most_common(15)),
               "latency_ms": {"p50": round(statistics.median(latencies)),
                              "p90": round(statistics.quantiles(latencies, n=10)[-1]) if len(latencies) > 9 else None},
               "completion_tokens_mean": round(statistics.mean(completion))}
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        with open(Path(args.out).with_suffix(".rows.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
