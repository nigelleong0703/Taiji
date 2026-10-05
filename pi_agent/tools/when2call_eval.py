"""When2Call (nvidia/When2Call, CC-BY-4.0): call a tool, ask for missing information, or decline?

Two ways to answer the same balanced sample of the MCQ test split:
  native  the model generates freely with the tools in its chat template; a <tool_call> counts as
          tool_call (and its function name is checked), a reply ending in a question as
          request_for_info, any other reply as cannot_answer.
  s1      the Taiji read-out ranks the three behaviours as one choice question over a state holding
          the request and the tool list (/v1/systemone).

  python tools/when2call_eval.py native --url http://<host>:8011/v1 --model /path/to/Qwen3.5-2B
  python tools/when2call_eval.py s1
"""
import argparse
import collections
import concurrent.futures
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_s1 import endpoint, post  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / ".state" / "datasets" / "when2call" / "test_mcq.jsonl"
LABELS = ("tool_call", "request_for_info", "cannot_answer")
SYSTEM = ("You are an assistant that can call the tools below. Call a tool only when the request is fully "
          "specified for it. If a required argument is missing, ask the user for it. If no tool can do what "
          "is asked, say you cannot do it.")
CRITERIA = {
    "tool_call": "Call one of the available tools now: the request is within a tool's purpose and every required argument is known.",
    "request_for_info": "Ask the user for missing information: a tool fits, but a required argument is not given.",
    "cannot_answer": "Decline: no available tool can perform this request.",
}


def sample(n, seed):
    rows = [json.loads(line) for line in open(DATA)]
    rng = random.Random(seed)
    out = []
    for label in LABELS:
        group = [r for r in rows if r["correct_answer"] == label]
        out += rng.sample(group, min(n, len(group)))
    rng.shuffle(out)
    return out


def tools_of(row):
    return [json.loads(t) for t in row["tools"]]


JSON_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array", "int": "integer",
              "bool": "boolean", "str": "string"}


def schema(value):
    """BFCL schemas use Python type names (dict, float, tuple, any); JSON Schema needs its own."""
    if isinstance(value, dict):
        return {k: (JSON_TYPES.get(v, v) if k == "type" and isinstance(v, str) else schema(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [schema(v) for v in value]
    return value


def native(row, args):
    tools = [{"type": "function", "function": schema({**t, "parameters": {**t["parameters"], "type": "object"}})}
             for t in tools_of(row)]
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}],
            "tools": tools, "tool_choice": "none", "chat_template_kwargs": {"enable_thinking": args.thinking}}
    started = time.perf_counter()
    request = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            text = json.load(response)["choices"][0]["message"].get("content") or ""
    except urllib.error.HTTPError as error:
        return "error", None, (time.perf_counter() - started) * 1000, error.read().decode()[:300]
    ms = (time.perf_counter() - started) * 1000
    reply = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    call = re.search(r"<function=([^>\s]+)>", reply)
    if call:
        target = json.loads(row["answers"]["tool_call"]).get("name") if row["correct_answer"] == "tool_call" else None
        return "tool_call", call.group(1) == target if target else None, ms, reply
    return ("request_for_info" if reply.rstrip().endswith("?") else "cannot_answer"), None, ms, reply


def s1(row, args):
    url, key = args.endpoint
    state = {"request": row["question"], "available_tools": [{"name": t["name"], "description": t.get("description", ""),
             "required": t.get("parameters", {}).get("required", [])} for t in tools_of(row)]}
    body = {"model": "taiji", "state": state, "questions": {"behaviour": {
        "type": "choice", "instructions": "Decide how to respond to the request with the available tools.",
        "criteria": CRITERIA}}}
    started = time.perf_counter()
    answer = post(url, key, body)["answers"]["behaviour"]
    return answer["choice"], None, (time.perf_counter() - started) * 1000, json.dumps(answer["probabilities"])


def mcq(row, args):
    """The base model's own multiple-choice answer over the same three behaviours (tools/mcq_eval.py)."""
    from mcq_eval import score
    state = {"request": row["question"], "available_tools": [{"name": t["name"], "description": t.get("description", ""),
             "required": t.get("parameters", {}).get("required", [])} for t in tools_of(row)]}
    started = time.perf_counter()
    choice, probabilities = score(args, state, {"type": "choice", "criteria": CRITERIA,
                                  "instructions": "Decide how to respond to the request with the available tools."})
    return choice, None, (time.perf_counter() - started) * 1000, json.dumps(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("native", "s1", "mcq"))
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "s1":
        args.endpoint, args.workers = endpoint(), 1
    rows = sample(args.per_label, args.seed)
    run = {"native": native, "s1": s1, "mcq": mcq}[args.mode]
    confusion, latencies, names, records = collections.Counter(), [], [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for row, (choice, name_ok, ms, raw) in zip(rows, pool.map(lambda r: run(r, args), rows)):
            confusion[(row["correct_answer"], choice)] += 1
            latencies.append(ms)
            if name_ok is not None:
                names.append(name_ok)
            records.append({"uuid": row["uuid"], "gold": row["correct_answer"], "choice": choice, "raw": raw[-300:]})
    correct = sum(v for (g, c), v in confusion.items() if g == c)
    summary = {"mode": args.mode, "thinking": args.thinking, "rows": len(rows), "accuracy": round(correct / len(rows), 3),
               "recall": {g: round(confusion[(g, g)] / max(1, sum(v for (x, _), v in confusion.items() if x == g)), 3)
                          for g in LABELS},
               "tool_name_correct": round(sum(names) / len(names), 3) if names else None,
               "confusion": {f"{g}->{c}": v for (g, c), v in sorted(confusion.items())},
               "latency_ms": {"p50": round(statistics.median(latencies)),
                              "p90": round(statistics.quantiles(latencies, n=10)[-1])}}
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        with open(Path(args.out).with_suffix(".rows.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
