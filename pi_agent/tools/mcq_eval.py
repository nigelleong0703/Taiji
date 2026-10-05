"""Zero-shot multiple-choice read-out: the base model's own answer distribution over option keys.

Each question becomes one chat turn (state, instructions, "[key] option" lines, "Answer with the key
only"), rendered by the model's chat template with thinking off. The reply is decoded under a
structured-output constraint that it is exactly one option key (greedy), so the answer is the model's
own choice in its native answer format, with no trained head. (Scoring every key from prompt
log-probabilities needs full-vocabulary logits for every prompt position and runs out of memory on 8 GB.) Same rows
and grouping as `tools/eval_s1.py`, scored the same way.

  python tools/mcq_eval.py .state/datasets/desktop-recovery-2026-10-04.jsonl \
      --url http://<host>:8011 --model /path/to/Qwen3.5-2B --out .state/eval/mcq-desktop.json
"""
import argparse
import collections
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_s1 import groups, variant_kind  # noqa: E402

def call(url, path, body):
    request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prompt_for(state, question):
    options = "\n".join(f"[{key}] {text(value)}" for key, value in question["criteria"].items())
    return (f"State:\n{text(state)}\n\nInstructions:\n{text(question.get('instructions', ''))}\n\nOptions:\n{options}\n\n"
            "Choose the single option best supported by the state and instructions. Answer with the option key only, "
            "without brackets.")


def score(args, state, question):
    """The model's own answer, decoded under a constraint that it is exactly one option key
    (vLLM structured outputs, greedy). Returns (key, {key: 1.0}): a choice, not a distribution."""
    keys = [str(key) for key in question["criteria"]]
    body = {"model": args.model, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": prompt_for(state, question)}],
            "chat_template_kwargs": {"enable_thinking": False}, "structured_outputs": {"choice": keys}}
    reply = call(args.url, "/v1/chat/completions", body)["choices"][0]["message"]["content"].strip()
    return reply, {reply: 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", nargs="+")
    parser.add_argument("--url", required=True, help="vLLM server root (no /v1)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.rows for line in open(path) if line.strip()]
    batches = groups(rows)[:args.limit] if args.limit else groups(rows)
    items = [row for batch in batches for row in batch]
    print(f"{len(items)} questions", file=sys.stderr)

    def run(row):
        started = time.perf_counter()
        choice, probabilities = score(args, row["state"], row["question"])
        return choice, probabilities, (time.perf_counter() - started) * 1000

    tally = collections.defaultdict(lambda: [0, 0])
    confusion = collections.defaultdict(collections.Counter)
    latencies, predictions = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for row, (choice, probabilities, ms) in zip(items, pool.map(run, items)):
            meta, name, gold = row["meta"], row["meta"].get("head"), str(row["label"])
            hit = int(gold == str(choice))
            for slot in ("all", f"head/{name}", f"label/{name}/{gold}", f"variant/{variant_kind(meta)}"):
                tally[slot][0] += hit
                tally[slot][1] += 1
            if name in ("dispatch", "operation"):
                confusion[name][f"{gold}->{choice}"] += 1
            latencies.append(ms)
            predictions.append({"meta": meta, "head": name, "gold": gold, "choice": str(choice), "probability": probabilities.get(gold)})
    summary = {"model": args.model, "rows": tally["all"][1],
               "accuracy": {slot: {"acc": round(hit / total, 3), "n": total} for slot, (hit, total) in sorted(tally.items())},
               "confusion": {name: dict(counter.most_common()) for name, counter in confusion.items()},
               "latency_ms_per_question": {"p50": round(statistics.median(latencies))}}
    out = json.dumps(summary, ensure_ascii=False, indent=1)
    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n")
        with open(Path(args.out).with_suffix(".rows.jsonl"), "w") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
