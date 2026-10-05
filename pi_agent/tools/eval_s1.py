"""Offline S1 evaluation on labelled decision rows ({state, question, label, meta}).

Rows that share a state and variant are sent as one /v1/systemone request (one question per
head), as the harness sends dispatch and operation together. Reports accuracy per head, recall
per gold label, accuracy per variant kind (recorded / injected), latency and cached tokens.
The endpoint comes from TAIJI_S1_URL / TAIJI_S1_API_KEY, or pi_agent/.state/s1.env.

  python tools/eval_s1.py .state/datasets/recovery-2026-10-04.jsonl --out .state/eval/v4-browser.json
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

STATE_ENV = Path(__file__).resolve().parents[1] / ".state" / "s1.env"


def endpoint():
    values = {}
    if STATE_ENV.exists():
        for line in STATE_ENV.read_text().splitlines():
            name, _, value = line.removeprefix("export ").partition("=")
            values[name.strip()] = value.strip().strip("'\"")
    url = os.environ.get("TAIJI_S1_URL") or values["TAIJI_S1_URL"]
    key = os.environ.get("TAIJI_S1_API_KEY") or values["TAIJI_S1_API_KEY"]
    return url.rstrip("/") + "/v1/systemone", key


def post(url, key, body, timeout=120):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def groups(rows):
    """Rows of one state and variant -> one request; a head name appears once per request."""
    grouped = collections.OrderedDict()
    for row in rows:
        meta = row.get("meta", {})
        key = (json.dumps(row["state"], sort_keys=True, ensure_ascii=False), meta.get("variant"))
        grouped.setdefault(key, []).append(row)
    return list(grouped.values())


def variant_kind(meta):
    variant = meta.get("variant", "recorded")
    return "recorded" if variant == "recorded" else "injected"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", nargs="+")
    parser.add_argument("--out", help="summary JSON; per-row predictions go beside it as .rows.jsonl")
    parser.add_argument("--limit", type=int, help="evaluate at most this many requests")
    args = parser.parse_args()
    url, key = endpoint()
    rows = [json.loads(line) for path in args.rows for line in open(path) if line.strip()]
    batches = groups(rows)[:args.limit] if args.limit else groups(rows)
    print(f"{len(rows)} rows, {len(batches)} requests -> {url}", file=sys.stderr)

    tally = collections.defaultdict(lambda: [0, 0])
    confusion = collections.defaultdict(collections.Counter)
    latencies, cached, prompt_tokens, predictions = [], [], [], []
    for number, batch in enumerate(batches, 1):
        names = [row["meta"].get("head", f"q{i}") for i, row in enumerate(batch)]
        body = {"model": "taiji", "state": batch[0]["state"],
                "questions": {name: row["question"] for name, row in zip(names, batch)}}
        started = time.perf_counter()
        try:
            result = post(url, key, body)
        except Exception as error:
            print(f"request {number} failed: {error}", file=sys.stderr)
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        usage = result.get("usage", {})
        cached.append(usage.get("cached_tokens", 0))
        prompt_tokens.append(usage.get("input_tokens", 0))
        for name, row in zip(names, batch):
            answer = result["answers"].get(name, {})
            gold, choice = str(row["label"]), str(answer.get("choice"))
            hit = int(gold == choice)
            meta = row.get("meta", {})
            for slot in ("all", f"head/{name}", f"label/{name}/{gold}", f"variant/{variant_kind(meta)}",
                         f"env/{meta.get('kind', 'env')}"):
                tally[slot][0] += hit
                tally[slot][1] += 1
            if name in ("dispatch", "operation"):
                confusion[name][f"{gold}->{choice}"] += 1
            predictions.append({"meta": meta, "head": name, "gold": gold, "choice": choice,
                                "probability": answer.get("probabilities", {}).get(gold)})
        if number % 50 == 0:
            done = tally["all"]
            print(f"{number}/{len(batches)} acc {done[0] / done[1]:.3f}", file=sys.stderr)

    summary = {
        "endpoint": url, "rows": tally["all"][1],
        "accuracy": {slot: {"acc": round(hit / total, 3), "n": total} for slot, (hit, total) in sorted(tally.items())},
        "confusion": {name: dict(counter.most_common()) for name, counter in confusion.items()},
        "latency_ms": {"p50": round(statistics.median(latencies)),
                       "p90": round(statistics.quantiles(latencies, n=10)[-1]) if len(latencies) > 9 else None},
        "tokens": {"prompt_mean": round(statistics.mean(prompt_tokens)), "cached_mean": round(statistics.mean(cached))},
    }
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        with open(out.with_suffix(".rows.jsonl"), "w") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
