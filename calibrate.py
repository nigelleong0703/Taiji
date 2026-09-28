"""Fit the softmax temperature on held-out rows and report accuracy, calibration and top-2 margins.

Writes <adapter>/temperature.json (a global temperature plus one per question kind: operation, target, choice, noul,
score), which load() and serve.py pick up, and calibration_report.json: accuracy per source/kind, recall per gold
operation (WAIT, DONE, ...), text exact-match per source, and the margin table the agent's TIE_MARGIN is re-chosen from.
"done" answers whether a missed DONE is close (DONE's probability and rank on DONE rows, what beats it) and what a DONE
threshold would do (DONE chosen whenever p(DONE) >= t): if DONE is usually a close second, a threshold fixes it
without retraining; if p(DONE) is near zero, the model has not learned it.
"""

import argparse
import collections
import json
import math
import random
from pathlib import Path

import torch
from s1 import encode, kind, load, read_rows, row_at


def metrics(logits, labels, temperature):
    nll = correct = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(15)]  # count, confidence sum, correct sum
    margins = []
    for scores, label in zip(logits, labels):
        probs = torch.softmax(scores / temperature, -1)
        top = probs.topk(min(2, len(probs)))
        confidence, right = top.values[0].item(), int(top.indices[0].item() == label)
        nll -= math.log(max(probs[label].item(), 1e-12))
        correct += right
        b = bins[min(14, int(confidence * 15))]
        b[0] += 1
        b[1] += confidence
        b[2] += right
        margin = (top.values[0] - top.values[1]).item() if len(probs) > 1 else 1.0
        margins.append((margin, right))
    n = len(labels)
    ece = sum(abs(c - r) for _, c, r in bins) / n
    return {"accuracy": correct / n, "nll": nll / n, "ece": ece}, margins


def done_report(ops, temperature):
    """ops: (logits, gold index, keys, source) for operation rows. DONE probability, rank and thresholds."""
    rows = []
    for logits, label, keys, source in ops:
        if "DONE" not in keys:
            continue
        probs = torch.softmax(logits / temperature, -1)
        order = probs.argsort(descending=True).tolist()
        done = keys.index("DONE")
        rows.append({"p_done": probs[done].item(), "rank": order.index(done) + 1, "gold": keys[label],
                     "top": keys[order[0]], "source": source})
    gold = [r for r in rows if r["gold"] == "DONE"]
    if not gold:
        return {"done_rows": 0}
    thresholds = {}
    for t in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        picked = [r for r in rows if r["p_done"] >= t or r["top"] == "DONE"]
        hits = sum(r["gold"] == "DONE" for r in picked)
        correct = sum(r["gold"] == ("DONE" if r["p_done"] >= t or r["top"] == "DONE" else r["top"]) for r in rows)
        thresholds[f"p_done>={t}"] = {"done_recall": round(hits / len(gold), 3),
                                      "done_precision": round(hits / max(len(picked), 1), 3),
                                      "operation_accuracy": round(correct / len(rows), 3)}
    by_source = collections.defaultdict(list)
    for r in gold:
        by_source[r["source"]].append(r)
    return {"done_rows": len(gold), "operation_rows": len(rows),
            "p_done_on_done_rows": {"mean": round(sum(r["p_done"] for r in gold) / len(gold), 3),
                                    "median": round(sorted(r["p_done"] for r in gold)[len(gold) // 2], 3)},
            "rank_on_done_rows": dict(sorted(collections.Counter(r["rank"] for r in gold).items())),
            "beaten_by": dict(collections.Counter(r["top"] for r in gold if r["top"] != "DONE").most_common()),
            "recall_by_source": {k: {"rows": len(v), "recall": round(sum(r["top"] == "DONE" for r in v) / len(v), 3),
                                     "mean_p_done": round(sum(r["p_done"] for r in v) / len(v), 3)}
                                 for k, v in sorted(by_source.items())},
            "false_done_rate": round(sum(r["top"] == "DONE" and r["gold"] != "DONE" for r in rows)
                                     / max(sum(r["gold"] != "DONE" for r in rows), 1), 3),
            "thresholds": thresholds}


def fit(logits, labels):
    grid = [math.exp(x / 50) for x in range(-80, 81)]  # temperatures 0.2 .. 5
    return min(grid, key=lambda t: metrics(logits, labels, t)[0]["nll"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--limit", type=int, default=2000, help="rows read from the val file")
    p.add_argument("--text-limit", type=int, default=200, help="text rows generated for exact-match")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--image-tokens", type=int, default=400)
    p.add_argument("--min-kind-rows", type=int, default=50, help="kinds with fewer rows use the global temperature")
    args = p.parse_args()

    processor, model = load(args.base, args.adapter)
    model.eval()
    offsets = read_rows(args.val)
    random.Random(0).shuffle(offsets)
    scored, text_rows = [], collections.defaultdict(lambda: [0, 0])  # (logits, label, kind, source, gold key)
    with torch.inference_mode():
        for offset in offsets[: args.limit]:
            random.seed(offset)  # fixed option order per row, still not the file's order
            row = row_at(args.val, offset)
            source = row.get("source", "all")
            if row.get("task") == "text":
                if sum(v[1] for v in text_rows.values()) < args.text_limit:
                    exact = model.write(processor, row["context"], args.max_len) == row["target"].strip()
                    text_rows[source][0] += int(exact)
                    text_rows[source][1] += 1
                continue
            try:
                encoded = encode(processor, row, args.max_len, args.image_tokens, shuffle=True)
            except (ValueError, KeyError, OSError):
                continue
            scored.append((model(encoded).float().cpu(), encoded["label"], kind(row), source,
                           encoded["keys"][encoded["label"]], list(encoded["keys"])))

    logits, labels = [x[0] for x in scored], [x[1] for x in scored]
    before, _ = metrics(logits, labels, 1.0)
    temperature = fit(logits, labels)
    by_kind = {}
    for k in sorted({x[2] for x in scored}):
        rows = [x for x in scored if x[2] == k]
        if len(rows) >= args.min_kind_rows:
            by_kind[k] = fit([x[0] for x in rows], [x[1] for x in rows])
    after, margins = metrics(logits, labels, temperature)
    wrong = [m for m, r in margins if not r]
    right = [m for m, r in margins if r]
    table = {
        f"margin<{t}": {
            "wrong_caught": round(sum(m < t for m in wrong) / max(len(wrong), 1), 3),
            "right_flagged": round(sum(m < t for m in right) / max(len(right), 1), 3),
        }
        for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    }

    def accuracy(group):
        return {key: {"rows": len(v), "accuracy": round(sum(int(x[0].argmax().item() == x[1]) for x in v) / len(v), 3)}
                for key, v in sorted(group.items())}

    groups, recall = collections.defaultdict(list), collections.defaultdict(list)
    confusion = collections.defaultdict(collections.Counter)  # gold operation -> chosen operation
    for x in scored:
        groups[f"{x[3]}/{x[2]}"].append(x)
        if x[2] == "operation":
            recall[x[4]].append(x)  # how often each gold operation (CLICK, WAIT, DONE, ...) is chosen
            confusion[x[4]][x[5][x[0].argmax().item()]] += 1
    ops = [(x[0], x[1], x[5], x[3]) for x in scored if x[2] == "operation"]
    report = {"choice_rows": len(labels), "temperature": temperature, "temperature_by_kind": by_kind,
              "before": before, "after": after, "tie_margin_table": table,
              "by_source_kind": accuracy(groups), "operation_recall": accuracy(recall),
              "operation_confusion": {k: dict(v.most_common()) for k, v in sorted(confusion.items())},
              "done": done_report(ops, by_kind.get("operation", temperature)),
              "text": {"rows": sum(v[1] for v in text_rows.values()),
                       "exact_match": round(sum(v[0] for v in text_rows.values())
                                            / max(sum(v[1] for v in text_rows.values()), 1), 3),
                       "by_source": {k: round(v[0] / v[1], 3) for k, v in sorted(text_rows.items())}}}
    print(json.dumps(report, indent=2))
    Path(args.adapter, "temperature.json").write_text(json.dumps({"temperature": temperature, "by_kind": by_kind}))
    Path(args.adapter, "calibration_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
