"""Jev on the same held-out decisions, for a reference next to calibrate.py's report. Makes paid TypeSafe API calls.

TYPESAFE_API_KEY=... python jev_baseline.py <val.jsonl> --rows 300 --out jev_report.json
Choice, noul (true at p >= 0.5) and score (nearest level) questions; hosted Jev gets the structured state, never the
screenshot, so rows S1 sees with a screenshot are compared on unequal inputs (reported separately). --dry-run counts.
"""

import argparse
import collections
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import httpx

URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")


def kind(row):
    """s1.kind() without importing torch (this runs on a laptop next to the agent's .env)."""
    question = row["question"]
    if "DONE" in question.get("criteria", {}):
        return "operation"
    instructions = question.get("instructions")
    if isinstance(instructions, dict) and "operation" in instructions:
        return "target"
    return question.get("type", "choice")


def predicted(answer):
    """Jev's typed answer -> the label form rows use: a choice key, "true"/"false", or the nearest score level."""
    if answer.get("type") == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    if answer.get("type") == "score":
        return str(round(answer["score"]))
    return answer.get("choice")


def ask(client, row):
    body = {"model": os.environ.get("TYPESAFE_MODEL", "jev-latest"), "state": row["state"],
            "questions": {"q": row["question"]}}
    headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    for _ in range(3):
        try:
            response = client.post(URL, json=body, headers=headers)
            if response.status_code == 200:
                return predicted(response.json()["answers"]["q"])
        except (httpx.HTTPError, KeyError, ValueError):
            pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("val")
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--out", default="jev_report.json")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sources", nargs="*", help="only these row sources, e.g. general tools")
    args = p.parse_args()
    with open(args.val) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if "label" in r and "question" in r and (not args.sources or r.get("source") in args.sources)]
    random.Random(0).shuffle(rows)
    rows = rows[: args.rows]
    groups = collections.Counter(f"{r.get('source', 'all')}/{kind(r)}" for r in rows)
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "paid_requests": len(rows), "by_source_kind": groups}, indent=2))
        return
    with httpx.Client(timeout=30) as client, ThreadPoolExecutor(args.workers) as pool:
        choices = list(pool.map(lambda r: ask(client, r), rows))
    stats = collections.defaultdict(lambda: [0, 0])
    for r, choice in zip(rows, choices, strict=True):
        for key in (f"{r.get('source', 'all')}/{kind(r)}", "screenshot" if r.get("image") else "no_screenshot", "all"):
            stats[key][0] += int(choice == str(r["label"]))
            stats[key][1] += 1
    report = {"rows": len(rows), "failed_requests": sum(c is None for c in choices),
              "accuracy": {k: {"rows": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(stats.items())},
              # Per row, in the shuffled order: compare two endpoints on the same questions.
              "predictions": [{"source": r.get("source"), "kind": kind(r), "label": str(r["label"]), "choice": c}
                              for r, c in zip(rows, choices, strict=True)]}
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
