"""Measure an S1 server the way the agent loop uses it: fixed request shapes, real timings.

    python3 bench_s1.py --url http://host:8123 --key <key> [--repeat 3]

Portable: stdlib only. It reports the server's own timing split when the server provides it,
because wall-clock alone cannot tell prefill from anything else.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def page_actions(count: int):
    """A page shaped like a real one: a short form, then a long list of suggestions or dates."""
    base = [("Main menu", "button"), ("Flights", "button"), ("Round trip", "button"),
            ("1 passenger", "button"), ("Economy", "button"), ("Where from?", "textbox"),
            ("Swap origin and destination.", "button"), ("Where to?", "textbox"),
            ("Departure", "textbox"), ("Return", "textbox"), ("Search", "button")]
    actions = [{"id": "e%d" % (i + 1), "node": i + 1,
                "kind": "fill" if role == "textbox" else "click", "label": label, "role": role,
                "value": ""} for i, (label, role) in enumerate(base)]
    month = 0
    while len(actions) < count:
        i = len(actions) + 1
        actions.append({"id": "e%d" % i, "node": i, "kind": "click", "role": "gridcell",
                        "label": "Day %d %s 2026, %d SGD" % (i % 28 + 1, MONTHS[month % 12], 180 + i % 90),
                        "value": ""})
        month += 1
    return actions[:count]


def request_body(count: int, goal: str):
    """The same shape the harness builds: one state, an operation head and per-operation target heads."""
    actions = page_actions(count)
    elements, indices, targets = [], {}, {}
    for action in actions:
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            elements.append({"index": index, "label": action["label"], "role": action["role"],
                             "value": action.get("value", ""), "operations": []})
        index = indices[node]
        operation = {"click": "CLICK", "fill": "TYPE_TEXT"}[action["kind"]]
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        targets.setdefault(operation, {})[index] = action
    labels = {"CLICK": "Click an element, button, menu option or suggestion.",
              "TYPE_TEXT": "Enter or replace text in an editable field."}
    choices = {key: labels[key] for key in targets} | {"DONE": "Every requirement is met."}
    questions = {"operation": {"type": "choice", "criteria": choices,
                               "instructions": {"goal": goal, "rules": []}}}
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {index: {"element": "[%s] %s" % (index, a["label"])} for index, a in candidates.items()},
            "instructions": {"goal": goal, "operation": operation}}
    return {"state": {"page": {"url": "https://www.google.com/travel/flights", "title": "Flights",
                              "text": "Flights form " * 120},
                      "elements": elements,
                      "recent_actions": [{"action": "Search", "kind": "click", "page_changed": False}]},
            "questions": questions}


def call(url: str, key: str, body: dict, timeout: float):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--sizes", default="27,60,155")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    goal = ("Find the cheapest round-trip economy flight from Singapore to Bali (Denpasar), "
            "departing December 27, 2026.")
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=10) as response:
            print("health:", response.read().decode()[:80])
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit("server unreachable: %s" % error)

    print("%-7s %-8s %-9s %-11s %-11s %-9s %s" % (
        "options", "tokens", "total ms", "server ms", "model ms", "tok/s", "prefix_hit"))
    for size in [int(part) for part in args.sizes.split(",")]:
        body = request_body(size, goal)
        rows = []
        for attempt in range(args.repeat):
            payload, total_ms = call(args.url, args.key, body, args.timeout)
            timing = payload.get("timing") or {}
            usage = payload.get("usage") or {}
            rows.append((total_ms, payload.get("latency_ms"), timing.get("model_ms"),
                         usage.get("input_tokens"), timing.get("prefix_hit")))
        # The first call may warm kernels or build a prefix cache; report the later, steady ones.
        steady = rows[1:] if len(rows) > 1 else rows
        total = sorted(row[0] for row in steady)[len(steady) // 2]
        server = sorted(float(row[1] or 0) for row in steady)[len(steady) // 2]
        model = sorted(float(row[2] or 0) for row in steady)[len(steady) // 2]
        tokens = steady[-1][3] or 0
        print("%-7d %-8s %-9.0f %-11.0f %-11.0f %-9.0f %s" % (
            size, tokens, total, server, model, (tokens / (model / 1000)) if model else 0,
            steady[-1][4]))
    print()
    print("tok/s here is prefill throughput over the whole prompt (the decision head generates nothing).")


if __name__ == "__main__":
    main()
