"""The agent on real sites with whichever S1 its env points at (TYPESAFE_URL), each outcome checked independently.

cd harness && uv run --env-file <env> python ../agent_eval.py --repeat 2 --out report.json
A DONE is not trusted: every run's final page is checked (URL/text), so false DONEs are counted, not scored.
"""

import argparse
import base64
import json
import statistics
import time
import traceback
from urllib.parse import parse_qs, urlparse

from jev_ultrafast import Agent

TASKS = {
    "wikipedia_godel": ("https://en.wikipedia.org/wiki/Main_Page",
                        "Find and open the Wikipedia article about Gödel's incompleteness theorems.",
                        lambda p: "incompleteness_theorems" in p["url"]),
    "wikipedia_ada": ("https://en.wikipedia.org/wiki/Main_Page",
                      "Search Wikipedia for Ada Lovelace and open her article.",
                      lambda p: "Ada_Lovelace" in p["url"]),
    "books_travel": ("https://books.toscrape.com/",
                     "Open the Travel category, then open the book \"It's Only the Himalayas\".",
                     lambda p: "its-only-the-himalayas" in p["url"]),
    "flights_oct20": ("https://www.google.com/travel/flights?hl=en",
                      "Find one-way flights from Zurich to London on October 20, 2026, for one adult in economy. "
                      "Stop when matching flight options are visible.",
                      lambda p: flights_ok(p)),
    "flights_bali": ("https://www.google.com/travel/flights?hl=en",
                     "Find the cheapest round-trip economy flight for one adult from Singapore to Bali (Denpasar), "
                     "departing December 27, 2026 and returning January 2, 2027. "
                     "Stop when the cheapest flight options are visible.",
                     lambda p: bali_ok(p)),
    # Same task with "Finish" for "Stop": does the goal's "Stop" pull the model to Google's "Stops" filter?
    "flights_bali_finish": ("https://www.google.com/travel/flights?hl=en",
                            "Find the cheapest round-trip economy flight for one adult from Singapore to Bali "
                            "(Denpasar), departing December 27, 2026 and returning January 2, 2027. "
                            "Finish when the cheapest flight options are visible.",
                            lambda p: bali_ok(p)),
    "tripcom_bali": ("https://www.trip.com/flights/",
                     "Find the cheapest round-trip economy flight for one adult from Singapore to Bali (Denpasar), "
                     "departing December 27, 2026 and returning January 2, 2027. "
                     "Stop when the cheapest flight options are visible.",
                     lambda p: tripcom_ok(p)),
}


def tripcom_ok(p):
    """trip.com's result URL carries the airports and dates (dcity/acity/ddate/rdate); results visible on the page."""
    url = p["url"].lower()
    return ("trip.com" in url and "sin" in url and "dps" in url and "2026-12-27" in url and "2027-01-02" in url
            and any(w in p["text"] for w in ("Cheapest", "Lowest price", "Best", "Book")))


def bali_ok(p):
    """Round trip Singapore -> Denpasar, both dates in tfs, results visible (form values when observed)."""
    values = {a["label"].strip(): a.get("value") or "" for a in p.get("actions", [])}
    params = tfs(p["url"])
    route = ("Singapore" in values.get("Where from?", "") and any(
        w in values.get("Where to?", "") for w in ("Denpasar", "Bali"))) if values else True
    return (urlparse(p["url"]).path == "/travel/flights/search" and b"2026-12-27" in params
            and b"2027-01-02" in params and route
            and any(w in p["text"] for w in ("results returned", "Cheapest", "Best")))


def tfs(url):
    """Google Flights' search parameters (base64 protobuf in ?tfs=): date and route appear as plain bytes."""
    encoded = parse_qs(urlparse(url).query).get("tfs", [""])[0]
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError:
        return b""


def flights_ok(p):
    """As examples/flights.py verifies: the route and date live in the form fields and the URL, not the page text.
    /m/08966 is Zurich and /m/04jpl London (Google's place ids in tfs)."""
    values = {a["label"].strip(): a.get("value") for a in p.get("actions", [])}
    params = tfs(p["url"])
    route = (values.get("Where from?") == "Zürich" and values.get("Where to?") == "London") if values else (
        b"/m/08966" in params and b"/m/04jpl" in params)
    return (urlparse(p["url"]).path == "/travel/flights/search" and b"2026-10-20" in params and route
            and any(w in p["text"] for w in ("results returned", "Best", "Cheapest", "departing flights")))


def run(name, url, goal, check):
    started, agent, error = time.time(), None, None
    try:
        with Agent(url, goal, verifier=check) as agent:
            for _ in agent.run():
                pass
    except Exception as e:  # noqa: BLE001 - a failed run is a result, not a crash of the evaluation
        error = f"{type(e).__name__}: {e}"[:200]
        traceback.print_exc()
    state = getattr(agent, "state", None)
    page = (state or {}).get("page") or {}
    page = {"url": page.get("url", ""), "text": page.get("text", ""), "title": page.get("title", ""),
            "actions": page.get("actions", [])}
    decisions = (state or {}).get("decisions") or []
    latencies = [d["latency_ms"] for d in decisions if d.get("latency_ms")]
    status = (state or {}).get("status")
    verification = (state or {}).get("verification") or {}
    success = verification.get("passed") is True
    return {"task": name, "status": status, "success": success, "false_done": status == "done" and not success,
            "actions": len((state or {}).get("history") or []), "decisions": len(decisions),
            "elapsed_s": round(time.time() - started, 1),
            "decision_ms_median": round(statistics.median(latencies)) if latencies else None,
            "final_url": page["url"][:200], "final_text": page["text"][:1500],
            "verification": verification,
            "blocked_reason": (state or {}).get("blocked_reason"), "error": error,
            "operations": [d.get("operation") for d in decisions],
            # Per decision: p(DONE), input tokens and time, to see whether DONE was close and where the time goes.
            "trace": [{"op": d.get("operation"), "p_done": round((d.get("operation_probabilities") or {}).get("DONE", 0), 3),
                       "tokens": (d.get("usage") or {}).get("input_tokens"), "ms": d.get("latency_ms"), "server_ms": d.get("server_ms"), "timing": d.get("server_timing")} for d in decisions],
            # S2 use: replans when stuck (reflections) and field text the fast text model could not write; decisions
            # beyond the executed actions were discarded because the page changed before they could run.
            "s2_reflections": len((state or {}).get("reflections") or []),
            "s2_text_calls": sum(bool(t.get("escalated")) for t in (state or {}).get("text_calls") or []),
            "stale_decisions": max(0, len(decisions) - len((state or {}).get("history") or []) - 1),
            "executed": [f'{h.get("operation")} {h.get("action", "")[:60]}'
                         + (f' = {h["text"][:40]}' if h.get("text") else "")
                         for h in (state or {}).get("history") or []],
            "s2_subgoals": [f'{r.get("trigger", "")[:50]} -> {r.get("verdict")}: {r.get("text", "")[:120]}'
                            for r in (state or {}).get("reflections") or []]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--out", default="agent_eval.json")
    args = p.parse_args()
    results = []
    for name in args.tasks.split(","):
        for _ in range(args.repeat):
            r = run(name, *TASKS[name])
            results.append(r)
            keys = ("task", "status", "success", "actions", "elapsed_s", "decision_ms_median", "s2_reflections",
                    "s2_text_calls", "stale_decisions", "error")
            print(json.dumps({k: r[k] for k in keys}), flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    summary = {name: {"runs": len(rs), "success": sum(r["success"] for r in rs),
                      "false_done": sum(r["false_done"] for r in rs)}
               for name in args.tasks.split(",") if (rs := [r for r in results if r["task"] == name])}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
