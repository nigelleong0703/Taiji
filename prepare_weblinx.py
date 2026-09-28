"""WebLINX (McGill-NLP/WebLINX, CC BY-NC-SA 4.0: research use only) -> S1 web rows.

Each turn becomes the page the agent would have observed (elements, visible text, recent actions), and the
questions come from the agent's own build_request (agent_format.py), so training rows and live requests are
built by the same code. Text values use its field_context for the same reason.

Kept: click/submit -> CLICK, text_input -> TYPE_TEXT (+ a text row), scroll -> SCROLL_DOWN/UP.
Dropped: say/load/tab/copy actions (the agent has no such operation) and turns whose gold element is not among
WebLINX's ten pre-ranked candidates (about a third; keeping them would teach a forced wrong answer).

recent_actions are rebuilt from the demo's own earlier turns in the agent's shape (element label, kind, text).
Target rows choose among only ~10 pre-ranked candidates (live pages offer 50-250), so only --target-keep of them
are kept; Mind2Web supplies full-viewport target rows.

DONE: the navigator's first message after the demo's last browser action (usually the answer) becomes a DONE
row. Its elements are that turn's own candidates; say turns carry no page HTML, so the page text is the last
observed page's, which keeps DONE rows from being recognisable by short text. Page states after the final action
are only in WebLINX-full (~500 GB).
"""

import argparse
import json
import random
import re
from pathlib import Path

from agent_format import build_request, field_context

CANDIDATE = re.compile(r"\(uid = (?P<uid>[^)]+)\) \[\[tag\]\] (?P<tag>\S+)(?P<rest>.*)")
ATTRIBUTE = re.compile(r"([\w:-]+)='([^']*)'")
BROWSER = {"click", "submit", "text_input", "scroll"}
TEXT_TYPES = {"", "text", "search", "email", "url", "tel", "number", "password"}


def field(rest, name):
    m = re.search(r"\[\[" + name + r"\]\] (.*?)(?= \[\[|$)", rest)
    return m.group(1).strip() if m else ""


def element(line, node):
    m = CANDIDATE.match(line.strip())
    if not m:
        return None
    tag, rest = m.group("tag").lower(), m.group("rest")
    attrs = dict(ATTRIBUTE.findall(field(rest, "attributes")))
    kind = attrs.get("type", "").lower()
    role = attrs.get("role") or {"a": "link", "button": "button", "select": "combobox", "textarea": "textbox"}.get(tag)
    if tag == "input":
        role = role or ("button" if kind in {"submit", "button", "reset", "image"}
                        else kind if kind in {"checkbox", "radio"} else "textbox")
    label = (attrs.get("aria-label") or field(rest, "text") or attrs.get("placeholder") or attrs.get("title")
             or (attrs.get("value") if role == "button" else "") or attrs.get("name") or tag)
    editable = (tag == "input" and kind in TEXT_TYPES) or tag == "textarea" or attrs.get("contenteditable") == "true"
    # WebLINX elides long attribute text with "..."; keep the readable part before it.
    label = " ".join(label.split("...")[0].split())[:200] or tag
    base = {"role": role or tag, "label": label, "value": attrs.get("value", "").split("...")[0], "node": node}
    return m.group("uid"), base, editable


def page_text(clean_html):
    text = re.sub(r'[\w:-]+="[^"]*"', " ", clean_html or "")
    text = re.sub(r"\(\w[\w-]*|\)", " ", text)
    return " ".join(text.split())[:6000]


def scroll_down(action):
    y = re.search(r"y=(-?[\d.]+)", action)
    # ponytail: WebLINX scroll() records a position, not a delta; y > 0 read as "down". Refine if it matters.
    return bool(y and float(y.group(1)) > 0)


def executed(turn):
    """The history entry the agent would have logged for this browser action."""
    action = turn["action"]
    name = action.split("(", 1)[0]
    if name == "scroll":
        return {"action": "Scroll down" if scroll_down(action) else "Scroll up", "kind": "scroll", "text": None,
                "page_changed": None}
    uid, text = re.search(r'uid="([^"]+)"', action), re.search(r'text="([^"]*)"', action)
    parsed = (element(line, 0) for line in (turn["candidates"] or "").split("\n"))
    label = next((p[1]["label"] for p in parsed if p and uid and p[0] == uid.group(1)), name)
    return {"action": label, "kind": "fill" if name == "text_input" else "click",
            "text": text.group(1) if text and name == "text_input" else None, "page_changed": None}


def page_for(candidate_lines, clean_html, url):
    actions, node_of = [], {}
    for node, line in enumerate(candidate_lines):
        parsed = element(line, node)
        if not parsed:
            continue
        cuid, base, editable = parsed
        node_of[cuid] = node
        if editable:
            actions.append({**base, "id": f"e{len(actions) + 1}", "kind": "fill"})
        actions.append({**base, "id": f"e{len(actions) + 1}", "kind": "click"})
    actions += [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
                {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
                {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]
    return {"url": url, "title": "", "text": page_text(clean_html), "actions": actions}, node_of


def clean_goal(utterances):
    return " ".join(re.sub(r"\[\d\d:\d\d\]|\[-\d\d:\d\d\]|;$", " ", utterances or "").split())


def done_row(say, clean_html, url, steps):
    prompt = say["clean_html"] or ""
    goal = re.search(r"utterances are: (.*?);\s*Viewport", prompt, re.S)
    if "top candidates for this turn:" not in prompt or not goal or not steps:
        return None
    candidates = prompt.split("top candidates for this turn:", 1)[1].split("<</SYS>>")[0]
    page, _ = page_for(re.split(r"\s*(?=\(uid = )", candidates.strip()), clean_html, url)
    body, _, _, _ = build_request(page, clean_goal(goal.group(1)), steps)
    return {"state": body["state"], "question": body["questions"]["operation"], "label": "DONE"}


def rows_for_turn(turn, steps, keep_target):
    action = turn["action"] or ""
    name = action.split("(", 1)[0]
    uid = re.search(r'uid="([^"]+)"', action)
    if name not in BROWSER:
        return []
    urls = re.findall(r'load\(url="([^"]+)"', turn["action_history"] or "")
    page, node_of = page_for((turn["candidates"] or "").split("\n"), turn["clean_html"], urls[-1] if urls else "")
    goal = clean_goal(turn["utterances"])
    body, targets, _, _ = build_request(page, goal, steps)
    state, questions = body["state"], body["questions"]

    if name == "scroll":
        operation = "SCROLL_DOWN" if scroll_down(action) else "SCROLL_UP"
        return [{"state": state, "question": questions["operation"], "label": operation}]
    if not uid or uid.group(1) not in node_of:
        return []  # gold element not among the candidates
    operation = "TYPE_TEXT" if name == "text_input" else "CLICK"
    node = node_of[uid.group(1)]
    target = next((k for k, a in targets.get(operation, {}).items() if a["node"] == node), None)
    if target is None:
        return []  # e.g. text typed into an element the snapshot rules do not treat as editable
    rows = [{"state": state, "question": questions["operation"], "label": operation}]
    if keep_target:
        rows.append({"state": state, "question": questions[operation.lower() + "_target"], "label": target})
    if operation == "TYPE_TEXT":
        text = re.search(r'text="([^"]*)"', action)
        fill = targets["TYPE_TEXT"][target]
        if text and text.group(1).strip():
            rows.append({"task": "text", "context": field_context(goal, fill, page, steps), "target": text.group(1)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True, help="WebLINX chat train parquet file(s)")
    p.add_argument("--val", nargs="+", required=True, help="WebLINX chat validation parquet file(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--target-keep", type=float, default=0.3, help="share of train target rows kept (val keeps all)")
    args = p.parse_args()
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    for split, paths in (("train", args.train), ("val", args.val)):
        demos = {}
        for path in paths:
            for turn in pq.read_table(path).to_pylist():
                demos.setdefault(turn["demo"], []).append(turn)
        counts = {"turns": 0, "turns_used": 0, "done_rows": 0}
        with open(out / f"weblinx_{split}.jsonl", "w") as f:
            for demo, turns in demos.items():
                turns.sort(key=lambda t: int(t["turn"]))
                names = [(t["action"] or "").split("(", 1)[0] for t in turns]
                last_browser = max((i for i, n in enumerate(names) if n in BROWSER), default=len(turns))
                steps, html, url, done = [], "", "", False
                for i, (turn, name) in enumerate(zip(turns, names)):
                    rows = []
                    if name in BROWSER:
                        keep = split == "val" or rnd.random() < args.target_keep
                        rows = rows_for_turn(turn, steps[-10:], keep)
                        steps.append(executed(turn))
                    elif name == "say" and not done and i > last_browser and 'speaker="navigator"' in turn["action"]:
                        done = True
                        row = done_row(turn, html, url, steps[-10:])
                        rows = [row] if row else []
                        counts["done_rows"] += bool(row)
                    if turn["clean_html"] and "top candidates for this turn:" not in turn["clean_html"]:
                        html = turn["clean_html"]
                    urls = re.findall(r'load\(url="([^"]+)"', (turn["action_history"] or "") + turn["action"])
                    url = urls[-1] if urls else url
                    counts["turns"] += 1
                    counts["turns_used"] += bool(rows)
                    f.writelines(json.dumps({**row, "host": demo}, ensure_ascii=False) + "\n" for row in rows)
        print(json.dumps({"split": split, **counts}))

if __name__ == "__main__":
    main()
