"""Synthetic WAIT, BLOCKED and SELECT rows: public datasets record no WAIT/BLOCKED moments and few native dropdowns.

Each row starts from a real web operation row (its goal, history and some of the page's controls) and is rebuilt with
the agent's own request builder, so the question and options match live requests:
  WAIT     the page right after a submit-like click, still loading: the content is replaced by a loading notice.
  BLOCKED  a page no supported operation can pass: a bot check, a sign-in wall for a goal that gives no credentials,
           an access-denied or region-blocked page.
  SELECT   a real page with a native <select>: the goal names one of its unselected options (and the field, when
           the field has a short name of its own), the operation and select_target rows follow. The page keeps every
           control, so CLICK and TYPE_TEXT stay offered next to SELECT.
CLICK stays offered on most rows and WAIT pages keep up to 40 controls, so the text, not the option list, decides.
Labels follow the agent's rules, not recorded behaviour: keep their share small and report them as their own
source ("synthetic") in evaluation.

python prepare_synthetic.py --rows weblinx_train.jsonl mind2web_train.jsonl --out synthetic_train.jsonl --per-label 1500
"""

import argparse
import json
import random
from urllib.parse import urlparse

from agent_format import build_request

KIND = {"CLICK": "click", "TYPE_TEXT": "fill", "SELECT": "select"}
SUBMITS = ["Search", "Submit", "Find flights", "Search hotels", "Continue", "Apply filters", "Go", "Next",
           "Show results", "Sign in", "Book now", "Check availability", "Update"]
LOADING = ["Loading…", "Loading results…", "Searching for the best options…",
           "Please wait while we process your request.", "Fetching results", "One moment…", "Updating results",
           "Loading, please wait", "Searching…", "We're finding the best matches for you", "Processing…",
           "Retrieving data"]
BLOCKS = {
    "bot_check": ("Just a moment...", "Checking if the site connection is secure\nVerify you are human by completing "
                  "the action below.\n{host} needs to review the security of your connection before proceeding.\n"
                  "Ray ID: {ray}"),
    "access_denied": ("Access Denied", "Access Denied\nYou don't have permission to access this page on this server.\n"
                      "Reference #{ray}"),
    "region": ("Not available", "This content is not available in your country or region.\nWe're sorry for the "
               "inconvenience."),
    "robot": ("Are you a robot?", "Please confirm that you are not a robot.\nPress & Hold\nReference ID {ray}"),
    "sign_in": ("Sign in", "Sign in to continue to {host}\nEnter your email and password to access your account.\n"
                "Forgot password?\nNew here? Create an account"),
}


def actions_from(elements):
    """The state's element table back into agent actions (inverse of agent_format.action_space)."""
    actions = []
    for e in elements:
        base = {k: e[k] for k in ("role", "value", "checked", "selected", "expanded") if k in e}
        base.update(node=e["index"], label=e["label"])
        for operation in e["operations"]:
            if operation == "SELECT":
                actions += [{**base, "kind": "select", "value": o["value"], "current_value": e.get("value", ""),
                             "label": o["label"]} for o in e.get("options", [])]
            else:
                actions.append({**base, "kind": KIND[operation]})
    return actions


def request(goal, history, url, title, text, actions, scroll=False):
    # Real rows usually offer SCROLL_DOWN; offering it here too keeps the option list from giving the label away.
    scrolls = [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}] if scroll else []
    actions = actions + scrolls + [{"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]
    for i, a in enumerate(actions):
        a.setdefault("id", f"e{i + 1}")
    body, targets, _, _ = build_request({"url": url, "title": title, "text": text, "actions": actions}, goal, history)
    return body, targets


def row(goal, history, url, title, text, actions, label, scroll=False):
    body, _ = request(goal, history, url, title, text, actions, scroll)
    return {"state": body["state"], "question": body["questions"]["operation"], "label": label, "source": "synthetic"}


def option_text(option):
    return option["label"].rsplit(" → ", 1)[-1].strip()


def placeholder(option):
    text = option_text(option).lower()
    return not text or str(option["value"]).strip() in {"", "-1", "0"} or text.startswith(("select", "choose", "-"))


def selectable(state):
    """Native dropdowns with an unselected, non-placeholder option."""
    return [(e, o) for e in state["elements"] if "SELECT" in e["operations"] for o in e.get("options", [])
            if not placeholder(o) and str(o["value"]) != str(e.get("value", "")) and option_text(o) != e.get("value")]


def synthesize_select(real, rnd):
    state = real["state"]
    element, option = rnd.choice(selectable(state))
    choice, field = option_text(option), " ".join(element["label"].split())
    named = 0 < len(field) <= 40 and sum(option_text(o) in field for o in element["options"]) < 2
    goal = rnd.choice([f'Set "{field}" to "{choice}".', f'Choose "{choice}" for {field}.',
                       f"Select {choice} as the {field}."] if named else
                      [f'Choose "{choice}" from the dropdown.', f"Select the {choice} option.",
                       f'Pick "{choice}" in the drop-down list.'])
    page = state["page"]
    body, targets = request(goal, [dict(h) for h in state["recent_actions"]], page.get("url", ""),
                            page.get("title", ""), page.get("text", ""), actions_from(state["elements"]),
                            scroll=rnd.random() < 0.5)
    key = next(k for k, a in targets["SELECT"].items() if a["node"] == element["index"] and
               str(a["value"]) == str(option["value"]))
    common = {"state": body["state"], "source": "synthetic_select"}
    rows = [{**common, "question": body["questions"]["operation"], "label": "SELECT"}]
    if len(targets["SELECT"]) > 1:
        rows.append({**common, "question": body["questions"]["select_target"], "label": key})
    return rows


def synthesize(real, kind, rnd):
    state, goal = real["state"], real["question"]["instructions"]["goal"]
    # WAIT keeps 2-40 of the page's controls (filters and navigation stay while results load), so few elements is
    # no shortcut to WAIT; BLOCKED pages trim this further below.
    kept = [e for e in state["elements"][: rnd.randint(2, 40)] if "SELECT" not in e["operations"]]
    actions = [a for a in actions_from(kept) if a["kind"] == "click"]
    page = state["page"]
    host = urlparse(page.get("url") or "").hostname or rnd.choice(["www.example.com", "shop.example.org"])
    history = [dict(h) for h in state["recent_actions"]]
    ray = "".join(rnd.choice("0123456789abcdef") for _ in range(16))
    if kind == "WAIT":
        submit = {"action": rnd.choice(SUBMITS), "kind": "click", "text": None, "page_changed": None}
        history = (history + [submit])[-10:]
        lines = [line for line in (page.get("text") or "").split("\n")[:3] if len(line) < 80]
        text = "\n".join(lines[: rnd.randint(0, len(lines))] + [rnd.choice(LOADING)])
        return row(goal, history, page.get("url", ""), page.get("title", ""), text, actions, "WAIT",
                   scroll=rnd.random() < 0.3)
    # A blocking page keeps at most two plain links or buttons (logo, help), never the site's own search or forms.
    actions = [a for a in actions if a.get("role") in {"link", "button", "menuitem"}][: rnd.randint(0, 2)]
    block = rnd.choice(sorted(BLOCKS))
    if block == "sign_in" and any(w in goal.lower() for w in ("@", "password", "passcode")):
        block = "access_denied"  # the goal may carry credentials; a sign-in wall would not block it
    title, text = BLOCKS[block]
    if block == "sign_in":  # password fields are never offered (snapshot.js skips them); the email field is
        actions = actions + [{"node": "email", "role": "textbox", "label": "Email", "value": "", "kind": "fill"},
                             {"node": "email", "role": "textbox", "label": "Email", "value": "", "kind": "click"},
                             {"node": "signin", "role": "button", "label": "Sign in", "value": "", "kind": "click"}]
    return row(goal, history, page.get("url", ""), title, text.format(host=host, ray=ray), actions, "BLOCKED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", required=True, help="web JSONL files (their operation rows are the bases)")
    p.add_argument("--out", required=True)
    p.add_argument("--per-label", type=int, default=1500)
    p.add_argument("--kinds", default="WAIT,BLOCKED", help="comma list of WAIT, BLOCKED, SELECT")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rnd = random.Random(args.seed)
    bases = []
    for path in args.rows:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "DONE" in (r.get("question") or {}).get("criteria", {}) and r["state"].get("elements"):
                    bases.append(r)
    rnd.shuffle(bases)
    selects = [r for r in bases if selectable(r["state"])]
    kinds, written = args.kinds.split(","), 0
    with open(args.out, "w") as f:
        for i in range(len(kinds) * args.per_label):
            kind = kinds[i % len(kinds)]
            if kind == "SELECT":
                rows = synthesize_select(selects[i // len(kinds) % len(selects)], rnd)
            else:
                rows = [synthesize(bases[i % len(bases)], kind, rnd)]
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            written += len(rows)
    print(json.dumps({"bases": len(bases), "select_bases": len(selects), "rows": written, "out": args.out}))


if __name__ == "__main__":
    main()
