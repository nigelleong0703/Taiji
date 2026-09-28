"""WebLINX-full (McGill-NLP/WebLINX-full, CC BY-NC-SA 4.0: research use only) -> S1 web rows with screenshots.

The chat version (prepare_weblinx.py) offers ten pre-ranked candidates, many of them non-interactive, and a few
hundred characters of page text. WebLINX-full keeps each browser event's real HTML (elements tagged with
data-webtasks-id), every element's viewport box and a screenshot, so rows are rebuilt the way the agent observes a
page: snapshot.js roles/names for every element centred in the viewport, the visible text, and the screenshot.
HTML parsing and the snapshot rules are shared with prepare_mind2web.py.

  click        -> CLICK on the nearest interactive ancestor of the clicked node (a span inside a button)
  textInput    -> TYPE_TEXT + a text row; consecutive inputs into one field keep the last (full) text
  change       -> SELECT when the value matches an <option>
  scroll       -> SCROLL_DOWN / SCROLL_UP from the change in scrollY
  final report -> DONE on a page saved after the final page-changing action: the last copy/hover before the
                  navigator's report (reading the answer), else the first event after it. Events after the
                  report are not steps towards the goal and make no rows.
Dropped: submit (duplicates the click or Enter that caused it), load/tab/paste events, clicks with no interactive
ancestor within six levels (clicks on page background).

The goal is every instructor message so far; the agent gets one goal, WebLINX instructions arrive over a chat.
Only the files a demo's rows need are downloaded (hub.py: bulk CDN links, several demos at once), then deleted;
converted demos are skipped on rerun.
"""

import argparse
import json
import multiprocessing
import os
import re
import shutil
import socket
from pathlib import Path

import hub
from agent_format import build_request, field_context
from prepare_mind2web import Page, actions_for, role

REPO = "McGill-NLP/WebLINX-full"
LABELED = {"click", "textInput", "change", "scroll"}
MUTATING = {"click", "textInput", "change", "paste", "submit"}


class FullPage(Page):
    """WebLINX HTML -> the Mind2Web attribute names prepare_mind2web's rules read; boxes from bboxes-N.json."""

    def __init__(self, boxes):
        super().__init__()
        self.boxes = boxes

    def handle_starttag(self, tag, attrs):
        a = {(k.replace("-", "_") if k.startswith("aria-") else k): v for k, v in attrs}
        if "selected" in a and tag == "option":
            a["option_selected"] = "true"
        if "checked" in a:
            a["input_checked"] = "true"
        if a.get("value") is not None:
            a["input_value"] = a["value"]
        uid = a.get("data-webtasks-id")
        if uid:
            a["backend_node_id"] = uid
            b = self.boxes.get(uid)
            if b:
                a["bounding_box_rect"] = f"{b['x']},{b['y']},{b['width']},{b['height']}"
        super().handle_starttag(tag, list(a.items()))


def interactive(node):
    for _ in range(6):
        if node is None:
            return None
        if role(node):
            return node
        node = node["parent"]
    return None


def observe(fetch, event, steps, goal):
    """One browser event's page as the agent would have observed it -> (state, targets, body) or None."""
    page_name, meta = event["state"].get("page"), event["action"]["arguments"]["metadata"]
    if not page_name:
        return None
    html, boxes = fetch(f"pages/{page_name}"), fetch(f"bboxes/bboxes-{page_name.split('-')[1]}.json")
    if not html or not boxes:
        return None
    page = FullPage(json.loads(Path(boxes).read_text()))
    page.feed(Path(html).read_text(errors="replace"))
    vw, vh = meta.get("viewportWidth") or 1280, meta.get("viewportHeight") or 720

    def inside(b):
        return b and 0 <= b[0] + b[2] / 2 < vw and 0 <= b[1] + b[3] / 2 < vh

    actions = []
    for node_id, node in page.nodes.items():
        rname = role(node)
        if (not rname or node["hidden"] or node["skip"] or not inside(node["box"]) or "disabled" in node["attrs"]
                or node["attrs"].get("type") in {"password", "file", "hidden"}):
            continue
        actions.extend(actions_for(node, node_id, rname))
    actions = actions[:250]
    actions += [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
                {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
                {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]
    for i, a in enumerate(actions):
        a.setdefault("id", f"e{i + 1}")
    visible = []
    for text, ancestors in page.texts:
        b = next((n["box"] for n in ancestors if n["box"]), None)
        if b and b[1] < vh and b[1] + b[3] > 0 and b[0] < vw and b[0] + b[2] > 0:
            visible.append(text)
    title = re.search(r"<title[^>]*>(.*?)</title>", Path(html).read_text(errors="replace")[:20000], re.S | re.I)
    state = {"url": meta.get("url", ""), "title": " ".join(title.group(1).split()) if title else "",
             "text": "\n".join(visible)[:6000], "actions": actions}
    return state, page


def image_row(fetch, event, images, demo):
    shot = event["state"].get("screenshot")
    if not shot or event["state"].get("screenshot_status") != "good":
        return {}
    local = fetch(f"screenshots/{shot}")
    if not local:
        return {}
    from PIL import Image

    path = images / f"{demo}_{Path(shot).stem}.jpg"
    image = Image.open(local).convert("RGB")
    image.resize((1024, round(1024 * image.height / image.width))).save(path, quality=85)
    return {"image": str(path)}


def goal_before(turns, index):
    said = [t["utterance"] for t in turns[:index] if t["type"] == "chat" and t["speaker"] == "instructor"]
    return " ".join(" ".join(said).split())[:1500]


def rows_for_demo(demo, replay, fetch, images):
    rows, steps, scroll_y = [], [], 0
    turns = replay["data"]
    browser = [i for i, t in enumerate(turns) if t["type"] == "browser"]
    final = max((i for i in browser if turns[i]["action"]["intent"] in MUTATING), default=None)
    report = next((i for i, t in enumerate(turns) if final is not None and i > final and t["type"] == "chat"
                   and t["speaker"] == "navigator"), len(turns))
    for position, i in enumerate(browser):
        if i > report:
            break  # the task is reported done; later events are not steps towards it
        event = turns[i]
        goal = goal_before(turns, i)
        intent, args = event["action"]["intent"], event["action"]["arguments"]
        uid = ((args.get("element") or {}).get("attributes") or {}).get("data-webtasks-id")
        if intent == "textInput" and position + 1 < len(browser):
            following = turns[browser[position + 1]]["action"]
            if following["intent"] == "textInput" and ((following["arguments"].get("element") or {}).get(
                    "attributes") or {}).get("data-webtasks-id") == uid:
                continue  # a later input into the same field carries the full text
        if intent not in LABELED:
            continue
        element = args.get("element") or {}
        label = " ".join((element.get("textContent") or (element.get("attributes") or {}).get("aria-label")
                          or "").split())[:200]
        text = (args.get("text") or "").strip()
        if (element.get("attributes") or {}).get("type") == "password":
            text = ""  # never train on, or show in history, a typed password
        operation = None
        if intent == "scroll":
            y = float(args.get("scrollY") or 0)
            if y != scroll_y:
                operation = "SCROLL_DOWN" if y > scroll_y else "SCROLL_UP"
                label = "Scroll down" if y > scroll_y else "Scroll up"
            scroll_y = y
        observed = observe(fetch, event, steps, goal) if goal and (operation or intent != "scroll") else None
        if observed:
            state, page = observed
            gold = interactive(page.nodes.get(uid)) if uid else None
            gold_id = gold["attrs"]["backend_node_id"] if gold else None
            if gold and gold["text"]:
                label = " ".join(" ".join(gold["text"]).split())[:200]
            if intent == "textInput":
                for a in state["actions"]:
                    if a.get("node") == gold_id:
                        a["value"] = ""  # the page was saved after the keystrokes; the field was empty before them
            body, targets, _, _ = build_request(state, goal, steps[-10:])
            new, key = [], None
            if operation:
                new = [{"state": body["state"], "question": body["questions"]["operation"], "label": operation}]
            elif gold_id:
                operation = {"click": "CLICK", "textInput": "TYPE_TEXT", "change": "SELECT"}[intent]
                wanted = str(args.get("value", "")).strip().lower()
                key = next((k for k, a in targets.get(operation, {}).items() if a["node"] == gold_id and (
                    operation != "SELECT" or wanted in {str(a["value"]).lower(), a["option"].strip().lower()})), None)
                if key and (operation != "TYPE_TEXT" or text):
                    new = [{"state": body["state"], "question": body["questions"]["operation"], "label": operation}]
                    question = body["questions"][operation.lower() + "_target"]
                    if len(question["criteria"]) > 1:
                        new.append({"state": body["state"], "question": question, "label": key})
            if new:
                extra = image_row(fetch, event, images, demo)
                rows += [{**r, **extra, "host": demo} for r in new]
                if operation == "TYPE_TEXT" and len(text) <= 200:  # S1 writes at most 64 tokens
                    context = field_context(goal, targets["TYPE_TEXT"][key], state, steps[-10:])
                    rows.append({"task": "text", "context": context, "target": text, "host": demo})
        if intent != "scroll" or operation:
            steps.append({"action": label or intent, "kind": {"textInput": "fill", "scroll": "scroll",
                                                              "change": "select"}.get(intent, "click"),
                          "text": text[:200] or None, "page_changed": None})

    # DONE: a page saved after the final page-changing action shows its result. Prefer the last copy/hover before
    # the report (the navigator reading the answer), else the first event after it.
    after = [i for i in browser if final is not None and i > final and turns[i]["state"].get("page")]
    reads = [i for i in after if i < report and turns[i]["action"]["intent"] in {"copy", "hover"}]
    pick = reads[-1] if reads else next((i for i in after if i > report), None)
    if pick is not None and report < len(turns):
        goal = goal_before(turns, report)
        observed = observe(fetch, turns[pick], steps, goal)
        if observed and goal:
            body, _, _, _ = build_request(observed[0], goal, steps[-10:])
            rows.append({"state": body["state"], "question": body["questions"]["operation"], "label": "DONE",
                         **image_row(fetch, turns[pick], images, demo), "host": demo})
    return rows


def needed_files(replay):
    files = set()
    for t in replay["data"]:
        if t["type"] == "browser" and t["state"].get("page"):
            page, shot = t["state"].get("page"), t["state"].get("screenshot")
            if page:
                files |= {f"pages/{page}", f"bboxes/bboxes-{page.split('-')[1]}.json"}
            if shot:
                files.add(f"screenshots/{shot}")
    return files


def convert_demo(job):
    """One demo in a worker process; a failure leaves no part file, so the next run retries that demo."""
    try:
        return convert_one(*job)
    except Exception as error:  # noqa: BLE001 - one broken demo must not stop the pool
        print(json.dumps({"demo": job[0], "error": repr(error)[:200]}), flush=True)
        return None


def convert_one(demo, out, images, cache, threads):
    """Download one demo's needed files (3 Hub requests, then CDN), write its rows as <demo>.jsonl, delete the files."""
    part = Path(out) / "weblinx_full_parts" / f"{demo}.jsonl"
    if part.exists():
        return None
    prefix = f"demonstrations/{demo}/"
    entries = {e["path"][len(prefix):]: e for e in hub.tree(REPO, prefix.rstrip("/"))}
    rows = []
    if "replay.json" in entries:
        replay_file = hub.download(REPO, [entries["replay.json"]], cache)[prefix + "replay.json"]
        replay = json.loads(Path(replay_file).read_text())
        files = [f for f in sorted(needed_files(replay)) if f in entries]
        got = hub.download(REPO, [entries[f] for f in files], cache, threads)
        rows = rows_for_demo(demo, replay, lambda f: got.get(prefix + f), Path(images))
    tmp = Path(f"{part}.{socket.gethostname()}-{os.getpid()}.tmp")  # two machines may race on one demo
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rename(part)
    shutil.rmtree(Path(cache) / "demonstrations" / demo, ignore_errors=True)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="data folder: writes weblinx_train.jsonl and weblinx_val.jsonl")
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True, help="scratch folder for the demos being converted")
    p.add_argument("--jobs", type=int, default=8, help="demos converted at once (mostly waiting on downloads)")
    p.add_argument("--threads", type=int, default=8, help="downloads at once per demo")
    p.add_argument("--reverse", action="store_true",
                   help="convert from the last demo backwards: a second machine on the same volume meets the first")
    args = p.parse_args()
    out, images = Path(args.out), Path(args.images)
    parts = out / "weblinx_full_parts"
    parts.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(hub.download(REPO, [{"path": "splits.json"}], args.cache)["splits.json"]).read_text())
    order = (("valid", "val"), ("train", "train")) if args.reverse else (("train", "train"), ("valid", "val"))
    for split, name in order:
        demos = list(reversed(splits[split])) if args.reverse else splits[split]
        jobs = [(d, str(out), str(images), args.cache, args.threads) for d in demos
                if not (parts / f"{d}.jsonl").exists()]
        with multiprocessing.Pool(args.jobs) as pool:
            for done, _ in enumerate(pool.imap_unordered(convert_demo, jobs), 1):
                if done % 20 == 0:
                    print(json.dumps({"split": name, "converted": done, "of": len(jobs)}), flush=True)
        with open(out / f"weblinx_{name}.jsonl", "w") as f:
            for demo in splits[split]:
                if (parts / f"{demo}.jsonl").exists():
                    f.write((parts / f"{demo}.jsonl").read_text())
        print(json.dumps({"split": name, "demos": len(splits[split])}), flush=True)

if __name__ == "__main__":
    main()
