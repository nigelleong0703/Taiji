"""Multimodal-Mind2Web (osunlp/Multimodal-Mind2Web, OpenRAIL: research use) -> S1 web rows with screenshots.

Human demonstrations on real sites. Each action row has the raw HTML (every element carries its page box) and a
full-page screenshot. The agent sees one viewport, so each row is cut to a 1280x720 viewport holding the gold
element at a random height; the element table is rebuilt from the raw HTML with the agent's snapshot.js rules
(role, name, editable, select options) for elements centred in that viewport, and the page text is the visible
text in it. Questions come from agent_format.build_request, as for WebLINX.

Kept: CLICK, TYPE (+ a text row), SELECT when the value matches an unselected <option>.
Dropped: rows without a screenshot, gold elements without a box, TYPE into elements snapshot.js does not treat
as editable.
Splits: train, test_task and test_domain -> train; test_website (unseen sites) -> val.
Parquet shards are downloaded one at a time and deleted after conversion; converted shards are skipped on rerun.
"""

import argparse
import io
import json
import random
from html.parser import HTMLParser
from pathlib import Path

from agent_format import build_request, field_context

VW, VH = 1280, 720
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SKIP = {"script", "style", "noscript", "template"}
ROLES = {"button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemradio", "option", "gridcell",
         "combobox", "textbox", "searchbox", "spinbutton"}
OPS = {"CLICK": "click", "TYPE": "fill", "SELECT": "select"}


def box(value):
    try:
        x, y, w, h = map(float, (value or "").split(","))
    except ValueError:
        return None
    return (x, y, w, h) if w > 0 and h > 0 else None


class Page(HTMLParser):
    """Mind2Web raw HTML -> nodes by backend_node_id (tag, attrs, box, text, options) and visible text runs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes, self.stack, self.texts = {}, [], []

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        parent = self.stack[-1] if self.stack else None
        node = {"tag": tag, "attrs": a, "box": box(a.get("bounding_box_rect")), "text": [], "len": 0, "parent": parent,
                "options": [], "hidden": (parent or {}).get("hidden", False) or a.get("aria_hidden") == "true",
                "skip": (parent or {}).get("skip", False) or tag in SKIP}
        if a.get("backend_node_id"):
            self.nodes[a["backend_node_id"]] = node
        if tag == "option":
            select = next((n for n in reversed(self.stack) if n["tag"] == "select"), None)
            if select:
                select["options"].append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        text = " ".join(data.split())
        if not text or not self.stack or self.stack[-1]["skip"]:
            return
        for node in self.stack:
            if node["len"] < 200:
                node["text"].append(text)
                node["len"] += len(text) + 1
        if not self.stack[-1]["hidden"]:
            self.texts.append((text, self.stack[-6:][::-1]))  # boxes are resolved after the candidate fill-in


def role(node):
    tag, a = node["tag"], node["attrs"]
    if a.get("role") in ROLES:
        return a["role"]
    kind = a.get("type", "").lower()
    if tag in {"button", "summary"}:
        return "button"
    if tag == "a":
        return "link"  # Mind2Web drops href; every <a> is kept (snapshot.js needs a[href])
    if tag == "select":
        return "combobox"
    if tag == "textarea" or a.get("contenteditable") == "true":
        return "textbox"
    if tag == "input":
        if kind in {"checkbox", "radio"}:
            return kind
        if kind in {"button", "submit", "reset", "image"}:
            return "button"
        return {"search": "searchbox", "number": "spinbutton"}.get(kind, "textbox" if kind in
                                                                   {"", "text", "email", "url", "tel"} else None)
    return None


def name(node):
    # ponytail: no aria-labelledby / <label for> resolution (snapshot.js has both); ids are not kept here.
    a, text = node["attrs"], " ".join(node["text"])[:200]
    return (a.get("aria_label") or (a.get("input_value") or a.get("value") if a.get("type") in
            {"button", "submit", "reset"} else "") or a.get("alt") or (text if node["tag"] != "input" else "")
            or a.get("title") or a.get("placeholder") or "")


def actions_for(node, node_id, rname):
    a = node["attrs"]
    base = {"node": node_id, "role": rname, "label": name(node) or rname}
    if rname in {"checkbox", "radio"}:
        base["checked"] = a.get("input_checked", "false")
    for key in ("checked", "selected", "expanded"):
        if a.get("aria_" + key):
            base[key] = a["aria_" + key]
    if node["tag"] == "select":
        options = [(o, " ".join(o["text"]).strip()) for o in node["options"]]
        current = ", ".join(t for o, t in options if o["attrs"].get("option_selected") == "true")
        return [{**base, "kind": "select", "value": o["attrs"].get("value", t), "option": t, "current_value": current,
                 "label": f"{base['label']} → {t}"} for o, t in options if o["attrs"].get("option_selected") != "true"]
    editable = a.get("readonly") is None and (rname in {"textbox", "searchbox", "spinbutton"} or
                                              (rname == "combobox" and node["tag"] in {"input", "textarea"}))
    value = a.get("input_value", a.get("value", "")) if node["tag"] in {"input", "textarea"} else ""
    if editable:
        return [{**base, "kind": "fill", "value": value}, {**base, "kind": "click", "value": value,
                                                          "label": "Open " + base["label"]}]
    return [{**base, "kind": "click", "value": value}]


def history(reprs):
    steps = []
    for r in reprs:
        element, _, op = r.partition(" -> ")
        op, _, value = op.partition(": ")
        steps.append({"action": element.split("]", 1)[-1].strip(), "kind": OPS.get(op.strip(), "click"),
                      "text": value or None, "page_changed": None})
    return steps[-10:]


def rows_for(record, images):
    from PIL import Image

    operation = json.loads(record["operation"])
    op, value = operation["op"], operation.get("value", "")
    if op not in OPS or not record["pos_candidates"] or not (record["screenshot"] or {}).get("bytes"):
        return []
    gold_id = json.loads(record["pos_candidates"][0])["backend_node_id"]
    page = Page()
    page.feed(record["raw_html"])
    # Some rows' raw HTML has no boxes; the candidate lists (gold + ~500 others) carry them.
    for candidate in map(json.loads, record["pos_candidates"] + record["neg_candidates"]):
        node = page.nodes.get(candidate["backend_node_id"])
        if node and not node["box"]:
            node["box"] = box(json.loads(candidate["attributes"]).get("bounding_box_rect"))
    gold = page.nodes.get(gold_id)
    if not gold or not gold["box"]:
        return []
    shot = Image.open(io.BytesIO(record["screenshot"]["bytes"])).convert("RGB")
    scale = shot.width / VW
    height = shot.height / scale
    rnd = random.Random(record["action_uid"])
    _, gy, _, gh = gold["box"]
    top = max(0.0, min(gy + gh / 2 - rnd.uniform(0.15, 0.85) * VH, height - VH))

    def inside(b):
        return b and 0 <= b[0] + b[2] / 2 < VW and top <= b[1] + b[3] / 2 < top + VH

    actions = []
    for node_id, node in page.nodes.items():
        rname = role(node)
        if node is gold:
            rname = rname or node["attrs"].get("role") or node["tag"]  # the human clicked it; keep it offered
        elif (not rname or node["hidden"] or node["skip"] or not inside(node["box"])
              or "disabled" in node["attrs"] or node["attrs"].get("type") in {"password", "file", "hidden"}):
            continue
        actions.extend(actions_for(node, node_id, rname))
    actions = actions[:250] if any(a["node"] == gold_id for a in actions[:250]) else None
    if not actions:
        return []
    if top + VH < height - 2:
        actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560})
    if top > 0:
        actions.append({"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560})
    actions.append({"id": "wait", "kind": "wait", "label": "Wait for the page to update"})
    for i, a in enumerate(actions):
        a.setdefault("id", f"e{i + 1}")
    visible = []
    for t, ancestors in page.texts:
        b = next((n["box"] for n in ancestors if n["box"]), None)
        if b and b[1] < top + VH and b[1] + b[3] > top and b[0] < VW:
            visible.append(t)
    text = "\n".join(visible)[:6000]
    state = {"url": "", "title": "", "text": text, "actions": actions}
    goal, steps = record["confirmed_task"], history(record["action_reprs"][: int(record["target_action_index"])])
    body, targets, _, _ = build_request(state, goal, steps)

    operation_name = {"CLICK": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}[op]
    wanted = value.strip().lower()
    target = next((k for k, a in targets.get(operation_name, {}).items() if a["node"] == gold_id and
                   (op != "SELECT" or a["option"].strip().lower() == wanted)), None)
    if target is None:
        return []
    path = images / f"{record['action_uid']}.jpg"
    crop = shot.crop((0, round(top * scale), shot.width, round((top + VH) * scale)))
    crop.resize((1024, round(1024 * crop.height / crop.width))).save(path, quality=85)
    common = {"image": str(path), "host": record["website"]}
    rows = [{"state": body["state"], "question": body["questions"]["operation"], "label": operation_name, **common},
            {"state": body["state"], "question": body["questions"][operation_name.lower() + "_target"],
             "label": target, **common}]
    if len(rows[1]["question"]["criteria"]) < 2:
        rows.pop(1)  # a one-option target teaches nothing
    if op == "TYPE" and value.strip():
        rows.append({"task": "text", "context": field_context(goal, targets["TYPE_TEXT"][target], state, steps),
                     "target": value, "host": record["website"]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="data folder: writes mind2web_train.jsonl and mind2web_val.jsonl")
    p.add_argument("--images", required=True, help="folder for the cropped viewport screenshots")
    p.add_argument("--cache", required=True, help="scratch folder for one parquet shard at a time")
    args = p.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out, images = Path(args.out), Path(args.images)
    parts = out / "mind2web_parts"
    parts.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    repo = "osunlp/Multimodal-Mind2Web"
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for shard in shards:
        part = parts / (Path(shard).stem + ".jsonl")
        if part.exists():
            continue
        local = hf_hub_download(repo, shard, repo_type="dataset", local_dir=args.cache)
        actions = kept = 0
        with open(str(part) + ".tmp", "w") as f:
            for batch in pq.ParquetFile(local).iter_batches(batch_size=8):
                for record in batch.to_pylist():
                    actions += 1
                    rows = rows_for(record, images)
                    kept += bool(rows)
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        Path(str(part) + ".tmp").rename(part)
        Path(local).unlink()
        print(json.dumps({"shard": shard, "actions": actions, "actions_used": kept}), flush=True)
    for split, prefix in (("train", ("train", "test_task", "test_domain")), ("val", ("test_website",))):
        with open(out / f"mind2web_{split}.jsonl", "w") as f:
            for part in sorted(parts.glob("*.jsonl")):
                if part.name.startswith(prefix):
                    f.write(part.read_text())


if __name__ == "__main__":
    main()
