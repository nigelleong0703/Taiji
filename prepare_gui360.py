"""GUI-360 (vyokky/GUI-360, MIT) -> S1 desktop rows with screenshots (Word, Excel, PowerPoint on Windows).

Not human-labelled: an automatic agent executed each task and an LLM judge kept the successful trajectories.
Only train/test "success" trajectories are used. Each step lists the screen's UI Automation controls, numbered
like the agent's element table, plus the executed action on one of them.

  click on a control          -> CLICK + target      (left single clicks only)
  type into a control         -> TYPE_TEXT + target + a text row; without a control, the last clicked editable one
  wheel_mouse_input           -> SCROLL_DOWN / SCROLL_UP
  final step (OVERALL_FINISH) -> DONE when it takes no action; one that still acts is that action
Dropped: Office API calls (set_cell_value, select_paragraph, ...), hotkeys ({VK_...}), drags, right/double clicks.

Two passes: the trajectory files are downloaded and converted (rows point at their future screenshot path), then
the image tarball is streamed once and only the needed clean screenshots are written, so the 26 GB archive never
touches disk. Rows whose screenshot did not arrive are left out of the final files.
"""

import argparse
import io
import json
import os
import random
import re
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hub
from agent_format import build_request, field_context

REPO = "vyokky/GUI-360"
ROLES = {"Button": "button", "MenuItem": "menuitem", "TabItem": "tab", "ListItem": "option", "TreeItem": "treeitem",
         "Edit": "textbox", "Document": "textbox", "ComboBox": "combobox", "CheckBox": "checkbox",
         "RadioButton": "radio", "Hyperlink": "link", "DataItem": "gridcell", "Spinner": "spinbutton",
         "ScrollBar": "scrollbar", "SplitButton": "button", "MenuBar": "menubar", "Text": "text"}
EDITABLE = {"Edit", "Document", "ComboBox", "DataItem", "Spinner"}


def actions_for(controls, gold):
    """UIA controls -> agent actions, at most 250 elements; the gold control always stays."""
    if len(controls) > 250:
        # Excel sheets list hundreds of cells (DataItem); keep every other control first, then cells in order.
        keep = {id(c) for c in sorted(controls, key=lambda c: (str(c["label"]) != gold, c["control_type"] ==
                                                                  "DataItem"))[:250]}
        controls = [c for c in controls if id(c) in keep]
    actions = []
    for c in controls:
        base = {"node": str(c["label"]), "role": ROLES.get(c["control_type"], c["control_type"].lower()),
                "label": " ".join((c.get("control_text") or "").split())[:200] or c["control_type"], "value": ""}
        if c["control_type"] in EDITABLE:
            actions += [{**base, "kind": "fill"}, {**base, "kind": "click", "label": "Open " + base["label"]}]
        else:
            actions.append({**base, "kind": "click"})
    actions += [{"kind": "scroll", "id": "scroll_down", "label": "Scroll down", "delta": 560},
                {"kind": "scroll", "id": "scroll_up", "label": "Scroll up", "delta": -560},
                {"kind": "wait", "id": "wait", "label": "Wait for the page to update"}]
    for i, a in enumerate(actions):
        a.setdefault("id", f"e{i + 1}")
    return actions


def rows_for_trajectory(steps, image_of):
    rows, history, last_editable = [], [], None
    for record in sorted(steps, key=lambda r: r["step_id"]):
        step, action = record["step"], record["step"]["action"]
        infos = step.get("control_infos") or {}
        controls = infos.get("merged_controls_info") or infos.get("uia_controls_info") or []
        by_label = {str(c["label"]): c for c in controls}
        function, args, gold = action.get("function"), action.get("args") or {}, str(action.get("control_label") or "")
        text = args.get("text") or args.get("keys") or ""
        operation = target_node = None
        if not controls:
            pass  # no control list for this step: it still enters the history below
        elif step["status"] == "OVERALL_FINISH" and not function:
            operation = "DONE"  # a finishing step that still acts (~35%) shows the screen before its last action
        elif function == "click" and gold in by_label and args.get("button", "left") == "left" \
                and not args.get("double"):
            operation, target_node = "CLICK", gold
        elif function == "type" and text.strip() and "{" not in text:
            target_node = gold if gold in by_label else last_editable
            if target_node in by_label and by_label[target_node]["control_type"] in EDITABLE:
                operation = "TYPE_TEXT"
        elif function == "wheel_mouse_input" and args.get("wheel_dist"):
            operation = "SCROLL_DOWN" if args["wheel_dist"] < 0 else "SCROLL_UP"

        if operation:
            actions = actions_for(controls, target_node or "")
            title = (infos.get("application_windows_info") or {}).get("control_text", "")
            texts = [c.get("control_text", "") for c in controls if c["control_type"] == "Text"]
            state = {"url": "", "title": title, "text": "\n".join(t for t in texts if t)[:6000], "actions": actions}
            body, targets, _, _ = build_request(state, record["request"], history)
            common = {"image": image_of(step["screenshot_clean"]), "host": record["execution_id"]}
            if operation in body["questions"]["operation"]["criteria"]:
                new = [{"state": body["state"], "question": body["questions"]["operation"], "label": operation}]
                key = next((k for k, a in targets.get(operation, {}).items() if a["node"] == target_node), None)
                if target_node and not key:
                    new = []
                elif key:
                    new.append({"state": body["state"], "question": body["questions"][operation.lower() + "_target"],
                                "label": key})
                rows += [{**r, **common} for r in new]
                if new and operation == "TYPE_TEXT":
                    context = field_context(record["request"], targets[operation][key], state, history)
                    rows.append({"task": "text", "context": context, "target": text, "host": common["host"]})

        if function == "click" and gold in by_label and by_label[gold]["control_type"] in EDITABLE:
            last_editable = gold
        label = " ".join((by_label.get(gold, {}).get("control_text") or action.get("control_test") or "").split())
        kind = {"click": "click", "type": "fill", "wheel_mouse_input": "scroll"}.get(function, function or "done")
        history = (history + [{"action": label[:200] or function, "kind": kind,
                               "text": text if function == "type" else None, "page_changed": None}])[-10:]
    return rows


def chosen_trajectories(split, count, seed=0):
    """The same `count` success trajectories on every run: tree entries by path, and the chosen paths."""
    entries = {e["path"]: e for e in hub.tree(REPO, f"{split}/data")
               if e["path"].endswith(".jsonl") and "/success/" in e["path"]}
    files = sorted(entries)
    random.Random(seed).shuffle(files)
    return entries, files[:count]


def convert(split, count, out, images, cache, seed):
    entries, chosen = chosen_trajectories(split, count, seed)
    needed = {}
    # Bulk CDN links (hub.py) in chunks of 200 files, so the scratch disk holds a few hundred MB at most.
    with open(out / f"gui360_{split}.pending.jsonl", "w") as f:
        for path, local in ((p, lp) for i in range(0, len(chosen), 200)
                            for p, lp in hub.download(REPO, [entries[q] for q in chosen[i:i + 200]], cache).items()):
            if not local:
                continue
            app, category = path.split("/")[2:4]

            def image_of(clean, folder=f"image/{app}/{category}/"):  # clean: "success/<id>/action_step3.png"
                needed[folder + clean] = images / (clean.split("/", 1)[1].replace("/", "_").rsplit(".", 1)[0] + ".jpg")
                return str(needed[folder + clean])

            steps = [json.loads(line) for line in Path(local).read_text().splitlines()]
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_for_trajectory(steps, image_of))
            Path(local).unlink()
    return needed


def save_jpeg(data, path):
    from PIL import Image

    Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=85)


def stream_images(split, want, expected=None):
    """Read the tarball once as a stream; `want(member name)` gives the destination of a needed screenshot (or None).
    Needed screenshots are re-encoded on every core while the stream keeps reading; stops early after `expected`."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{split}/image.tar.gz"
    saved = 0
    with urllib.request.urlopen(url) as response, ProcessPoolExecutor(os.cpu_count()) as pool, \
            tarfile.open(fileobj=io.BufferedReader(response, 16 << 20), mode="r|gz") as tar:
        pending = []
        for member in tar:
            path = want(member.name)
            if path and not path.exists():
                pending.append(pool.submit(save_jpeg, tar.extractfile(member).read(), path))
                if len(pending) > 256:  # bound the bytes waiting in memory
                    pending.pop(0).result()
            saved += bool(path)
            if expected is not None and saved >= expected:
                break
        for future in pending:
            future.result()
    return saved


def clean_screenshots(images, chosen):
    """Chooser for every clean screenshot of the chosen trajectories, known before any trajectory is converted."""
    folders = {}
    for path in chosen:  # <split>/data/<app>/<category>/success/<id>.jsonl
        app, category, _, name = path.split("/")[2:6]
        folders[f"image/{app}/{category}/success/{name[:-len('.jsonl')]}"] = name[:-len(".jsonl")]

    def want(member):
        folder, _, file = member.rpartition("/")
        if folder in folders and re.fullmatch(r"action_step\d+\.png", file):
            return images / f"{folders[folder]}_{file[:-len('.png')]}.jpg"
        return None
    return want


def finalize(out, split, name):
    """Keep the pending rows whose screenshot arrived -> gui360_<name>.jsonl."""
    pending, kept = out / f"gui360_{split}.pending.jsonl", 0
    with open(out / f"gui360_{name}.jsonl", "w") as f, open(pending) as rows:
        for line in rows:
            row = json.loads(line)
            if "image" not in row or Path(row["image"]).exists():
                f.write(line)
                kept += 1
    pending.unlink()
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="data folder: writes gui360_train.jsonl and gui360_val.jsonl")
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True, help="scratch folder for trajectory files")
    p.add_argument("--train-trajectories", type=int, default=2000, help="of 13,750")
    p.add_argument("--val-trajectories", type=int, default=200, help="from the test split, of 3,439")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--images-only", action="store_true",
                   help="only stream the chosen trajectories' screenshots (can run beside a conversion)")
    p.add_argument("--finalize", action="store_true", help="only filter an existing pending file")
    args = p.parse_args()
    out, images = Path(args.out), Path(args.images)
    images.mkdir(parents=True, exist_ok=True)
    for split, name, count in (("train", "train", args.train_trajectories), ("test", "val", args.val_trajectories)):
        if split not in args.splits.split(","):
            continue
        if args.images_only:
            saved = stream_images(split, clean_screenshots(images, chosen_trajectories(split, count)[1]))
            print(json.dumps({"split": name, "screenshots": saved}), flush=True)
            continue
        if not args.finalize:
            needed = convert(split, count, out, images, args.cache, seed=0)
            stream_images(split, needed.get, expected=len(needed))
        print(json.dumps({"split": name, "trajectories": count, "rows": finalize(out, split, name)}), flush=True)

if __name__ == "__main__":
    main()
