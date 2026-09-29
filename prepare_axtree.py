"""Released web-agent trajectories with accessibility trees -> S1 rows, in the harness's own request format.

  nnetnav    stanfordnlp/nnetnav-live (Apache-2.0): live open-web steps; actions click [id], type [id] [text] [enter],
             scroll [down|up], stop [answer]
  agenttrek  xlangai/AgentTrek (license not stated on the hub; research use): tutorial-guided live-web steps;
             actions click('id'), fill('id', 'text'), scroll(x, y), send_msg_to_user(...), report_infeasible(...)

Each step's tree becomes harness actions (fill + "Open" click for editable fields, click for the rest, scroll/wait
controls), then agent_format.build_request builds the exact request the agent sends. Rows: the operation, the target
(when there is a choice) and the typed text (field_context). stop / send_msg_to_user -> DONE; stop [N/A] /
report_infeasible -> BLOCKED. hover, key presses, goto, tabs and dropdown options have no harness equivalent: skipped.
Text only (neither set ships screenshots). Split by site, so no site is on both sides.

python prepare_axtree.py --source nnetnav --out $S1_HOME/data [--limit N]
"""

import argparse
import hashlib
import json
import re
from urllib.parse import urlparse

import pyarrow.parquet as pq
from agent_format import build_request, field_context
from huggingface_hub import HfApi, hf_hub_download

REPOS = {"nnetnav": "stanfordnlp/nnetnav-live", "agenttrek": "xlangai/AgentTrek"}
LINE = re.compile(r"^(\s*)\[([\w-]+)\]\s+(\w+)\s+'((?:[^'\\]|\\.)*)'(.*)$")
TEXT = re.compile(r"^\s*StaticText\s+'((?:[^'\\]|\\.)*)'")
EDITABLE = {"textbox", "searchbox", "combobox"}
CLICKABLE = {"link", "button", "checkbox", "radio", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "tab",
             "switch", "treeitem", "gridcell"}
CONTROLS = [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]


def page_from_tree(tree, url, visible_only):
    """Accessibility tree -> the harness's observed page (url, title, text, actions)."""
    title, words, actions = "", [], []
    for line in tree.splitlines():
        if not title and line.strip().startswith("RootWebArea"):
            title = re.findall(r"'((?:[^'\\]|\\.)*)'", line)[:1] or [""]
            title = title[0]
        if m := TEXT.match(line):
            words.append(m.group(1))
            continue
        m = LINE.match(line)
        if not m:
            continue
        _, node, role, name, rest = m.groups()
        if visible_only and "visible" not in rest:
            continue
        editable = role in EDITABLE and "readonly" not in rest
        if not (editable or role in CLICKABLE or "clickable" in rest):
            continue
        value = (re.findall(r"value='((?:[^'\\]|\\.)*)'", rest)[:1] or [""])[0]
        label = " ".join(name.split())[:200] or role
        base = {"node": node, "role": role, "label": label, "value": value}
        for key in ("checked", "expanded", "selected"):
            if f"{key}=" in rest:
                base[key] = "true" if f"{key}=True" in rest or f"{key}='true'" in rest else "false"
        actions.append({**base, "kind": "fill" if editable else "click"})
        if editable:
            actions.append({**base, "kind": "click", "label": "Open " + label})
    actions = actions[:250]
    for i, a in enumerate(actions):
        a["id"] = f"e{i + 1}"
    return {"url": url, "title": title, "text": "\n".join(words)[:6000], "actions": actions + CONTROLS}


def parse_nnetnav(row):
    user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    tree = user.split("OBSERVATION:", 1)[-1].split("\nURL:", 1)[0]
    url = (re.findall(r"^URL: (.+)$", user, re.M) or [""])[-1].strip()
    goal = (re.findall(r"^OBJECTIVE: (.+)$", user, re.M) or [""])[-1].strip()
    previous = user.split("PREVIOUS ACTIONS:", 1)[-1].strip().splitlines()
    history = [re.sub(r"^\d+:\s*", "", p).strip() for p in previous if p.strip() and not p.strip().endswith("None")]
    found = re.findall(r"```(.*?)```", row["output"] or "", re.S)
    action = found[-1].strip() if found else ""
    if m := re.match(r"click \[([\w-]+)\]", action):
        step = ("CLICK", m.group(1), None)
    elif m := re.match(r"type \[([\w-]+)\] \[(.*?)\]", action, re.S):
        step = ("TYPE_TEXT", m.group(1), m.group(2).strip())
    elif m := re.match(r"scroll \[(down|up)\]", action):
        step = ("SCROLL_" + m.group(1).upper(), None, None)
    elif m := re.match(r"stop \[(.*)\]", action, re.S):
        step = ("BLOCKED" if m.group(1).strip().upper() in {"N/A", "NA", ""} else "DONE", None, None)
    else:
        step = None
    return goal, tree, url, history, step, True


def parse_agenttrek(row):
    messages = row["messages"]
    user = [m for m in messages if m["role"] == "user"][-1]["content"]
    goal = user.split("## Goal:", 1)[-1].split("# Observation", 1)[0].strip()
    tree = user.split("## AXTree:", 1)[-1].split("# History of interaction", 1)[0]
    history = [a.strip() for a in re.findall(r"### Action:\s*\n(.+)", user)]
    answer = [m for m in messages if m["role"] == "assistant"][-1]["content"]
    found = re.findall(r"<action>\s*(.*?)\s*</action>", answer, re.S)
    action = found[-1].strip() if found else ""
    if m := re.match(r"click\(\s*['\"]([\w-]+)['\"]", action):
        step = ("CLICK", m.group(1), None)
    elif m := re.match(r"fill\(\s*['\"]([\w-]+)['\"]\s*,\s*(['\"])(.*)\2\s*\)$", action, re.S):
        step = ("TYPE_TEXT", m.group(1), m.group(3).strip())
    elif m := re.match(r"scroll\(\s*[-\d.]+\s*,\s*([-\d.]+)", action):
        step = ("SCROLL_DOWN" if float(m.group(1)) > 0 else "SCROLL_UP", None, None)
    elif action.startswith("send_msg_to_user("):
        step = ("DONE", None, None)
    elif action.startswith("report_infeasible("):
        step = ("BLOCKED", None, None)
    else:
        step = None
    urls = re.findall(r"url='(https?://[^']+)'", tree)
    return goal, tree, urls[0] if urls else "", history, step, False


def rows_for(source, row):
    goal, tree, url, history, step, visible_only = (parse_nnetnav if source == "nnetnav" else parse_agenttrek)(row)
    if not goal or not step:
        return []
    operation, node, text = step
    page = page_from_tree(tree, url, visible_only)
    steps = [{"action": h[:200], "kind": "click", "text": None, "page_changed": None} for h in history[-10:]]
    body, targets, _, _ = build_request(page, goal, steps)
    host = urlparse(url).hostname or ("site-" + hashlib.sha1(page["title"].encode()).hexdigest()[:8])
    common = {"state": body["state"], "source": source, "host": host}
    if operation not in body["questions"]["operation"]["criteria"]:
        return []
    rows = [{**common, "question": body["questions"]["operation"], "label": operation}]
    if node is not None:
        kind = "fill" if operation == "TYPE_TEXT" else "click"
        key = next((k for k, a in targets.get(operation, {}).items() if a["node"] == node and a["kind"] == kind), None)
        if key is None:
            return []  # the gold element is not a harness target (hidden, unnamed or past the 250 cap)
        question = body["questions"][operation.lower() + "_target"]
        if len(question["criteria"]) > 1:
            rows.append({**common, "question": question, "label": key})
        if operation == "TYPE_TEXT":
            if not text or len(text) > 200:
                return []
            rows.append({"task": "text", "context": field_context(goal, targets["TYPE_TEXT"][key], page, steps),
                         "target": text, "source": source, "host": host})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=sorted(REPOS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="stop after this many input steps (a quick look)")
    p.add_argument("--val-share", type=float, default=0.05)
    args = p.parse_args()
    repo = REPOS[args.source]
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
             if f.endswith(".parquet") and "/train/" in f]
    counts, seen = {"steps": 0, "train": 0, "val": 0, "labels": {}}, 0
    with open(f"{args.out}/{args.source}_train.jsonl", "w") as train, open(f"{args.out}/{args.source}_val.jsonl", "w") as val:
        for name in files:
            path = hf_hub_download(repo, name, repo_type="dataset", revision="refs/convert/parquet")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    seen += 1
                    for r in rows_for(args.source, row):
                        split = "val" if int(hashlib.sha1(r["host"].encode()).hexdigest(), 16) % 1000 < args.val_share * 1000 else "train"
                        (val if split == "val" else train).write(json.dumps(r, ensure_ascii=False) + "\n")
                        counts[split] += 1
                        label = "text" if r.get("task") == "text" else r["label"] if r["question"]["criteria"].get("DONE") else "target"
                        counts["labels"][label] = counts["labels"].get(label, 0) + 1
                    if args.limit and seen >= args.limit:
                        break
                if args.limit and seen >= args.limit:
                    break
            if args.limit and seen >= args.limit:
                break
    counts["steps"] = seen
    print(json.dumps({"source": args.source, **counts}), flush=True)


if __name__ == "__main__":
    main()
