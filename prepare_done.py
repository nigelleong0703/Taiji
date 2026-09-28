"""DONE rows at the end of every WebLINX sub-task, not only the final one (weblinx has ~500 DONE rows otherwise).

WebLINX instructions arrive over a chat: the instructor asks for something, the navigator does it and reports back,
the instructor asks for the next thing. The page the navigator leaves after its report is the result of the goal so
far, so the first observed row after the next instructor message, asked with the goal *before* that message, is a
DONE row: same page, same history, the earlier goal.
A boundary counts only when the navigator acted (a page-changing event) since the previous instructor message and
then reported without asking a question, and the next instructor message does not read like a correction.

Rows come from the converted (and cleaned) weblinx_<split>.jsonl; only each demo's replay.json (the chat) is
downloaded, no pages or screenshots.
python prepare_done.py --data $S1_HOME/data --cache $SCRATCH/done  -> weblinx_done_{train,val}.jsonl
"""

import argparse
import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hub
from prepare_weblinx_full import MUTATING, REPO, goal_before
from s1 import kind, row_at

CORRECTION = re.compile(r"^(no\b|not\b|nope|wrong|that'?s not|actually|sorry|wait\b|undo|go back|instead|oops)")


def boundaries(turns):
    """Instructor turn indices that start a new request after the navigator finished the previous one."""
    found, previous, acted, reported = [], None, False, False
    for j, t in enumerate(turns):
        if t["type"] == "browser" and t["action"]["intent"] in MUTATING:
            acted, reported = True, False
        elif t["type"] == "chat" and t["speaker"] == "navigator" and acted:
            reported = not t["utterance"].strip().endswith("?")
        elif t["type"] == "chat" and t["speaker"] == "instructor":
            if previous is not None and acted and reported and not CORRECTION.match(t["utterance"].strip().lower()):
                found.append(j)
            previous, acted, reported = j, False, False
    return found


def goal_pairs(turns):
    """(goal before, goal after) each sub-task boundary; both goals cut at the length limit are skipped."""
    pairs = [(goal_before(turns, j), goal_before(turns, j + 1)) for j in boundaries(turns)]
    return [(before, after) for before, after in pairs if before != after]


def done_rows(rows, pairs, load):
    """DONE rows for one demo: its operation rows as (goal, ref) in order, its goal_pairs, load(ref) -> row."""
    out = []
    for before, after in pairs:
        ref = next((ref for goal, ref in rows if goal == after), None)
        if ref is not None:
            first = load(ref)
            question = copy.deepcopy(first["question"])
            question["instructions"]["goal"] = before
            out.append({**{k: v for k, v in first.items() if k not in {"question", "label"}},
                        "question": question, "label": "DONE"})
    return out


def demos(path):
    """(demo, [(goal, byte offset)] of its operation rows in order); a demo's rows are contiguous. Offsets, not
    rows: the whole file's rows do not fit a small pod."""
    demo, rows, position = None, [], 0
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line)
            if r.get("host") != demo:
                if rows:
                    yield demo, rows
                demo, rows = r.get("host"), []
            if kind(r) == "operation":
                rows.append((r["question"]["instructions"]["goal"], position))
            position += len(line)
    if rows:
        yield demo, rows


def replay_pairs(demo, cache):
    """The demo's goal_pairs; the replay (every browser event, several MB) is dropped right away."""
    path = Path(cache) / demo / "replay.json"
    if not path.exists():
        response = hub._hub("GET", f"{hub.HUB}/datasets/{REPO}/resolve/main/demonstrations/{demo}/replay.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return goal_pairs(json.loads(path.read_text())["data"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cache", required=True, help="replay.json files are kept here")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    for split in ("train", "val"):
        path = Path(args.data) / f"weblinx_{split}.jsonl"
        todo = list(demos(path))
        with ThreadPoolExecutor(args.threads) as pool:
            pairs = list(pool.map(lambda d: replay_pairs(d[0], args.cache), todo))
        written = 0
        with open(Path(args.data) / f"weblinx_done_{split}.jsonl", "w") as f:
            for (_, rows), demo_pairs in zip(todo, pairs, strict=True):
                new = done_rows(rows, demo_pairs, lambda offset: row_at(path, offset))
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
                written += len(new)
        print(json.dumps({"split": split, "demos": len(todo), "done_rows": written}), flush=True)


if __name__ == "__main__":
    turns = [{"type": "chat", "speaker": "instructor", "utterance": "Open Gmail"},
             {"type": "browser", "action": {"intent": "click"}},
             {"type": "chat", "speaker": "navigator", "utterance": "Done."},
             {"type": "chat", "speaker": "instructor", "utterance": "Now open the first email"},
             {"type": "browser", "action": {"intent": "click"}},
             {"type": "chat", "speaker": "navigator", "utterance": "Which one?"},
             {"type": "chat", "speaker": "instructor", "utterance": "The top one"},
             {"type": "browser", "action": {"intent": "click"}},
             {"type": "chat", "speaker": "navigator", "utterance": "Opened."},
             {"type": "chat", "speaker": "instructor", "utterance": "No, the unread one"}]
    assert boundaries(turns) == [3], boundaries(turns)
    assert goal_pairs(turns) == [("Open Gmail", "Open Gmail Now open the first email")], goal_pairs(turns)
    main()
