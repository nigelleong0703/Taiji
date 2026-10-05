"""Teacher-labelled S1 decision rows, with injected failed attempts for recovery.

Input: states from `tools/extract-states.ts`, for any environment (browser, desktop, ...)
and for the general tool path. Each state is written as recorded plus variants where one
action was already attempted k times without effect (environments: page unchanged; tools:
the call failed). S2 (the teacher) labels every variant. Environment rows compile through
`env_decision.prepare` and tool rows reuse the recorded rank request, so `state` and
`question` are exactly what the served S1 receives, in the `s1.encode` row format
({state, question, label}).

  python tools/recovery_dataset.py states.jsonl rows.jsonl --inject 1 3 --votes 3
  python tools/recovery_dataset.py states.jsonl - --limit 1 --dry-run   # print one teacher prompt
"""
import argparse
import collections
import concurrent.futures
import importlib.util
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src" / "env_decision.py"
spec = importlib.util.spec_from_file_location("pi_env_decision", path)
decision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision)

TEACHER = """You label training data for S1, a fast executor for browser and desktop environments that picks one option per question.
S1 sees the state below and must answer three questions. Pick what an expert would do NEXT.

Facts about execution:
- recent_actions lists what was already executed in this task. page_changed=false means the action ran
  but the observed page did not change: repeating it is very unlikely to help.
- S1 cannot navigate to URLs, open apps, run scripts, or use other tools. ASK_S2 hands control to a planner that can.
- READ re-observes the page; LOOK takes a screenshot. DONE only when the phase done condition visibly holds.

State:
{state}

Question "dispatch" (options: {dispatch}):
{dispatch_rules}

Question "operation" (options: {operations}):
{operation_criteria}

Targets per operation (option keys are element indices in state.elements):
{targets}

Answer with JSON only:
{{"dispatch": "<LOCAL|DONE|ASK_S2>", "operation": "<operation key>", "target": "<index or null>", "reason": "<one sentence>"}}
If dispatch is DONE or ASK_S2, set operation to the same key and target to null."""

TOOL_TEACHER = """You label training data for S1, a fast executor that chooses the next non-terminal MCP tool.
S1 sees the state below and answers one question. Pick what an expert would do NEXT.

Facts about execution:
- recent_actions lists the tool calls already made in this task and their results. is_error=true means
  the call failed; an option label that says a call failed or repeated exactly is evidence against
  calling it again unchanged.
- S1 writes at most 64 tokens of arguments or answer. ASK_S2 hands control to a planner for reasoning,
  terminal work, code, recovery or longer answers.

Question: {question}

State:
{state}

Options:
{options}

Answer with JSON only: {{"choice": "<option key>", "reason": "<one sentence>"}}"""

ATTEMPTS = re.compile(r"\((already tried \d+x; page did not change)\)$")
EVIDENCE = re.compile(r" \(called \d+x in this task; last result [^)]*\)$")


def renumber(history):
    """Recompute unchanged_attempts exactly as bridge.ts envContext does."""
    out = []
    for index, entry in enumerate(history):
        aid = entry.get("action_id")
        count = sum(1 for h in history[:index + 1] if aid and h.get("action_id") == aid and h.get("page_changed") is False)
        out.append({**entry, "unchanged_attempts": count})
    return out


def failed_target(request, next_call, rng):
    """The action to mark as already failed: the one actually taken next, else a random element."""
    actions = [a for a in request["page"]["actions"] if a.get("node") is not None and a["kind"] in {"click", "fill"}]
    if not actions:
        return None
    chosen = None
    if next_call and str(next_call.get("name", "")).endswith("__act"):
        chosen = next_call["arguments"].get("action_id")
    return next((a for a in actions if a["id"] == chosen), None) or rng.choice(actions)


def tool_evidence(recent, name):
    """Mirror of tool-policy.ts toolEvidence over the toolState recent_actions."""
    calls = [o for o in recent if o.get("action") == name]
    if not calls:
        return ""
    last, previous = calls[-1], calls[-2] if len(calls) > 1 else None
    repeated = previous is not None and previous.get("text") == last.get("text") \
        and json.dumps(previous.get("arguments")) == json.dumps(last.get("arguments"))
    result = "was an error" if last.get("is_error") else "repeated the previous call exactly" if repeated else "succeeded"
    return f" (called {len(calls)}x in this task; last result {result})"


def tool_variants(state, inject, rng):
    request = state["tool_request"]
    yield "recorded", request
    tools = [o for o in request["options"] if o["id"] not in {"FINISH", "ASK_S2"}]
    if not tools:
        return
    option = rng.choice(tools)
    name = EVIDENCE.sub("", option["label"]).split(": ", 1)[0]
    for k in inject:
        failure = {"action": name, "kind": "tool", "arguments": {}, "is_error": True,
                   "text": "Error: the call failed", "omitted_chars": 0, "output_truncated": False}
        recent = request["state"].get("recent_actions", []) + [failure] * k
        options = [{**o, "label": EVIDENCE.sub("", o["label"]) + tool_evidence(recent, EVIDENCE.sub("", o["label"]).split(": ", 1)[0])}
                   if o["id"] not in {"FINISH", "ASK_S2"} else o for o in request["options"]]
        yield f"injected_{option['id']}_x{k}", {**request, "state": {**request["state"], "recent_actions": recent}, "options": options}


def variants(state, inject, rng):
    if "tool_request" in state:
        yield from tool_variants(state, inject, rng)
        return
    request = state["request"]
    yield "recorded", request
    action = failed_target(request, state.get("next_call"), rng)
    if action is None:
        return
    for k in inject:
        attempt = {"action": action["label"], "kind": action["kind"], "action_id": action["id"],
                   "page_changed": False, "url_changed": False, "fields_changed": []}
        history = renumber([{k2: v for k2, v in h.items() if k2 != "unchanged_attempts"} for h in request["history"]]
                           + [attempt] * k)
        yield f"injected_{action['id']}_x{k}", {**request, "history": history}


def compile_body(request):
    page, goal, history, _context, build = decision.prepare(request)
    return build(page, goal, history)


def teacher_prompt(body, targets):
    questions = body["questions"]
    by_operation = {op: {index: {k: v for k, v in body["state"]["elements"][int(index.split(":")[0]) - 1].items()
                                 if k in ("label", "role", "value", "href")}
                         | ({"note": note.group(1)} if (note := ATTEMPTS.search(questions[op.lower() + "_target"]["criteria"][index])) else {})
                         for index in candidates}
                    for op, candidates in targets.items()}
    return TEACHER.format(
        state=json.dumps(body["state"], ensure_ascii=False, indent=1),
        dispatch=", ".join(questions["dispatch"]["criteria"]),
        dispatch_rules=json.dumps(questions["dispatch"], ensure_ascii=False),
        operations=", ".join(questions["operation"]["criteria"]),
        operation_criteria=json.dumps(questions["operation"]["criteria"], ensure_ascii=False, indent=1),
        targets=json.dumps(by_operation, ensure_ascii=False, indent=1))


def ask_teacher(prompt, args):
    body = {"model": args.teacher, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    request = urllib.request.Request(args.teacher_url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {args.teacher_key}"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        text = json.load(response)["choices"][0]["message"]["content"] or ""
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else None


def valid(label, body, targets):
    if not isinstance(label, dict):
        return None
    questions = body["questions"]
    dispatch, operation, target = label.get("dispatch"), label.get("operation"), label.get("target")
    if dispatch not in questions["dispatch"]["criteria"]:
        return None
    if dispatch in {"DONE", "ASK_S2"}:
        operation, target = dispatch, None
    if operation not in questions["operation"]["criteria"]:
        return None
    if operation in targets:
        target = None if target is None else str(target)
        if target not in targets[operation]:
            return None
    else:
        target = None
    return dispatch, operation, target


def rows_for(body, label, meta):
    dispatch, operation, target = label
    state = body["state"]
    out = [{"state": state, "question": body["questions"]["dispatch"], "label": dispatch, "meta": {**meta, "head": "dispatch"}},
           {"state": state, "question": body["questions"]["operation"], "label": operation, "meta": {**meta, "head": "operation"}}]
    if target is not None:
        head = operation.lower() + "_target"
        out.append({"state": state, "question": body["questions"][head], "label": target, "meta": {**meta, "head": head}})
    return out


def label_tool_variant(item, args):
    source, name, request = item
    question = {"type": "choice", "instructions": request["question"],
                "criteria": {o["id"]: o["label"] for o in request["options"]}}
    prompt = TOOL_TEACHER.format(question=request["question"], state=json.dumps(request["state"], ensure_ascii=False, indent=1),
                                 options=json.dumps(question["criteria"], ensure_ascii=False, indent=1))
    if args.dry_run:
        return prompt, []
    votes = []
    for _ in range(args.votes):
        try:
            answer = ask_teacher(prompt, args)
            votes.append(answer.get("choice") if isinstance(answer, dict) and answer.get("choice") in question["criteria"] else None)
        except Exception as error:
            print(f"teacher error {source} {name}: {error}", file=sys.stderr)
    counted = collections.Counter(v for v in votes if v)
    if not counted or counted.most_common(1)[0][1] * 2 <= args.votes:
        return None, []
    label, agree = counted.most_common(1)[0]
    meta = {**source, "variant": name, "teacher": args.teacher, "votes": agree, "of": args.votes, "head": "rank"}
    return None, [{"state": request["state"], "question": question, "label": label, "meta": meta}]


def label_variant(item, args):
    source, name, request = item
    if source.get("kind") == "tools":
        return label_tool_variant(item, args)
    body, targets, _controls, _operations = compile_body(request)
    prompt = teacher_prompt(body, targets)
    if args.dry_run:
        return prompt, []
    votes = []
    for _ in range(args.votes):
        try:
            votes.append(valid(ask_teacher(prompt, args), body, targets))
        except Exception as error:  # one failed sample is a missing vote, not a failed dataset
            print(f"teacher error {source} {name}: {error}", file=sys.stderr)
    counted = collections.Counter(v for v in votes if v)
    if not counted:
        return None, []
    label, agree = counted.most_common(1)[0]
    if agree * 2 <= args.votes:  # require a strict majority
        return None, []
    meta = {**source, "variant": name, "teacher": args.teacher, "votes": agree, "of": args.votes}
    return None, rows_for(body, label, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states")
    parser.add_argument("out", help="rows.jsonl, or - for stdout")
    parser.add_argument("--inject", type=int, nargs="*", default=[1, 3], help="no-effect attempt counts to inject")
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--teacher", default=os.environ.get("TAIJI_S2_MODEL", "opencode-go/muse-spark-1.3-contributor"))
    parser.add_argument("--teacher-url", default=os.environ.get("TAIJI_S2_BASE_URL", "http://127.0.0.1:10100/v1"))
    parser.add_argument("--teacher-key", default=os.environ.get("TAIJI_S2_API_KEY", "opencodex"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="label at most this many variants")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print teacher prompts; call nothing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items, seen = [], set()
    for line in open(args.states):
        state = json.loads(line)
        for name, request in variants(state, args.inject, rng):
            # Many recorded states repeat the same page and history; label each once.
            key = json.dumps([request["state"], request["options"]] if "options" in request else
                             [request["page"]["url"], request["page"]["text"][:2000], request["history"], request["goal"]], sort_keys=True)
            if key not in seen:
                seen.add(key)
                items.append(({"kind": "env", **state["source"]}, name, request))
    items = items[:args.limit] if args.limit else items
    print(f"{len(items)} variants to label", file=sys.stderr)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    written = labelled = 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for prompt, rows in pool.map(lambda item: label_variant(item, args), items):
            if prompt:
                print(prompt, file=out)
            labelled += bool(rows)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"{labelled}/{len(items)} variants labelled, {written} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
