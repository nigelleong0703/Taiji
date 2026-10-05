"""S1 request layout v5: stable text first, so vLLM's prefix cache reuses it across steps.

The trained layout (v3/v4) serialises `State: {page, elements, recent_actions}` and then
`Instructions: {goal, rules}`: the page text changes every step and sits first, so the goal and
the shared rules behind it are recomputed on every decision. v5 keeps the same encoder
(`s1.encode`: State, then Instructions, then Options) and only reorders the row JSON:

    state = {task{goal, rules, context?, steps?}, capabilities?, elements, recent_actions, page, screenshot?}
    question.instructions = only what differs between questions (operation, TARGET rule, ...)

History is compact (the four trained fields) and windowed in blocks of five, so its prefix stays
stable for five steps instead of shifting every step.

The training repository (qwen-decision/layout.py) carries a verbatim copy; test/layout_fixture.json
pins both copies to the same output.
"""

VERSION = "v5"
HISTORY_FIELDS = ("action", "kind", "text", "page_changed")
STATE_ORDER = ("elements", "recent_actions")


def history_window(history):
    """All entries below ten; otherwise from the latest multiple of five that leaves at least five."""
    start = 0 if len(history) < 10 else (len(history) - 5) // 5 * 5
    return [{field: entry.get(field) for field in HISTORY_FIELDS} for entry in history[start:]]


def without_shared(rules, shared):
    """The question-specific part of `rules` once `shared` has moved into state.task."""
    if isinstance(rules, str):
        rest = rules[len(shared):].strip() if rules.startswith(shared) else rules
        return rest or None
    if isinstance(rules, list):
        rest = [rule for rule in rules if rule != shared]
        return rest or None
    return rules


def relayout(state, question, shared_rules, steps=None):
    """One (state, question) in v5 layout. Questions without a goal (general, tools) pass through."""
    instructions = question.get("instructions")
    if not isinstance(instructions, dict) or "goal" not in instructions or "task" in state:
        return state, question
    task = {"goal": instructions["goal"], "rules": shared_rules}
    if state.get("task_context") is not None:
        task["context"] = state["task_context"]
    if steps:
        task["steps"] = list(steps)
    out = {"task": task}
    if state.get("available_capabilities") is not None:
        out["capabilities"] = state["available_capabilities"]
    for key in STATE_ORDER:
        if key in state:
            out[key] = history_window(state[key]) if key == "recent_actions" else state[key]
    skipped = {"page", "screenshot", "task_context", "available_capabilities", *STATE_ORDER}
    out.update({key: value for key, value in state.items() if key not in skipped})
    out["page"] = state["page"]
    if "screenshot" in state:
        out["screenshot"] = state["screenshot"]
    return out, question_only(question, shared_rules)


def question_only(question, shared_rules):
    """The question with its goal and the shared rules removed (both live in state.task)."""
    rest = {key: value for key, value in question["instructions"].items() if key != "goal"}
    rules = without_shared(rest.pop("rules", None), shared_rules)
    if rules is not None:
        rest["rules"] = rules
    return {**question, "instructions": rest}


def relayout_body(body, shared_rules, steps=None):
    """Every question of one request; they share one state, hence one cached prefix."""
    state, questions = body["state"], {}
    for name, question in body["questions"].items():
        new_state, questions[name] = relayout(body["state"], question, shared_rules, steps)
        state = new_state if "task" in new_state else state
    return {**body, "state": state, "questions": questions}
