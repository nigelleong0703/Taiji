"""Check a trace for the three ways a browser agent silently gets wrong input.

    python3 lint_trace.py <trace.json>

Every one of these has bitten this project:
  * frozen-state   the runtime stops refreshing its page copy, so the model decides on a stale page
  * id-not-offered the executed action id was never in the observation the model chose from
  * truncated      the observation the runtime read was cut by a size cap
"""

from __future__ import annotations

import json
import sys


def load(path):
    trace = json.loads(open(path).read())
    decisions = [event for event in trace.get("trace", []) if event["event"] == "s1_decision"]
    tools = [event for event in trace.get("trace", []) if event["event"] == "mcp_tool"]
    steps = {step.get("step"): step for step in trace.get("steps", [])}
    return trace, decisions, tools, steps


def observed_ids(tools, step):
    """Action ids the runtime held when it made the decision at this step."""
    previous = [tool for tool in tools if (tool.get("step") or 0) < step and tool.get("observed_actions")]
    return {action["id"] for action in previous[-1]["observed_actions"]} if previous else set()


def main():
    trace, decisions, tools, steps = load(sys.argv[1])
    problems = []

    previous_state = None
    frozen_streak = 0
    for decision in decisions:
        # Only the observation matters here. Comparing the whole state would hide a frozen page,
        # because the action history changes on every step.
        state = (decision.get("request") or {}).get("state") or {}
        observation = json.dumps({"page": state.get("page"), "elements": state.get("elements")},
                                 sort_keys=True)
        frozen_streak = frozen_streak + 1 if observation == previous_state else 0
        previous_state = observation
        if frozen_streak == 1:
            problems.append(("frozen-state", "step %s" % decision.get("step"),
                             "the request state is byte-identical to the previous decision's"))
        offered = observed_ids(tools, decision.get("step") or 0)
        chosen = decision.get("choice")
        if offered and chosen and chosen not in offered and chosen not in {"DONE", "BLOCKED", "FINISH"}:
            problems.append(("id-not-offered", "step %s" % decision.get("step"),
                             "chose %s but the runtime held only %d ids (%s...)"
                             % (chosen, len(offered), ", ".join(sorted(offered)[:4]))))

    for step_number, step in steps.items():
        text = step.get("result")
        if isinstance(text, str) and "actions" in text and text.rstrip().endswith("[truncated]"):
            problems.append(("truncated", "step %s" % step_number,
                             "the observation the runtime read was cut at %d chars" % len(text)))

    print("trace:", sys.argv[1])
    print("decisions:", len(decisions), "| tool calls:", len(tools),
          "| status:", trace.get("status"))
    if not problems:
        print("no problems found")
        return
    seen = set()
    for kind, where, detail in problems:
        key = (kind, detail.split("(")[0])
        if key in seen:
            continue
        seen.add(key)
        print("  %-15s %-10s %s" % (kind, where, detail))
    print()
    print("total findings: %d (deduplicated above)" % len(problems))


if __name__ == "__main__":
    main()
