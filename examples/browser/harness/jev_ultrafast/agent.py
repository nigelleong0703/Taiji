"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import os
import time
from pathlib import Path

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text, reflect
from .questions import MAX_REFLECTIONS, MAX_STEPS

# Consecutive decisions voided by page changes: wait for the page to settle, then hand the page to System 2.
STALE_SETTLE, STALE_ESCALATE = 3, 8


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir) or bool(os.environ.get("TYPESAFE_SCREENSHOT"))
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            reflections=[],
            subgoal=None,
            blocked_reason=None,
            reflected_at=0,
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                result = self.command("act", {"fingerprint": state["page"]["fingerprint"]})
                state["stale_streak"] = 0
                return result
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                # A page that keeps changing (live prices, animations) voids decision after decision. Let it settle;
                # if it never does, System 2 decides what to do instead of spending the model-call budget.
                state["stale_streak"] = state.get("stale_streak", 0) + 1
                if state["stale_streak"] >= STALE_ESCALATE:
                    state["stale_streak"] = 0
                    self.escalate("The page keeps changing before a decision can run.")
                elif state["stale_streak"] >= STALE_SETTLE:
                    state["browser"].settle()
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            goal = state["goal"] + (f"\nPlanner subgoal: {state['subgoal']}" if state.get("subgoal") else "")
            state["decision"] = choose(state["page"], goal, state["history"])
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page, page_only=True):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                if selected == "BLOCKED" and not any(a["kind"] in {"click", "fill", "select"} for a in page["actions"]):
                    # Mechanical: an empty snapshot is a page mid-transition, not a reasoning problem. No System 2.
                    time.sleep(0.2)
                    state["page"] = state["browser"].observe(screenshot=self.screenshots)
                    state["status"] = "ready"
                elif selected == "BLOCKED":
                    self.escalate("The fast policy chose BLOCKED.")
                else:
                    state["status"] = "done"
                    state["plan_index"] = 1
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page, action):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "before": page["fingerprint"],
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            since = state["history"][state.get("reflected_at", 0) :]
            stuck = len(since) >= 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in since[-3:])
            # A loop is the same action on the same page again: paging a calendar with Next changes the page each time.
            moves = [(h["action"], h["before"]) for h in since[-6:] if h["kind"] != "wait"]
            looping = any(moves.count(move) >= 3 for move in moves)
            state["status"] = "ready"
            if stuck or looping:
                self.escalate("Three actions did not change the page." if stuck else "The same action keeps repeating.")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def escalate(self, trigger):
        """System 1 is stuck: ask System 2 for a language subgoal, or stop: finished, or infeasible with its reason."""
        state = self.state
        state.setdefault("reflections", [])
        if len(state["reflections"]) >= MAX_REFLECTIONS:
            state["status"], state["blocked_reason"] = "blocked", trigger
            return
        # A System 2 call changes nothing in the browser, so one retry of a malformed or failed answer is safe.
        for attempt in (1, 2):
            try:
                verdict, text, meta = reflect(state["goal"], state["page"], state["history"], trigger)
                break
            except (ValueError, RuntimeError) as error:
                if attempt == 2:
                    state["status"] = "blocked"
                    state["blocked_reason"] = f"{trigger} System 2 gave no usable answer: {error}"
                    return
        state["reflections"].append({**meta, "trigger": trigger, "verdict": verdict, "text": text})
        state["reflected_at"] = len(state["history"])
        if verdict == "infeasible":
            state["status"], state["blocked_reason"] = "blocked", text
        elif verdict == "done":  # like a DONE choice: claimed, not proven; callers verify the outcome
            state["status"], state["plan_index"] = "done", 1
        else:
            state["subgoal"], state["status"] = text, "ready"

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
