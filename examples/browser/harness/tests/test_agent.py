"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
        "verification": None,
    }
    a.verifier = None
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_stuck_policy_asks_system2_then_uses_its_subgoal(runner, monkeypatch):
    s2 = Mock(return_value=("subgoal", "Click Go to submit the search", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "reflect", s2)
    for _ in range(3):
        runner.state["decision"] = decision("e3")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert s2.call_count == 1 and runner.state["status"] == "ready"
    assert runner.state["subgoal"] == "Click Go to submit the search"
    runner.state["browser"].fresh.return_value = True
    choose = Mock(return_value=decision("e3"))
    monkeypatch.setattr(loop, "choose", choose)
    runner.command("predict")
    assert "Planner subgoal: Click Go" in choose.call_args.args[1]


def test_system2_infeasible_stops_with_reason(runner, monkeypatch):
    monkeypatch.setattr(loop, "reflect", Mock(return_value=("infeasible", "Date is in the past", {"latency_ms": 1})))
    runner.state["decision"] = {**decision("BLOCKED"), "operation": "BLOCKED"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "blocked" and runner.state["blocked_reason"] == "Date is in the past"


def test_empty_snapshot_blocked_reobserves_without_system2(runner, monkeypatch):
    s2 = Mock()
    monkeypatch.setattr(loop, "reflect", s2)
    runner.state["page"]["actions"] = [{"id": "wait", "kind": "wait", "label": "Wait"}]
    runner.state["decision"] = {**decision("BLOCKED"), "operation": "BLOCKED"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    s2.assert_not_called()
    assert runner.state["status"] == "ready"


@pytest.mark.parametrize("content", ['{"verdict":"click","text":"e3"}', '{"verdict":"subgoal"}', "Just click it"])
def test_system2_rejects_anything_but_a_verdict(monkeypatch, content):
    monkeypatch.setattr(model, "text_model", Mock(return_value=(content, {})))
    with pytest.raises(ValueError, match="no valid verdict"):
        model.reflect("Find a book", page(), [], "stuck")


def test_system2_done_is_invalid_after_verifier_failure(monkeypatch):
    monkeypatch.setattr(model, "text_model", Mock(return_value=('{"verdict":"done","text":"Looks complete"}', {})))
    with pytest.raises(ValueError, match="no valid verdict"):
        model.reflect("Find a book", page(), [], "verification failed", allow_done=False)


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def test_null_from_fast_text_model_escalates_once_to_system2(monkeypatch):
    calls = Mock(side_effect=[('{"text":null}', {"model": "fast"}), ('{"text":"Zurich"}', {"model": "slow"})])
    monkeypatch.setattr(model, "text_model", calls)
    value, meta = model.field_text({"goal": "Fly from Zurich"})
    assert value == "Zurich" and meta["escalated"] and calls.call_args.args[2] == "S2_MODEL"


def near_tie_post(margin, tie_pick):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        if "tie" in body["questions"]:
            return {"model": "test", "answers": {"tie": choice(list(body["questions"]["tie"]["criteria"]), tie_pick)}}
        top = (1 + margin) / 2
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "click_target": {"choice": "1", "confidence": top, "probabilities": {"1": top, "2": 1 - top}},
            },
        }

    return calls, post


def test_near_tie_asks_system1_once_with_only_the_two_candidates(monkeypatch):
    calls, post = near_tie_post(0.1, "2")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 2 and set(calls[1]["questions"]["tie"]["criteria"]) == {"1", "2"}
    assert d["target"] == "2" and d["choice"] == "e3" and d["tie_break"]["first"] == "1"


def test_clear_winner_makes_no_extra_request(monkeypatch):
    calls, post = near_tie_post(0.6, "2")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1 and d["target"] == "1" and d["tie_break"] is None


def done_second_post(done_probability):
    def post(url, key, body, headers=None):
        rest = (1 - done_probability) / (len(body["questions"]["operation"]["criteria"]) - 1)
        probabilities = {k: rest for k in body["questions"]["operation"]["criteria"]}
        probabilities["CLICK"], probabilities["DONE"] = rest + 0.3, done_probability
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}
        return {"model": "test", "answers": {
            "operation": {"choice": "CLICK", "confidence": 0.5, "probabilities": probabilities},
            "click_target": choice(list(body["questions"]["click_target"]["criteria"]), "1")}}

    return post


@pytest.mark.parametrize("floor, expected", [(0.0, "CLICK"), (0.15, "DONE"), (0.5, "CLICK")])
def test_done_floor_picks_done_only_when_set_and_reached(monkeypatch, floor, expected):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", done_second_post(0.2))
    monkeypatch.setattr(model, "DONE_FLOOR", floor)
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == expected and d["done_floor"] == (expected == "DONE")
    if expected == "DONE":
        assert d["choice"] == "DONE" and d["target"] is None


def test_done_checks_only_the_page_identity_not_live_content(runner):
    runner.state["decision"] = {**decision("DONE"), "operation": "DONE", "target": None}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].fresh.assert_called_with(runner.state["page"], page_only=True)
    assert runner.state["status"] == "done"


def test_done_with_failed_independent_check_replans(runner, monkeypatch):
    runner.verifier = Mock(return_value={"passed": False, "checks": {"results_visible": False}})
    s2 = Mock(return_value=("subgoal", "Submit the populated search form", {"latency_ms": 1}))
    monkeypatch.setattr(loop, "reflect", s2)
    runner.state["decision"] = {**decision("DONE"), "operation": "DONE", "target": None}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "ready"
    assert runner.state["subgoal"] == "Submit the populated search form"
    assert runner.state["verification"]["checks"]["results_visible"] is False
    assert s2.call_args.kwargs["allow_done"] is False


def test_done_with_passed_independent_check_finishes(runner, monkeypatch):
    runner.verifier = Mock(return_value={"passed": True, "checks": {"results_visible": True}})
    s2 = Mock()
    monkeypatch.setattr(loop, "reflect", s2)
    runner.state["decision"] = {**decision("DONE"), "operation": "DONE", "target": None}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "done"
    assert runner.state["verification"]["passed"] is True
    s2.assert_not_called()


def test_failed_independent_check_cannot_be_overridden_by_system2_done(runner, monkeypatch):
    runner.verifier = Mock(return_value=False)
    monkeypatch.setattr(loop, "reflect", Mock(return_value=("done", "It looks complete", {"latency_ms": 1})))
    runner.state["decision"] = {**decision("DONE"), "operation": "DONE", "target": None}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "blocked"
    assert runner.state["verification"]["passed"] is False
    assert "verification failed" in runner.state["blocked_reason"]


def test_fill_checks_the_field_and_its_context_before_text_generation(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(return_value=("book", {"model": "t", "latency_ms": 1})))
    runner.state["decision"] = decision("e1")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    first = runner.state["browser"].fresh.call_args_list[0]
    assert first.args[1]["id"] == "e1" and first.args[1]["kind"] == "fill"


def test_page_that_never_settles_goes_to_system2(runner, monkeypatch):
    monkeypatch.setattr(loop, "choose", Mock(return_value=decision("e3")))
    runner.state["browser"].act.side_effect = StalePage("changed")
    escalate = Mock()
    monkeypatch.setattr(runner, "escalate", escalate)
    for _ in range(loop.STALE_ESCALATE):
        runner.command("tick")
    escalate.assert_called_once()
    assert runner.state["browser"].settle.call_count == loop.STALE_ESCALATE - loop.STALE_SETTLE


def test_settle_waits_until_two_reads_agree(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.evaluate = Mock(side_effect=["a", "b", "c", "c"])
    monkeypatch.setattr(browser.time, "sleep", Mock())
    assert b.settle() is True and b.evaluate.call_count == 4


def test_malformed_system2_answer_is_retried_once_then_stops_cleanly(runner, monkeypatch):
    bad = Mock(side_effect=ValueError("System 2 returned no valid verdict; nothing executed."))
    monkeypatch.setattr(loop, "reflect", bad)
    runner.escalate("Three actions did not change the page.")
    assert bad.call_count == 2 and runner.state["status"] == "blocked"
    assert "no usable answer" in runner.state["blocked_reason"]


def test_system2_retry_recovers_after_one_bad_answer(runner, monkeypatch):
    reflect = Mock(side_effect=[ValueError("bad"), ("subgoal", "Open the calendar's next month", {"model": "s2"})])
    monkeypatch.setattr(loop, "reflect", reflect)
    runner.escalate("Three actions did not change the page.")
    assert runner.state["subgoal"] == "Open the calendar's next month" and runner.state["status"] == "ready"


def test_text_context_lists_the_other_fields_and_their_values():
    p = page()
    p["actions"] = [
        {"id": "e1", "kind": "fill", "label": "Where from?", "role": "combobox", "value": "Singapore", "node": 1},
        {"id": "e2", "kind": "fill", "label": "Departure", "role": "textbox", "value": "", "node": 2},
        {"id": "e3", "kind": "click", "label": "Search", "role": "button", "value": "", "node": 3},
        {"id": "e4", "kind": "fill", "label": "Return", "role": "textbox", "value": "", "node": 4},
    ]
    context = model.field_context("Fly SIN-DPS Dec 27", p["actions"][1], p, [])
    assert context["form_fields"] == [{"label": "Where from?", "value": "Singapore"}, {"label": "Return", "value": ""}]


def paged(text):
    p = page()
    p["text"] = text
    p["fingerprint"] = fingerprint(p)
    return p


def test_paging_with_the_same_button_is_not_a_loop(runner, monkeypatch):
    s2 = Mock()
    monkeypatch.setattr(loop, "reflect", s2)
    months = [paged(m) for m in ("October", "November", "December", "January")]
    runner.state["browser"].observe.side_effect = months
    for _ in range(4):
        runner.state["decision"] = decision("e3")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    s2.assert_not_called()


def test_returning_to_the_same_page_with_the_same_action_is_a_loop(runner, monkeypatch):
    s2 = Mock(return_value=("subgoal", "Stop swapping", {"model": "test", "latency_ms": 1}))
    monkeypatch.setattr(loop, "reflect", s2)
    a, b = paged("Singapore to Bali"), paged("Bali to Singapore")
    runner.state["page"] = a
    runner.state["browser"].observe.side_effect = [b, a] * 3
    for _ in range(5):
        runner.state["decision"] = decision("e3")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert s2.call_count == 1 and s2.call_args.args[-1] == "The same action keeps repeating."


def test_screenshot_is_sent_at_the_size_the_s1_server_uses():
    import base64
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (1120, 780), (200, 40, 40)).save(out, "JPEG")
    small = Image.open(io.BytesIO(base64.b64decode(model.shrink(base64.b64encode(out.getvalue()).decode()))))
    scale = (model.S1_IMAGE_PIXELS / (1120 * 780)) ** 0.5
    assert small.size == (int(1120 * scale), int(780 * scale))


def test_system2_done_finishes_the_run(runner, monkeypatch):
    monkeypatch.setattr(loop, "reflect", Mock(return_value=("done", "Cheapest results are shown", {"latency_ms": 1})))
    runner.state["decision"] = {**decision("BLOCKED"), "operation": "BLOCKED"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["status"] == "done" and runner.state["reflections"][-1]["verdict"] == "done"
