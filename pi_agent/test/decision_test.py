"""Protocol checks for the Pi-only decision adapter, without model or browser calls."""
import contextlib
import io
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "src" / "browser_decision.py"
spec = importlib.util.spec_from_file_location("pi_browser_decision", path)
decision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision)
model = decision.browser_policy.model
dataset_spec = importlib.util.spec_from_file_location("recovery_dataset", Path(__file__).resolve().parents[1] / "tools" / "recovery_dataset.py")
dataset = importlib.util.module_from_spec(dataset_spec)
dataset_spec.loader.exec_module(dataset)


def answer(criteria, choice):
    return {"choice": choice, "confidence": 1,
            "probabilities": {key: float(key == choice) for key in criteria}}


class Decisions(unittest.TestCase):
    def setUp(self):
        self.page = {"url": "https://example.test", "title": "Example", "text": "Open item",
                     "actions": [{"id": "r1", "kind": "click", "node": 1, "label": "Open item"},
                                 {"id": "ask_s2", "kind": "handoff", "node": None, "label": "Ask S2"}]}

    def test_dispatch_has_goal_and_receives_transition_evidence(self):
        build = decision.enriched_builder(model.build_request, ["browser_act"])
        body, _, _, _ = build(self.page, "User goal", [{"unchanged_attempts": 4}])
        self.assertEqual(body["questions"]["dispatch"]["instructions"]["goal"], "User goal")
        self.assertEqual(body["state"]["recent_actions"][0]["unchanged_attempts"], 4)

    def test_full_loop_history_survives_request_encoding_without_a_last_ten_slice(self):
        history = [{"action": f"click {i}", "page_changed": False, "unchanged_attempts": i + 1} for i in range(35)]
        body, _, _, _ = decision.enriched_builder(model.build_request, [])(self.page, "Goal", history)
        self.assertEqual(body["state"]["recent_actions"], history)
        self.assertEqual(body["state"]["recent_actions"][0]["action"], "click 0")

    def test_model_dispatch_owns_action_and_unused_targets_cannot_act(self):
        def post(_url, _key, body):
            questions = body["questions"]
            return {"model": "test", "answers": {
                "dispatch": answer(questions["dispatch"]["criteria"], "ASK_S2"),
                "operation": answer(questions["operation"]["criteria"], "CLICK"),
                "click_target": {"choice": "unobserved", "probabilities": {}},
            }}
        build = decision.enriched_builder(model.build_request, [])
        with patch.object(model, "build_request", build), patch.object(model, "post_json", post), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}):
            result = decision.choose(self.page, "Different goal", [], "test")
        self.assertEqual(result["operation"], "ASK_S2")
        self.assertEqual(result["choice"], "ask_s2")
        self.assertIsNone(result["target"])

    def test_s1_phase_done_does_not_compete_with_a_click_target_or_need_another_call(self):
        calls = []
        def post(_url, _key, body):
            calls.append(body)
            return {"model": "test", "answers": {
                "dispatch": answer(body["questions"]["dispatch"]["criteria"], "DONE"),
                "operation": answer(body["questions"]["operation"]["criteria"], "CLICK")}}
        build = decision.enriched_builder(model.build_request, [])
        with patch.object(model, "build_request", build), patch.object(model, "post_json", post), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}):
            result = decision.choose(self.page, "Open the target; done when target is visible", [], "test")
        self.assertEqual(result["operation"], "DONE")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result["target"])

    def test_local_phase_preserves_parent_task_as_separate_state(self):
        build = decision.enriched_builder(model.build_request, ["browser_act"], "User requires route/date constraints")
        body, _, _, _ = build(self.page, "Click the visible next control; stop when new page is shown", [])
        self.assertEqual(body["state"]["task_context"], "User requires route/date constraints")
        self.assertIn("CURRENT execution phase", body["questions"]["operation"]["criteria"]["DONE"])
        self.assertIn("choose DONE", body["questions"]["operation"]["instructions"]["rules"])
        self.assertEqual(body["questions"]["dispatch"]["instructions"]["goal"], "Click the visible next control; stop when new page is shown")

    def test_large_observation_reaches_model_without_character_guard(self):
        self.page["text"] = "evidence " * 4000
        def post(_url, _key, body):
            self.assertGreater(decision.head_size(body), 20000)
            return {"model": "test", "answers": {
                "dispatch": answer(body["questions"]["dispatch"]["criteria"], "ASK_S2"),
                "operation": answer(body["questions"]["operation"]["criteria"], "CLICK")}}
        build = decision.enriched_builder(model.build_request, [])
        with patch.object(model, "build_request", build), patch.object(model, "post_json", post), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}):
            result = decision.choose(self.page, "User goal", [], "test")
        self.assertEqual(result["operation"], "ASK_S2")

    def test_budget_counts_one_sequence_not_sum_of_questions(self):
        body = {"state": {"page": "x" * 2000}, "questions": {
            str(i): {"instructions": "x" * 10000} for i in range(4)}}
        self.assertGreater(len(json.dumps(body)), 20000)
        self.assertLess(decision.head_size(body), 20000)

    def test_indexed_targets_keep_all_page_metadata(self):
        self.page["actions"][0].update(role="link", href="https://example.test/item", value="Value")
        body, targets, _, _ = decision.enriched_builder(model.build_request, [])(self.page, "User goal", [])
        self.assertEqual(body["questions"]["click_target"]["criteria"], {"1": "[1] Open item"})
        self.assertEqual(body["state"]["elements"][0]["label"], "Open item")
        self.assertEqual(body["state"]["elements"][0]["href"], "https://example.test/item")
        self.assertEqual(body["state"]["elements"][0]["value"], "Value")
        self.assertEqual(targets["CLICK"]["1"]["id"], "r1")

    def test_failed_attempts_annotate_the_matching_target_option(self):
        history = [{"action_id": "r1", "page_changed": False}] * 3 + [{"action_id": "r1", "page_changed": True}]
        body, _, _, _ = decision.enriched_builder(model.build_request, [])(self.page, "User goal", history)
        self.assertEqual(body["questions"]["click_target"]["criteria"],
                         {"1": "[1] Open item (already tried 3x; page did not change)"})

    def test_prepare_is_repeatable_for_dataset_compilation(self):
        request = {"page": self.page, "goal": "Goal", "history": [], "capabilities": []}
        decision.prepare(request)
        decision.prepare(request)
        self.assertEqual(model.NEXT_ACTION.count(decision.RECOVERY_RULES), 1)

    def test_environment_selects_its_operation_rules_without_changing_request_shape(self):
        request = {"page": self.page, "goal": "Goal", "history": [], "capabilities": []}
        page, goal, history, _, build = decision.prepare({**request, "env": "desktop"})
        desktop, *_ = build(page, goal, history)
        self.assertTrue(model.NEXT_ACTION.startswith(decision.DESKTOP_RULES))
        page, goal, history, _, build = decision.prepare(request)
        web, *_ = build(page, goal, history)
        self.assertTrue(model.NEXT_ACTION.startswith(decision.WEB_RULES))
        self.assertEqual(set(desktop["state"]), set(web["state"]))
        self.assertEqual(set(desktop["questions"]), set(web["questions"]))

    def test_operation_before_target_and_unused_heads_do_not_block_wait(self):
        self.page["actions"].append({"id": "wait", "kind": "wait", "node": None, "label": "Wait"})
        calls = []

        def post(_url, _key, body):
            calls.append(list(body["questions"]))
            return {"model": "test", "answers": {
                "dispatch": answer(body["questions"]["dispatch"]["criteria"], "LOCAL"),
                "operation": answer(body["questions"]["operation"]["criteria"], "WAIT"),
            }}

        def build(*args):
            body, *rest = decision.enriched_builder(original, [])(*args)
            body["questions"]["click_target"]["instructions"]["rules"] = ["x" * 50000]
            return body, *rest

        original = model.build_request
        with patch.object(model, "build_request", build), patch.object(model, "post_json", post), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}):
            result = decision.choose(self.page, "Wait for results", [], "test")
        self.assertEqual(result["operation"], "WAIT")
        self.assertEqual(calls, [["dispatch", "operation"]])

    def test_click_requests_only_selected_target_head(self):
        calls = []

        def post(_url, _key, body):
            calls.append(list(body["questions"]))
            if len(calls) == 1:
                return {"model": "test", "answers": {
                    "dispatch": answer(body["questions"]["dispatch"]["criteria"], "LOCAL"),
                    "operation": answer(body["questions"]["operation"]["criteria"], "CLICK"),
                }}
            return {"model": "test", "answers": {
                "click_target": answer(body["questions"]["click_target"]["criteria"], "1")}}

        with patch.object(model, "build_request", decision.enriched_builder(model.build_request, [])), patch.object(model, "post_json", post), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}):
            result = decision.choose(self.page, "Open item", [], "test")
        self.assertEqual(result["choice"], "r1")
        self.assertEqual(calls, [["dispatch", "operation"], ["click_target"]])


class Dataset(unittest.TestCase):
    def test_tool_variants_inject_failures_with_the_served_evidence_text(self):
        request = {"question": "Choose", "state": {"page": {"url": "tools://"}, "recent_actions": []},
                   "options": [{"id": "t0", "label": "mcp__files__search: Search files"},
                               {"id": "FINISH", "label": "done"}, {"id": "ASK_S2", "label": "help"}]}
        variants = dict(dataset.variants({"tool_request": request}, [2], __import__("random").Random(0)))
        injected = variants["injected_t0_x2"]
        self.assertEqual(injected["options"][0]["label"],
                         "mcp__files__search: Search files (called 2x in this task; last result was an error)")
        self.assertEqual(len(injected["state"]["recent_actions"]), 2)
        self.assertEqual(variants["recorded"], request)

    def test_desktop_states_compile_with_desktop_rules(self):
        page = {"url": "app://TextEdit/Untitled", "title": "Untitled", "text": "",
                "actions": [{"id": "e2", "kind": "fill", "node": 2, "label": "Body"}]}
        request = {"env": "desktop", "page": page, "goal": "Type hello", "history": [], "capabilities": []}
        body, targets, _, _ = dataset.compile_body(request)
        self.assertIn("TYPE_TEXT", targets)
        self.assertTrue(body["questions"]["operation"]["instructions"]["rules"].startswith(decision.DESKTOP_RULES))


class ControlEvidence(unittest.TestCase):
    def test_repeated_ineffective_controls_are_marked_on_the_operation_option(self):
        page = {"url": "https://example.test", "title": "Example", "text": "Article",
                "actions": [{"id": "r1", "kind": "click", "node": 1, "label": "Open item"},
                            {"id": "scroll_up", "kind": "scroll", "node": None, "label": "Scroll up"}]}
        history = [{"action": "Scroll up", "kind": "scroll", "action_id": "scroll_up", "page_changed": False}] * 3
        body, _, _, _ = decision.enriched_builder(model.build_request, [])(page, "Goal", history)
        self.assertEqual(body["questions"]["operation"]["criteria"]["SCROLL_UP"],
                         "Scroll up (already tried 3x; page did not change)")


if __name__ == "__main__":
    unittest.main()
