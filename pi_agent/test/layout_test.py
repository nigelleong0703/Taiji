"""Layout v5: stable state first, one shared state per request, fixture shared with the training copy."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import layout  # noqa: E402

spec = importlib.util.spec_from_file_location("pi_env_decision", ROOT / "src" / "env_decision.py")
decision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision)
model = decision.browser_policy.model
FIXTURE = Path(__file__).with_name("layout_fixture.json")

PAGE = {"url": "app://Calculator/Calculator", "title": "Calculator", "text": "Edit field: 12",
        "actions": [{"id": "e1", "kind": "click", "node": 1, "label": "1"},
                    {"id": "e2", "kind": "click", "node": 2, "label": "Multiply"},
                    {"id": "key_enter", "kind": "key", "node": None, "label": "Press Enter"}]}


def request(history, steps=None):
    return {"env": "desktop", "layout": "v5", "page": PAGE, "goal": "Compute 12 × 7", "subgoal": "enter × 7 =",
            "done_when": "display shows 84", "history": history, "capabilities": ["mcp__desktop__act"],
            "steps": steps}


def body_for(req):
    page, goal, history, _context, build = decision.prepare(req)
    return build(page, goal, history)[0]


class Layout(unittest.TestCase):
    def test_stable_parts_lead_and_the_page_is_last(self):
        body = body_for(request([{"action": "1", "kind": "click", "action_id": "e1", "page_changed": True}],
                                ["press ×", "press 7", "press ="]))
        state = body["state"]
        self.assertEqual(list(state), ["task", "capabilities", "elements", "recent_actions", "page"])
        self.assertEqual(state["task"]["steps"], ["press ×", "press 7", "press ="])
        self.assertTrue(state["task"]["goal"].startswith("Execute this delegated phase"))
        self.assertEqual(state["task"]["rules"], model.NEXT_ACTION)
        self.assertEqual(state["recent_actions"], [{"action": "1", "kind": "click", "text": None, "page_changed": True}])
        for name, question in body["questions"].items():
            self.assertNotIn("goal", question["instructions"], name)
            self.assertNotIn(model.NEXT_ACTION, json.dumps(question["instructions"]), name)
        self.assertTrue(body["questions"]["operation"]["instructions"]["rules"].startswith("First check"))
        self.assertIn("rules", body["questions"]["dispatch"]["instructions"])

    def test_history_window_moves_in_blocks_of_five(self):
        sizes = [len(layout.history_window([{}] * n)) for n in range(0, 21)]
        self.assertEqual(sizes, [*range(10), 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5])
        prefixes = [json.dumps(layout.history_window([{"action": str(i)} for i in range(n)]))[:-1] for n in (15, 16, 17)]
        self.assertTrue(prefixes[1].startswith(prefixes[0]) and prefixes[2].startswith(prefixes[1]))

    def test_trained_layout_is_unchanged_without_the_flag(self):
        trained = body_for({**request([]), "layout": None})
        self.assertEqual(list(trained["state"])[:3], ["page", "elements", "recent_actions"])
        self.assertIn("goal", trained["questions"]["operation"]["instructions"])

    def test_questions_without_a_goal_and_v5_states_pass_through(self):
        tool = {"type": "choice", "instructions": "Pick a tool", "criteria": {"a": "A"}}
        self.assertEqual(layout.relayout({"page": {}}, tool, "R"), ({"page": {}}, tool))
        state = {"task": {"goal": "g"}, "page": {}}
        question = {"instructions": {"goal": "g", "rules": "R"}}
        self.assertEqual(layout.relayout(state, question, "R"), (state, question))

    def test_fixture_matches(self):
        """The same fixture is asserted by qwen-decision/tests/test_layout.py against its copy."""
        fixture = json.loads(FIXTURE.read_text())
        for case in fixture:
            state, question = layout.relayout(case["state"], case["question"], case["shared_rules"], case.get("steps"))
            self.assertEqual({"state": state, "question": question}, case["expected"])


if __name__ == "__main__":
    unittest.main()
