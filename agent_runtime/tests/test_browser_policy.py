import importlib.util
import subprocess
import sys
from pathlib import Path

from taiji_agent import browser_policy


def test_policy_import_does_not_import_browser_transport():
    subprocess.run([sys.executable, "-c",
                    "import sys; import taiji_agent.browser_policy as p; "
                    "assert p.AVAILABLE; assert 'jev_ultrafast.browser' not in sys.modules"], check=True)


def test_wide_pilot_snapshot_keeps_scroll_controls():
    path = Path(__file__).resolve().parents[1] / "examples" / "pilot_browser_server.py"
    spec = importlib.util.spec_from_file_location("pilot_adapter_test", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.MAX_ELEMENTS = 320
    nodes = [{"ref": n, "role": "button", "label": str(n)} for n in range(300)]
    page = adapter.normalise({"nodes": nodes}, '{"text":"Visible page\\nPrice SGD 376","viaFrames":true}')
    assert page["text"] == "Visible page\nPrice SGD 376"
    ids = {a["id"] for a in page["actions"]}
    assert {"r299", "scroll_down", "scroll_up", "look", "wait"} <= ids


def test_relative_tie_ignores_probability_mass_from_distractors(monkeypatch):
    model = browser_policy.model
    calls = []

    def post(url, key, body):
        calls.append(body)
        answers = {}
        for name, question in body["questions"].items():
            ids = list(question["criteria"])
            probabilities = {key: float(key == ids[0]) for key in ids}
            if name == "click_target":
                probabilities = {key: (0.3 if key == "1" else 0.1 if key == "2" else 0.06) for key in ids}
            answers[name] = {"choice": ids[0], "confidence": probabilities[ids[0]],
                             "probabilities": probabilities}
        return {"answers": answers, "model": "test"}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    page = {"url": "https://example.org", "title": "T", "text": "page", "actions": [
        {"id": f"e{n}", "node": n, "kind": "click", "label": str(n)} for n in range(12)]}
    assert model.choose(page, "task", [], relative_tie=True)["tie_break"] is None
    assert len(calls) == 1
    calls.clear()
    assert model.choose(page, "task", [])["tie_break"] is not None
    assert len(calls) == 2
