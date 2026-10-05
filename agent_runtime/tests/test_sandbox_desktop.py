import importlib.util
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

SNAPSHOT = {"target_id": "42.0", "app": "Chromium", "title": "Wikipedia", "pid": 42, "active": True,
            "frame": {"x": 0, "y": 0, "w": 1000, "h": 800},
            "text": "Welcome to Wikipedia", "elements": [
                {"path": [3, 0, 1], "role": "entry", "label": "Search Wikipedia", "value": "", "actions": ["activate"],
                 "editable": True, "states": ["focused"], "frame": {"x": 10, "y": 20, "w": 100, "h": 30}},
                {"path": [3, 0, 2], "role": "push button", "label": "Search", "value": "", "actions": ["press"],
                 "editable": False, "states": [], "frame": {"x": 120, "y": 20, "w": 40, "h": 30}},
                {"path": [3, 0, 3], "role": "check box", "label": "Remember", "value": "", "actions": ["toggle"],
                 "editable": False, "states": ["checked"], "frame": {"x": 0, "y": 0, "w": 0, "h": 0}}]}


class FakeBox:
    def __init__(self):
        self.calls = []

    def press(self, key):
        self.calls.append(("press", key))

    def move_mouse(self, x, y):
        self.calls.append(("move_mouse", x, y))

    def scroll(self, direction, amount):
        self.calls.append(("scroll", direction, amount))

    def left_click(self, x, y):
        self.calls.append(("left_click", x, y))

    def write(self, text):
        self.calls.append(("write", text))


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "sandbox_desktop_server.py"
    spec = importlib.util.spec_from_file_location("sandbox_desktop_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    box, requests = FakeBox(), []
    module.replies = {}

    async def vm(*args, check=True):
        requests.append(args)
        if args[0] == "windows":
            return [{"target_id": "42.0", "app": "Chromium", "title": "Wikipedia", "active": True}]
        if args[0] == "snapshot":
            return module.snapshots.pop(0) if module.snapshots else SNAPSHOT
        return module.replies.get(args[-2] if args[-2] in ("click", "set_text") else args[-1], {"ok": True})

    async def no_sleep(_):
        pass

    async def activate(snapshot):
        requests.append(("activate", snapshot["title"]))

    monkeypatch.setattr(module, "vm", vm)
    monkeypatch.setattr(module, "activate", activate)
    monkeypatch.setattr(module, "sandbox", lambda: box)
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    module.box, module.requests, module.snapshots = box, requests, []
    module._target = "42.0"
    return module


def test_snapshot_becomes_the_trained_page_shape(adapter):
    page = adapter.normalise(SNAPSHOT)
    assert page["url"] == "app://Chromium/Wikipedia" and page["target_id"] == "42.0"
    assert page["text"] == "Welcome to Wikipedia"
    elements = [(a["id"], a["kind"], a["label"]) for a in page["actions"] if a["node"] is not None]
    assert elements == [("e1", "fill", "Search Wikipedia"), ("e1-open", "click", "Open Search Wikipedia"),
                        ("e2", "click", "Search"), ("e3", "click", "Remember")]
    assert next(a for a in page["actions"] if a["id"] == "e3")["checked"] is True
    assert {a["id"] for a in page["actions"] if a["node"] is None} >= {"scroll_down", "wait", "look", "key_enter"}


async def test_actions_resolve_against_the_latest_snapshot(adapter):
    await adapter.observe()
    await adapter.act("e2")
    await adapter.act("e1", text="Singapore")
    await adapter.act("key_enter")
    await adapter.act("scroll_down")
    acts = [r for r in adapter.requests if r[0] == "act"]
    assert acts[0] == ("act", "42.0", json.dumps([3, 0, 2]), "push button", "Search", "click")
    assert acts[1] == ("act", "42.0", json.dumps([3, 0, 1]), "entry", "Search Wikipedia", "set_text", "Singapore")
    assert adapter.box.calls == [("press", "enter"), ("move_mouse", 500, 400), ("scroll", "down", 5)]


async def test_failed_element_actions_fall_back_to_pointer_and_keyboard(adapter):
    await adapter.observe()
    adapter.replies = {"click": {"ok": False}, "set_text": {"ok": False}}
    await adapter.act("e2")
    await adapter.act("e1", text="Bali")
    assert adapter.box.calls == [("left_click", 140, 35), ("left_click", 60, 35), ("press", ["ctrl", "a"]),
                                 ("write", "Bali")]
    adapter.replies = {"click": {"ok": False}}
    with pytest.raises(ToolError, match="no screen position"):
        await adapter.act("e3")


async def test_stale_or_unknown_elements_are_refused_not_guessed(adapter):
    await adapter.observe()
    adapter.replies = {"click": {"error": "stale", "found": ["push button", "Go"]}}
    with pytest.raises(ToolError, match="changed since the last snapshot"):
        await adapter.act("e2")
    with pytest.raises(ToolError, match="latest window snapshot"):
        await adapter.act("e99")
    with pytest.raises(ToolError, match="not open"):
        await adapter.observe("7.0")


async def test_an_inactive_window_is_raised_before_it_is_read(adapter):
    adapter.snapshots = [{**SNAPSHOT, "active": False, "elements": []}, SNAPSHOT]
    page = await adapter.observe()
    assert ("activate", "Wikipedia") in adapter.requests
    assert page["actions"][0]["id"] == "e1"


def test_numbers_stay_with_their_element_when_others_appear(adapter):
    first = {e["label"]: a["id"] for e, a in zip(SNAPSHOT["elements"], [a for a in adapter.normalise(SNAPSHOT)["actions"]
                                                                       if a["node"] is not None and not a["id"].endswith("-open")])}
    popup = {"path": [3, 0, 9], "role": "push button", "label": "20 October", "value": "", "actions": ["press"],
             "editable": False, "states": [], "frame": {"x": 5, "y": 5, "w": 10, "h": 10}}
    page = adapter.normalise({**SNAPSHOT, "elements": [popup, *SNAPSHOT["elements"]]})
    again = {a["label"]: a["id"] for a in page["actions"] if a["node"] is not None and not a["id"].endswith("-open")}
    assert again["Search"] == first["Search"] and again["Search Wikipedia"] == first["Search Wikipedia"]
    assert again["20 October"] == "e4"
    adapter.normalise({**SNAPSHOT, "target_id": "43.0"})  # another window starts its own numbering
    assert adapter._ids_window == "43.0" and len(adapter._ids) == 3
