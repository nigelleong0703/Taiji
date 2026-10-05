import importlib.util
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

TREE = {"app_name": "TextEdit", "window_title": "Untitled", "elements": [
    {"element_index": 1, "element_token": "s0000000a:1", "role": "AXStaticText", "label": "Untitled — Edited"},
    {"element_index": 2, "element_token": "s0000000a:2", "role": "AXTextArea", "label": "", "value": "draft"},
    {"element_index": 3, "element_token": "s0000000a:3", "role": "AXButton", "label": "Close", "actions": ["AXPress"]},
    {"element_index": 4, "element_token": "s0000000a:4", "role": "AXGroup", "label": "", "actions": ["AXPress"]},
    {"element_index": 5, "element_token": "s0000000a:5", "role": "AXRow", "label": "hidden", "frame": {"h": 1}},
]}


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "cua_desktop_server.py"
    spec = importlib.util.spec_from_file_location("cua_desktop_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    async def call(name, args=None):
        calls.append((name, args or {}))
        if name == "list_windows":
            return {"windows": [{"window_id": 77, "pid": 501, "app_name": "TextEdit", "title": "Untitled", "is_on_screen": True}]}
        if name == "get_window_state":
            return module.state_override.pop(0) if module.state_override else TREE
        if name == "set_value" and args["value"] == "refuse":
            raise ToolError("cua-driver set_value failed")
        return {}

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module, "call", call)
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    module.calls = calls
    module.state_override = []
    return module


def test_window_snapshot_becomes_the_trained_page_shape(adapter):
    page = adapter.normalise(TREE, 77)
    assert page["url"] == "app://TextEdit/Untitled" and page["target_id"] == 77
    assert page["text"] == "Untitled — Edited"
    elements = [(a["id"], a["kind"], a["label"]) for a in page["actions"] if a["node"] is not None]
    assert elements == [("e2", "fill", "TextArea"), ("e2-open", "click", "Open TextArea"),
                        ("e3", "click", "Close"), ("e4", "click", "Group")]
    assert {a["id"] for a in page["actions"] if a["node"] is None} >= {"scroll_down", "wait", "look", "key_enter"}
    assert page["actions"][0]["value"] == "draft"


async def test_actions_use_tokens_from_the_latest_snapshot(adapter):
    await adapter.observe(77)
    await adapter.act("e3")
    await adapter.act("e2", text="hello")
    await adapter.act("e2", text="refuse")
    await adapter.act("key_enter")
    sent = [(name, args) for name, args in adapter.calls if name not in {"list_windows", "get_window_state"}]
    assert sent == [
        ("click", {"pid": 501, "element_token": "s0000000a:3"}),
        ("set_value", {"pid": 501, "element_token": "s0000000a:2", "value": "hello"}),
        ("set_value", {"pid": 501, "element_token": "s0000000a:2", "value": "refuse"}),
        ("type_text", {"pid": 501, "element_token": "s0000000a:2", "text": "refuse"}),
        ("press_key", {"pid": 501, "window_id": 77, "key": "return"}),
    ]


async def test_unknown_or_stale_ids_are_refused_not_guessed(adapter):
    await adapter.observe(77)
    with pytest.raises(ToolError, match="latest window snapshot"):
        await adapter.act("e99")
    with pytest.raises(ToolError, match="not open"):
        await adapter.observe(12)


async def test_unresolved_window_is_retried_then_refused_instead_of_observed_empty(adapter):
    unresolved = {"degraded": True, "degraded_reason": "ax_window_unresolved", "elements": []}
    adapter.state_override = [unresolved, unresolved]
    page = await adapter.observe(77)
    assert page["actions"][0]["id"] == "e2"
    adapter.state_override = [unresolved] * 6
    with pytest.raises(ToolError, match="no readable accessibility tree"):
        await adapter.observe()
    assert ("bring_to_front", {"pid": 501, "window_id": 77}) in adapter.calls


def test_window_text_and_menu_bar_are_separated_from_actions(adapter):
    state = {"app_name": "Calculator", "window_title": "Calculator", "tree_markdown": "\n".join([
        '- [0] AXWindow "Calculator" [id=main actions=[raise]]',
        '    - AXStaticText = "12 × 7" (Last Expression)',
        '    - AXStaticText = "84" (Edit field)',
        '    - [1] AXButton (Equals) [id=Equals actions=[press]]',
        '- [2] AXMenuBar [actions=[cancel]]',
        '  - [3] AXMenuBarItem "Apple" [actions=[press]]']),
        "elements": [
            {"element_index": 0, "element_token": "s0000000b:0", "role": "AXWindow", "label": "Calculator"},
            {"element_index": 1, "element_token": "s0000000b:1", "role": "AXButton", "label": "Equals", "parent_index": 0},
            {"element_index": 2, "element_token": "s0000000b:2", "role": "AXMenuBar", "actions": ["AXCancel"]},
            {"element_index": 3, "element_token": "s0000000b:3", "role": "AXMenuBarItem", "label": "Apple", "parent_index": 2}]}
    page = adapter.normalise(state, 9)
    assert page["text"] == "Last Expression: 12 × 7\nEdit field: 84"
    assert [a["label"] for a in page["actions"] if a["node"] is not None] == ["Equals"]
    # The tree may list the menu bar first (window in another Space or not frontmost).
    lines = state["tree_markdown"].splitlines()
    first = {**state, "tree_markdown": "\n".join([*lines[4:], *lines[:4]])}
    assert adapter.normalise(first, 9)["text"] == "Last Expression: 12 × 7\nEdit field: 84"
