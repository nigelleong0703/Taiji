import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "pilot_browser_server.py"
    spec = importlib.util.spec_from_file_location("pilot_tab_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TAIJI_PILOT_TAB_STATE", str(tmp_path / "tab.json"))
    calls, tabs = [], []

    async def call(name, args=None):
        args = args or {}
        calls.append((name, args))
        if name == "browser_new_tab":
            tabs.append({"tabId": max([41, *[t["tabId"] for t in tabs]]) + 1, "url": args["url"]})
            return json.dumps(tabs[-1])
        if name == "browser_list_tabs":
            return json.dumps({"tabs": tabs})
        if name == "browser_navigate":
            next(t for t in tabs if t["tabId"] == args["tabId"])["url"] = args["url"]
        if name == "browser_snapshot":
            tab = next(t for t in tabs if t["tabId"] == args["tabId"])
            return json.dumps({"url": tab["url"], "nodes": []})
        if name == "browser_get_text":
            return json.dumps({"text": "Loaded page"})
        if name == "browser_close_tab":
            tabs[:] = [tab for tab in tabs if tab["tabId"] != args["tabId"]]
        return "{}"

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module, "call", call)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    return module, calls, tabs


async def test_navigation_and_restart_reuse_session_tab(adapter):
    module, calls, _ = adapter
    assert (await module.open("https://example.org/a"))["tab_id"] == 42
    assert (await module.open("https://example.org/b"))["url"].endswith("/b")
    # Simulate a new MCP process restoring the same session's persisted ownership.
    module._tab_id = None
    module._tab_loaded = False
    assert (await module.observe())["tab_id"] == 42
    await module.open("https://example.org/c")
    assert len([c for c in calls if c[0] == "browser_new_tab"]) == 1
    assert [c[1]["tabId"] for c in calls if c[0] == "browser_navigate"] == [42, 42]


async def test_navigation_failure_never_creates_extra_tab(adapter, monkeypatch):
    module, calls, _ = adapter
    await module.open("https://example.org/a")
    original = module.call

    async def fail(name, args=None):
        if name == "browser_navigate":
            raise RuntimeError("Navigation timeout")
        return await original(name, args)

    monkeypatch.setattr(module, "call", fail)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="timeout"):
            await module.open("https://example.org/b")
    assert module._tab_id == 42
    assert len([c for c in calls if c[0] == "browser_new_tab"]) == 1


async def test_closed_tab_recreated_once_and_close_clears_state(adapter):
    module, calls, tabs = adapter
    await module.open("https://example.org/a")
    tabs.clear()
    await module.open("https://example.org/b")
    await module.open("https://example.org/c")
    assert len([c for c in calls if c[0] == "browser_new_tab"]) == 2
    await module.close()
    assert json.loads(module.tab_state_path().read_text())["tabId"] is None


async def test_unknown_creation_outcome_survives_restart(adapter, monkeypatch):
    module, _, _ = adapter
    creations = []
    original = module.call

    async def fail(name, args=None):
        if name == "browser_new_tab":
            creations.append(args)
            raise RuntimeError("Lost connection after send")
        return await original(name, args)

    monkeypatch.setattr(module, "call", fail)
    with pytest.raises(RuntimeError, match="Lost connection"):
        await module.open("https://example.org/a")
    module._tab_loaded = False
    module._tab_pending = False
    with pytest.raises(ToolError, match="unknown outcome"):
        await module.open("https://example.org/a")
    assert len(creations) == 1


async def test_new_session_does_not_adopt_other_sessions_tab(adapter, monkeypatch, tmp_path):
    module, calls, _ = adapter
    await module.open("https://example.org/a")
    monkeypatch.setenv("TAIJI_PILOT_TAB_STATE", str(tmp_path / "other-session.json"))
    module._tab_id = None
    module._tab_loaded = False
    await module.open("https://example.org/b")
    assert len([c for c in calls if c[0] == "browser_new_tab"]) == 2


async def test_explicit_tab_navigation_never_creates_and_selection_persists(adapter):
    module, calls, tabs = adapter
    tabs.extend([{"tabId": 10, "url": "https://example.org/a"},
                 {"tabId": 20, "url": "https://example.org/b"}])
    assert len((await module.list_tabs())["tabs"]) == 2
    await module.navigate(20, "https://example.org/c")
    assert (await module.observe())["url"] == "https://example.org/c"
    assert ("browser_navigate", {"tabId": 20, "url": "https://example.org/c"}) in calls
    await module.observe(tab_id=10)
    assert json.loads(module.tab_state_path().read_text())["tabId"] == 10
    assert not any(name == "browser_new_tab" for name, _ in calls)
    with pytest.raises(ToolError, match="not open"):
        await module.navigate(999, "https://example.org/d")
    assert module._tab_id == 10


async def test_agent_can_explicitly_create_another_tab(adapter):
    module, calls, tabs = adapter
    first = await module.new_tab("https://example.org/a")
    second = await module.new_tab("https://example.org/b")
    assert first["tab_id"] != second["tab_id"]
    assert len([c for c in calls if c[0] == "browser_new_tab"]) == 2
    await module.close_tab(second["tab_id"])
    assert tabs == [{"tabId": first["tab_id"], "url": "https://example.org/a"}]


async def test_pending_creation_can_be_inspected_and_selected_by_agent(adapter):
    module, _, tabs = adapter
    module._tab_pending = True
    tabs.append({"tabId": 10, "url": "https://example.org/a"})
    assert (await module.list_tabs())["creation_outcome_unknown"]
    await module.observe(tab_id=10)
    assert not module._tab_pending


def test_inline_page_keeps_full_actions_without_observation_files(adapter, monkeypatch, tmp_path):
    module, _, _ = adapter
    monkeypatch.setenv("TAIJI_PI_INLINE_STATE", "1")
    page = {"url": "https://example.test", "text": "Price 56.88; stock 6", "actions": [{"id": "r1", "value": "Bali"}] * 300}
    before = set(tmp_path.iterdir())
    result = module.tool_response(page)
    assert result.structured_content == page
    assert "Price 56.88" in result.content[0].text
    assert "actions" not in json.loads(result.content[0].text)
    assert set(tmp_path.iterdir()) == before


async def test_inline_inventory_keeps_all_tabs_and_is_compact(adapter, monkeypatch):
    module, _, tabs = adapter
    monkeypatch.setenv("TAIJI_PI_INLINE_STATE", "1")
    tabs.extend({"tabId": i, "url": f"https://example.test/{i}", "title": "Page", "active": False, "windowId": 7, "pilotGroup": "workspace"} for i in range(300))
    result = await module.list_tabs()
    shown = json.loads(result.content[0].text)
    assert len(shown["tabs"]) == 300
    assert result.structured_content["tabs"] == tabs
    assert "windowId" not in result.content[0].text


@pytest.mark.parametrize("flag", ["is_error", "isError"])
async def test_pilot_error_is_not_discarded_as_a_successful_observation(adapter, monkeypatch, flag):
    from types import SimpleNamespace
    module, _, _ = adapter
    spec = importlib.util.spec_from_file_location("pilot_error_call", Path(module.__file__))
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    class Registry:
        async def call(self, name, args):
            return SimpleNamespace(**{flag: True}, content=[SimpleNamespace(text="stale ref")])
    raw._registry = Registry()
    with pytest.raises(ToolError, match="Pilot browser_click failed: stale ref"):
        await raw.call("browser_click", {"ref": 34})
