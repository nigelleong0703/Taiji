"""Drive the browser through Pilot's MCP tools, normalised into Taiji's decision schema.

Pilot (the local browser agent: extension + bridge daemon + MCP) owns the browser side: tabs, refs
that survive across frames, cross-origin page text, and screenshots with an optional coordinate grid.
This adapter exposes the vocabulary the runtime looks for -- open / observe / act / screenshot /
close -- so the same S1/S2 loop, heads, audit trail and conversation continue to work, and it is the
only place that knows how Pilot names things:

    Pilot snapshot node {ref, role, label, value?, tag}  ->  our {id, node, kind, label, role, value}
    our action id                                        ->  pilot browser_click / browser_type /
                                                             browser_select_option / browser_scroll

Configure the runtime with an MCP config that runs this file; the server name becomes the tool prefix
(browser__open, browser__observe, ...). Set PILOT_MCP_SCRIPT when Pilot is not at the default path.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PILOT_SCRIPT = os.environ.get(
    "PILOT_MCP_SCRIPT", str(Path.home() / "Desktop" / "pilot" / "mcp-server" / "dist" / "index.js"))
PAGE_TEXT_CHARS = 6000      # the harness caps page text at the same size
# One decision question carries 1-255 options by default; Pilot's own snapshot allows 300. Raise this with
# TAIJI_PILOT_MAX_ELEMENTS (and TAIJI_MAX_CRITERIA on S1) to test pages with larger action spaces.
MAX_ELEMENTS = int(os.environ.get("TAIJI_PILOT_MAX_ELEMENTS", "200"))
SCROLL_DELTA = 560          # matches the scroll actions the harness appends
NON_TEXT_INPUTS = {"checkbox", "radio", "button", "submit", "reset", "file", "image", "range", "color"}
TEXT_ROLES = {"textbox", "searchbox", "spinbutton", "combobox"}
META_ACTIONS = (
    {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    {"id": "look", "kind": "look", "label": "Look at the screen (screenshot of the current view)"},
)
KEYS = {"key_enter": "Enter", "key_escape": "Escape", "key_tab": "Tab",
        "key_down": "ArrowDown", "key_up": "ArrowUp"}

_registry = None
_tab_id: int | None = None
_tab_loaded = False
_tab_pending = False
_tab_lock = asyncio.Lock()


def tab_state_path() -> Path | None:
    value = os.environ.get("TAIJI_PILOT_TAB_STATE")
    return Path(value) if value else None


def save_tab_state() -> None:
    path = tab_state_path()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"tabId": _tab_id, "pending": _tab_pending}))
        temporary.replace(path)


async def restore_tab(*, check_pending: bool = True) -> None:
    """Restore only this session's owned tab; never adopt another session's tab."""
    global _tab_id, _tab_loaded, _tab_pending
    if not _tab_loaded:
        path = tab_state_path()
        if path and path.exists():
            state = json.loads(path.read_text())
            _tab_id = state.get("tabId")
            _tab_pending = state.get("pending", False)
        _tab_loaded = True
    if _tab_id is not None:
        listed = json.loads(await call("browser_list_tabs"))
        if not any(tab.get("tabId") == _tab_id for tab in listed["tabs"]):
            _tab_id = None
            save_tab_state()
    if _tab_pending and check_pending:
        raise ToolError("Previous tab creation has an unknown outcome; inspect Pilot tabs before retrying.")


@asynccontextmanager
async def lifespan(_server):
    """One connection to Pilot for the whole session.

    Opening it inside a tool call puts the stdio client's cancel scope in the wrong task, so it is
    opened here, in the server's own task, exactly once.
    """
    global _registry
    from taiji_agent.mcp_registry import MCPRegistry

    wrapper = os.environ.get("TAIJI_PILOT_STDIO_WRAPPER")
    args = [wrapper, PILOT_SCRIPT] if wrapper else [PILOT_SCRIPT]
    _registry = await MCPRegistry.connect({"pilot": {"command": "node", "args": args}})
    try:
        yield {"pilot": _registry}
    finally:
        try:
            await _registry.close()
        except Exception:
            pass
        _registry = None


server = MCPServer("taiji-pilot", lifespan=lifespan)


def pilot():
    if _registry is None:
        raise ToolError("Pilot connection is not ready")
    return _registry


def full_text(result) -> str:
    """The complete text of a tool result: the rendered form is capped and a snapshot must not be."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured)
    return "\n".join(getattr(block, "text", "") for block in (getattr(result, "content", None) or []))


async def call(tool: str, arguments: dict | None = None) -> str:
    result = await pilot().call(f"pilot__{tool}", arguments or {})
    text = full_text(result)
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise ToolError(f"Pilot {tool} failed: {text}")
    return text


def is_editable(node: dict) -> bool:
    """Pilot's snapshot carries tag/role, not the input type; text-ish roles are what we can trust."""
    tag = str(node.get("tag") or "").lower()
    role = str(node.get("role") or "").lower()
    if tag == "textarea" or role in TEXT_ROLES:
        return True
    return tag == "input" and role not in NON_TEXT_INPUTS - {"checkbox", "radio"}


def kind_of(node: dict) -> str:
    if str(node.get("tag") or "").lower() == "select" or str(node.get("role") or "") == "select":
        return "select"
    return "fill" if is_editable(node) else "click"


def action_of(node: dict) -> dict:
    """One Pilot node -> our action. Editable fields get the second, click-only entry the harness uses."""
    ref = int(node["ref"])
    kind = kind_of(node)
    label = str(node.get("label") or node.get("role") or f"element {ref}")
    action = {"id": f"r{ref}", "node": ref, "kind": kind, "label": label,
              "role": node.get("role") or node.get("tag") or ""}
    if node.get("value"):
        action["value"] = str(node["value"])
    if node.get("href"):
        action["href"] = str(node["href"])
    return action


def normalise(snapshot: dict, text: str) -> dict:
    """Pilot's page -> the page the runtime hands to the decision loop."""
    # browser_get_text returns a JSON envelope. Unwrap it before applying the page
    # budget so metadata and JSON escapes do not displace visible page content.
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        text = payload["text"]
    actions = []
    # Pilot returns every rendered interactive element (capped at 300). A decision question takes a
    # bounded number of criteria, so the list is capped here; filtering by viewport instead of truncating
    # needs geometry in the snapshot node, which Pilot does not expose yet.
    for node in (snapshot.get("nodes") or [])[:MAX_ELEMENTS]:
        action = action_of(node)
        actions.append(action)
        if action["kind"] == "fill":
            # The harness offers both "type here" and "click here" for an editable field, as two ids
            # that share one node, so the operation heads keep their usual choice.
            actions.append({**action, "id": f"{action['id']}-open", "kind": "click",
                            "label": f"Open {action['label']}"})
    # Controls belong to the operation head, not the element target head. A wide page must
    # retain navigation even when its target question contains more than 255 elements.
    actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down",
                    "node": None, "delta": SCROLL_DELTA})
    actions.append({"id": "scroll_up", "kind": "scroll", "label": "Scroll up",
                    "node": None, "delta": -SCROLL_DELTA})
    actions.extend({"node": None, **meta} for meta in META_ACTIONS)
    actions.extend({"id": action_id, "kind": "key", "node": None,
                    "label": f"Press {key} on the currently focused control"}
                   for action_id, key in KEYS.items())
    return {"url": snapshot.get("url") or "", "title": snapshot.get("title") or "",
            "text": (text or "")[:PAGE_TEXT_CHARS], "actions": actions}


def tool_response(page: dict):
    if os.environ.get("TAIJI_PI_INLINE_STATE") != "1":
        return page
    # Keep native Pi's text converter below its spill threshold. The entire
    # actionable state stays in structured_content, not an observation file.
    visible = {k: page.get(k) for k in ("tab_id", "url", "title", "text")}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(visible))], structured_content=page)


async def observe_page() -> dict:
    await restore_tab()
    snapshot = json.loads(await call("browser_snapshot", {"tabId": _tab_id} if _tab_id else {}))
    text = await call("browser_get_text", {"tabId": _tab_id} if _tab_id else {})
    return {**normalise(snapshot, text), "tab_id": _tab_id or snapshot.get("tabId")}


async def select_tab(tab_id: int | None) -> None:
    global _tab_id, _tab_pending
    await restore_tab(check_pending=tab_id is None)
    if tab_id is not None:
        listed = json.loads(await call("browser_list_tabs"))
        if not any(tab.get("tabId") == tab_id for tab in listed["tabs"]):
            raise ToolError(f"Tab {tab_id} is not open; list tabs before choosing a target.")
        _tab_id = tab_id
        _tab_pending = False
        save_tab_state()


async def settled_page() -> dict:
    for _ in range(20):
        await asyncio.sleep(0.3)
        page = await observe_page()
        if any(action.get("node") is not None for action in page["actions"]) or page["text"]:
            return page
    return await observe_page()


@server.tool()
async def list_tabs() -> dict:
    """List available tabs and their ids/URLs; choose a tab explicitly for navigation or observation."""
    await restore_tab(check_pending=False)
    result = {**json.loads(await call("browser_list_tabs")), "selected_tab_id": _tab_id,
              "creation_outcome_unknown": _tab_pending}
    if os.environ.get("TAIJI_PI_INLINE_STATE") == "1":
        # Drop repeated window/group metadata, retaining every tab and its URL/title.
        visible = {"tab_columns": ["tabId", "url", "title", "active"],
                   "tabs": [[t.get(k) for k in ("tabId", "url", "title", "active")] for t in result.get("tabs", [])],
                   "selected_tab_id": _tab_id, "creation_outcome_unknown": _tab_pending}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(visible))], structured_content=result)
    return result


@server.tool()
async def new_tab(url: str) -> dict:
    """Explicitly create and select an additional tab. Use navigate to change an existing tab's URL."""
    global _tab_id, _tab_pending
    async with _tab_lock:
        await restore_tab()
        _tab_pending = True
        save_tab_state()
        opened = json.loads(await call("browser_new_tab", {"url": url}))
        tab_id = opened.get("tabId")
        if not isinstance(tab_id, int) or isinstance(tab_id, bool):
            raise ToolError("Pilot returned no tab id; inspect tabs before retrying.")
        _tab_id = tab_id
        _tab_pending = False
        save_tab_state()
        return tool_response(await settled_page())


@server.tool()
async def navigate(tab_id: int, url: str) -> dict:
    """Navigate the chosen existing tab and select it for following local actions. Never creates tabs."""
    async with _tab_lock:
        await select_tab(tab_id)
        await call("browser_navigate", {"url": url, "tabId": tab_id})
        return tool_response(await settled_page())


@server.tool()
async def close_tab(tab_id: int) -> dict:
    """Close the explicitly chosen tab. Other tabs are preserved."""
    global _tab_id
    await restore_tab()
    await call("browser_close_tab", {"tabId": tab_id})
    if _tab_id == tab_id:
        _tab_id = None
        save_tab_state()
    return {"closed_tab_id": tab_id}


@server.tool()
async def open(url: str) -> dict:
    """Navigate in this session's workspace tab; create one only when it has no owned tab."""
    global _tab_id, _tab_pending
    async with _tab_lock:
        await restore_tab()
        if _tab_id is None:
            _tab_pending = True
            save_tab_state()
            opened = json.loads(await call("browser_new_tab", {"url": url}))
            tab_id = opened.get("tabId")
            if not isinstance(tab_id, int) or isinstance(tab_id, bool):
                raise ToolError("Pilot returned no tab id; inspect tabs before retrying.")
            _tab_id = tab_id
            _tab_pending = False
            save_tab_state()
        else:
            # Even navigation errors keep ownership; retries must not spawn another tab.
            await call("browser_navigate", {"url": url, "tabId": _tab_id})
        for _ in range(20):
            await asyncio.sleep(0.3)
            page = await observe_page()
            if any(action.get("node") is not None for action in page["actions"]) or page["text"]:
                return tool_response(page)
        return tool_response(await observe_page())


@server.tool()
async def observe(tab_id: int | None = None) -> dict:
    """Read the current page: elements with the refs Pilot can act on, plus its visible text."""
    await select_tab(tab_id)
    return tool_response(await observe_page())


@server.tool()
async def act(action_id: str, text: str = "", tab_id: int | None = None) -> dict:
    """Perform one action by id and return the page that follows."""
    await select_tab(tab_id)
    arguments = {"tabId": _tab_id} if _tab_id else {}
    if action_id in KEYS:
        await call("browser_key", {"key": KEYS[action_id], **arguments})
    elif action_id in {"wait", "look"}:
        await asyncio.sleep(0.2)
    elif action_id == "scroll_down":
        await call("browser_scroll", {"dy": SCROLL_DELTA, **arguments})
    elif action_id == "scroll_up":
        await call("browser_scroll", {"dy": -SCROLL_DELTA, **arguments})
    else:
        base, _, suffix = action_id.partition("-")
        ref = int(base.removeprefix("r"))
        if suffix == "open":
            await call("browser_click", {"ref": ref, **arguments})
        elif action_id.startswith("r") and suffix == "" and text:
            await call("browser_type", {"ref": ref, "text": text, **arguments})
        elif text:
            await call("browser_type", {"ref": ref, "text": text, **arguments})
        else:
            await call("browser_click", {"ref": ref, **arguments})
    return tool_response(await observe_page())


@server.tool()
async def screenshot() -> dict:
    """Return the viewport as bare base64, the shape the runtime hands to a vision S1."""
    await restore_tab()
    result = await pilot().call("pilot__browser_screenshot", {"tabId": _tab_id} if _tab_id else {})
    for block in getattr(result, "content", None) or []:
        data = getattr(block, "data", None)
        if data:
            return {"screenshot": data}
    return {"screenshot": ""}


@server.tool()
async def screenshot_image(tab_id: int | None = None):
    """Return a native image content block for multimodal agent sessions."""
    from mcp.types import CallToolResult

    await select_tab(tab_id)
    result = await pilot().call("pilot__browser_screenshot", {"tabId": _tab_id} if _tab_id else {})
    images = [block for block in (getattr(result, "content", None) or []) if getattr(block, "data", None)]
    if not images:
        raise ToolError("Pilot returned no screenshot")
    return CallToolResult(content=images)


@server.tool()
async def close() -> dict:
    """Close the tab this adapter opened."""
    global _tab_id
    await restore_tab()
    if _tab_id:
        await call("browser_close_tab", {"tabId": _tab_id})
        _tab_id = None
        save_tab_state()
    return {"closed": True}


if __name__ == "__main__":
    server.run(transport="stdio")
