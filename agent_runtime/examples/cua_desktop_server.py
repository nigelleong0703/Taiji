"""Drive macOS desktop apps through cua-driver's MCP tools, normalised into Taiji's decision schema.

cua-driver (MIT, https://github.com/trycua/cua) owns the desktop side: windows, the accessibility
tree, per-snapshot element tokens, and background AX actions that do not steal focus. This adapter
exposes the same environment contract as the browser adapter -- observe / act / screenshot_image plus
S2-side preparation (list_windows, focus_app) -- so S1 sees the trained page shape for a window:

    cua element {element_index, element_token, role, label, value, actions, frame}
        ->  our {id: "e<index>", node, kind: click|fill, label, role, value}
    our action id  ->  cua click / set_value (type_text fallback) / press_key / scroll

The server name in the MCP config becomes the environment name (mcp__desktop__observe, ...).
Install with `uv pip install 'cua-driver>=0.33,<0.34'` and grant permissions once with
`cua-driver permissions grant`. Set TAIJI_CUA_DRIVER to use another cua-driver binary.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE_TEXT_CHARS = 6000      # same page text budget as the browser adapter
MAX_ELEMENTS = int(os.environ.get("TAIJI_DESKTOP_MAX_ELEMENTS", "200"))
TEXT_ROLES = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXSecureTextField"}
CLICK_ROLES = {"AXButton", "AXCheckBox", "AXRadioButton", "AXMenuItem", "AXMenuBarItem", "AXMenuButton",
               "AXPopUpButton", "AXLink", "AXTab", "AXDisclosureTriangle", "AXSegment", "AXCell", "AXRow",
               "AXIncrementor", "AXSlider", "AXColorWell", "AXDockItem"}
PRESS_ACTIONS = {"AXPress", "AXPick", "AXOpen", "AXConfirm", "press", "pick", "open", "confirm"}
META_ACTIONS = (
    {"id": "wait", "kind": "wait", "label": "Wait for the window to update"},
    {"id": "look", "kind": "look", "label": "Look at the screen (screenshot of the current window)"},
)
# One tree_markdown row: indent, optional [index], role, optional "title", optional = "value", optional (description).
TREE_LINE = re.compile(r'^\s*- (\[\d+\] )?(AX\w+)(?: "([^"]*)")?(?: = "([^"]*)")?(?: \(([^)]*)\))?')
KEYS = {"key_enter": "return", "key_escape": "escape", "key_tab": "tab", "key_down": "down", "key_up": "up"}

_registry = None
_target: dict | None = None         # {"pid", "window_id"} of the window S1 acts in
_tokens: dict[int, str] = {}        # element_index -> element_token of the latest snapshot


def cua_binary() -> str:
    if os.environ.get("TAIJI_CUA_DRIVER"):
        return os.environ["TAIJI_CUA_DRIVER"]
    from cua_driver import get_binary_path
    return str(get_binary_path())


@asynccontextmanager
async def lifespan(_server):
    """One cua-driver MCP connection per session, opened in the server's own task (see the Pilot adapter)."""
    global _registry
    from taiji_agent.mcp_registry import MCPRegistry

    _registry = await MCPRegistry.connect({"cua": {"command": cua_binary(), "args": ["mcp"]}})
    try:
        yield {"cua": _registry}
    finally:
        try:
            await _registry.close()
        except Exception:
            pass
        _registry = None


server = MCPServer("taiji-desktop", lifespan=lifespan)


def cua():
    if _registry is None:
        raise ToolError("cua-driver connection is not ready")
    return _registry


def payload(result) -> dict:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    text = "\n".join(getattr(block, "text", "") for block in (getattr(result, "content", None) or []))
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except ValueError:
        return {"text": text}


async def call(tool: str, arguments: dict | None = None) -> dict:
    result = await cua().call(f"cua__{tool}", arguments or {})
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise ToolError(f"cua-driver {tool} failed: {json.dumps(payload(result))[:2000]}")
    return payload(result)


def kind_of(element: dict) -> str | None:
    role = str(element.get("role") or "")
    if role in TEXT_ROLES:
        return "fill"
    if role in CLICK_ROLES or PRESS_ACTIONS & set(element.get("actions") or []):
        return "click"
    return None


def window_text(markdown: str) -> list[str]:
    """Readable, non-actionable rows of the window (e.g. a display value). Structured elements list only
    actionable rows, so text comes from the markdown rendering; the app menu bar is not window content."""
    lines, menu_depth = [], None
    for row in markdown.splitlines():
        match = TREE_LINE.match(row)
        if not match:
            continue
        depth = len(row) - len(row.lstrip())
        if menu_depth is not None and depth > menu_depth:
            continue  # inside the menu bar, which the tree may list before or after the window
        menu_depth = None
        indexed, role, title, value, description = match.groups()
        if role == "AXMenuBar":
            menu_depth = depth
            continue
        text = ": ".join(part for part in (description or title, value) if part)
        if not indexed and text and text not in lines:
            lines.append(text)
    return lines


def menu_bar(elements: list[dict]) -> set[int]:
    """Indices in the app menu bar subtree: menus are app chrome, reachable by S2, not window actions."""
    parents = {e.get("element_index"): e.get("parent_index") for e in elements}
    roots = {e.get("element_index") for e in elements if e.get("role") == "AXMenuBar"}
    inside = set()
    for index in parents:
        node, seen = index, set()
        while node is not None and node not in seen:
            if node in roots:
                inside.add(index)
                break
            seen.add(node)
            node = parents.get(node)
    return inside


def normalise(state: dict, window_id: int) -> dict:
    """A cua window snapshot -> the page the runtime hands to the decision loop."""
    global _tokens
    tokens, actions = {}, []
    lines = window_text(state.get("tree_markdown") or "")
    elements = state.get("elements") or []
    chrome = menu_bar(elements)
    for element in elements:
        index, token = element.get("element_index"), element.get("element_token")
        if index is None or index in chrome:
            continue
        if token:
            tokens[int(index)] = token
        role = str(element.get("role") or "")
        label = str(element.get("label") or "").strip()
        value = element.get("value")
        kind = kind_of(element)
        frame = element.get("frame") or {}
        if kind is None or not token:
            text = label or (str(value).strip() if value not in (None, "") else "")
            if not state.get("tree_markdown") and text and text not in lines:
                lines.append(text)  # older drivers: no markdown rendering to read text from
            continue
        if frame.get("h", 2) <= 1 or len(actions) >= MAX_ELEMENTS:
            continue  # virtualised off-viewport rows report 1px frames
        action = {"id": f"e{index}", "node": int(index), "kind": kind,
                  "label": (label or role.removeprefix("AX") or f"element {index}")[:200], "role": role}
        if value not in (None, ""):
            action["value"] = str(value)[:500]
        actions.append(action)
        if kind == "fill":
            # As in the browser adapter: an editable field is offered as "type here" and "click here".
            actions.append({**action, "id": f"{action['id']}-open", "kind": "click", "label": f"Open {action['label']}"})
    _tokens = tokens
    for direction in ("down", "up"):
        actions.append({"id": f"scroll_{direction}", "kind": "scroll", "label": f"Scroll {direction}", "node": None})
    actions.extend({"node": None, **meta} for meta in META_ACTIONS)
    actions.extend({"id": action_id, "kind": "key", "node": None,
                    "label": f"Press {key} on the currently focused control"} for action_id, key in KEYS.items())
    app, title = state.get("app_name") or "", state.get("window_title") or ""
    return {"url": f"app://{app}/{title}", "title": title or app, "text": "\n".join(lines)[:PAGE_TEXT_CHARS],
            "actions": actions, "target_id": window_id}


def tool_response(page: dict):
    if os.environ.get("TAIJI_PI_INLINE_STATE") != "1":
        return page
    visible = {k: page.get(k) for k in ("target_id", "url", "title", "text")}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(visible))], structured_content=page)


async def windows() -> list[dict]:
    listed = await call("list_windows", {"on_screen_only": False})
    return [{"target_id": w.get("window_id"), "pid": w.get("pid"), "app": w.get("app_name"), "title": w.get("title"),
             "on_screen": w.get("is_on_screen")} for w in listed.get("windows", [])]


async def select_target(target_id: int | None) -> dict:
    global _target
    if target_id is not None and (_target is None or _target["window_id"] != target_id):
        window = next((w for w in await windows() if w["target_id"] == target_id), None)
        if window is None:
            raise ToolError(f"Window {target_id} is not open; list windows before choosing a target.")
        _target = {"pid": window["pid"], "window_id": target_id}
    if _target is None:
        raise ToolError("No desktop window selected; call list_windows or focus_app first.")
    return _target


async def observe_page() -> dict:
    """A window snapshot. An unresolved accessibility surface (e.g. a window still settling or on another
    Space) is retried, then refused: an empty tree is not an observation of the window."""
    target = await select_target(None)
    for attempt in range(6):
        state = await call("get_window_state", {**target, "include_screenshot": False})
        if not (state.get("degraded") and not state.get("elements")):
            return normalise(state, target["window_id"])
        if attempt == 2:
            await call("bring_to_front", target)
        await asyncio.sleep(0.3)
    raise ToolError(f"Window {target['window_id']} has no readable accessibility tree: {state.get('degraded_reason')}")


@server.tool()
async def list_windows() -> dict:
    """List open application windows; target_id is the window id to observe or delegate."""
    return {"windows": await windows()}


@server.tool()
async def focus_app(name: str) -> dict:
    """Launch or reuse an app (in the background) and make its frontmost window the target."""
    global _target
    await call("launch_app", {"name": name})
    for _ in range(20):
        found = [w for w in await windows() if str(w["app"]).lower() == name.lower()]
        if found:
            window = next((w for w in found if w["on_screen"]), found[0])
            _target = {"pid": window["pid"], "window_id": window["target_id"]}
            # A background launch can leave the window off the current Space, unreadable by AX.
            await call("bring_to_front", _target)
            return tool_response(await observe_page())
        await asyncio.sleep(0.3)
    raise ToolError(f"{name} has no window; open one before observing it.")


@server.tool()
async def observe(target_id: int | None = None) -> dict:
    """Read the target window's accessibility tree as the current page."""
    await select_target(target_id)
    return tool_response(await observe_page())


@server.tool()
async def act(action_id: str, text: str = "", target_id: int | None = None) -> dict:
    """Perform one action by id in the target window and return the window state that follows."""
    target = await select_target(target_id)
    if action_id in KEYS:
        await call("press_key", {**target, "key": KEYS[action_id]})
    elif action_id in {"wait", "look"}:
        await asyncio.sleep(0.2)
    elif action_id in {"scroll_down", "scroll_up"}:
        await call("scroll", {**target, "direction": action_id.removeprefix("scroll_"), "by": "page", "amount": 1})
    else:
        base, _, suffix = action_id.partition("-")
        token = _tokens.get(int(base.removeprefix("e"))) if base.startswith("e") and base[1:].isdigit() else None
        if token is None:
            raise ToolError(f"{action_id} is not in the latest window snapshot; observe before acting.")
        if text and suffix != "open":
            try:
                await call("set_value", {"pid": target["pid"], "element_token": token, "value": text})
            except ToolError:
                await call("type_text", {"pid": target["pid"], "element_token": token, "text": text})
        else:
            await call("click", {"pid": target["pid"], "element_token": token})
    await asyncio.sleep(0.3)
    return tool_response(await observe_page())


@server.tool()
async def screenshot_image(target_id: int | None = None):
    """Return the target window as a native image content block."""
    target = await select_target(target_id)
    result = await cua().call("cua__get_window_state", {**target, "include_accessibility_tree": False})
    images = [block for block in (getattr(result, "content", None) or []) if getattr(block, "data", None)]
    if not images:
        raise ToolError("cua-driver returned no screenshot")
    return CallToolResult(content=images)


if __name__ == "__main__":
    server.run(transport="stdio")
