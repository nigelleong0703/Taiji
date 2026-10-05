"""Expose the existing Taiji browser environment as MCP tools for the shared agent runtime."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path
from urllib.parse import urlparse

from mcp.server import MCPServer

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "examples" / "browser" / "harness"
sys.path.insert(0, str(HARNESS))

from jev_ultrafast.browser import Browser  # noqa: E402

server = MCPServer("taiji-browser")
_browser: Browser | None = None
_page: dict | None = None
_actions: dict[str, dict] = {}


def _observe() -> dict:
    global _page, _actions
    if _browser is None:
        raise ValueError("Open a page with browser__open first")
    page = _browser.observe(screenshot=False)
    _page = page
    _actions = {str(action["id"]): action for action in page.get("actions", [])}
    return {
        "url": page.get("url"),
        "title": page.get("title"),
        "text": page.get("text", "")[:12000],
        "actions": [
            {key: action[key] for key in ("id", "kind", "label", "role", "value", "current_value") if key in action}
            for action in _actions.values()
        ],
    }


@server.tool()
def open(url: str) -> dict:
    """Open an HTTP or HTTPS page in the managed browser and return visible text and actions."""
    global _browser, _page
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Browser URLs must use http or https")
    if _browser is not None:
        _browser.close()
    _browser = Browser(url)
    _page = None
    return _observe()


@server.tool()
def observe() -> dict:
    """Read the current browser page and the actions currently available on it."""
    return _observe()


@server.tool()
def act(action_id: str, text: str = "") -> dict:
    """Perform one action by its ID from the latest browser observation; provide text only for fill actions."""
    if _browser is None or _page is None:
        raise ValueError("Open and observe a page before acting")
    action = _actions.get(str(action_id))
    if action is None:
        raise ValueError("That action ID is stale or was not in the latest observation")
    if action.get("kind") == "fill" and not text:
        raise ValueError("A fill action requires text")
    _browser.act(action, _page, text=text if action.get("kind") == "fill" else None)
    return _observe()


@server.tool()
def close() -> dict[str, bool]:
    """Close the managed browser window."""
    global _browser, _page, _actions
    if _browser is not None:
        _browser.close()
    _browser, _page, _actions = None, None, {}
    return {"closed": True}


atexit.register(close)


if __name__ == "__main__":
    server.run(transport="stdio")
