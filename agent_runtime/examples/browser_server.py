"""Expose the existing Taiji browser environment as MCP tools for the shared agent runtime."""

from __future__ import annotations

import atexit
import base64
import io
import sys
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont
from mcp.server import MCPServer

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "examples" / "browser" / "harness"
sys.path.insert(0, str(HARNESS))

from jev_ultrafast.browser import Browser, StalePage  # noqa: E402
from jev_ultrafast.model import action_space  # noqa: E402

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
            {key: action[key] for key in ("id", "kind", "label", "role", "node", "value", "current_value",
                                          "checked", "selected", "expanded", "delta", "href") if key in action}
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
def screenshot() -> dict:
    """Return the viewport as base64 JPEG with every observed element boxed and labelled.

    The text table says what can be chosen; the picture says where it is. Boxing each observed
    element and printing its label joins the two, so a vision S1 can act on what it actually sees
    instead of inferring the layout from a list.
    """
    if _browser is None:
        raise ValueError("Open a page with browser__open first")
    captured = _browser.call("Page.captureScreenshot", format="jpeg", quality=72)
    raw = base64.b64decode(captured["data"])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    viewport = _browser.evaluate("({w: innerWidth, h: innerHeight})") or {"w": image.width, "h": image.height}
    scale_x = image.width / max(1, viewport["w"])
    scale_y = image.height / max(1, viewport["h"])
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    # The number on the box must be the number the decision question offers, so it comes from the
    # harness's own action_space() rather than a second numbering invented here.
    elements, _targets, _controls = action_space(list(_actions.values()))
    ordered = []
    seen_nodes = set()
    for action in _actions.values():
        if action.get("kind") not in {"click", "fill", "select"} or action.get("node") in seen_nodes:
            continue
        seen_nodes.add(action.get("node"))
        ordered.append(action)
    numbered = [(element.get("index"), action) for element, action in zip(elements, ordered)]

    for index, action in numbered:
        rect = action.get("rect") or {}
        if not rect.get("w") or not rect.get("h"):
            continue
        x, y = rect["x"] * scale_x, rect["y"] * scale_y
        w, h = rect["w"] * scale_x, rect["h"] * scale_y
        draw.rectangle([x, y, x + w, y + h], outline=(220, 30, 30), width=2)
        # Dense grids (a calendar) get the bare number; wider controls can afford their label.
        text = f"[{index}]" if (w < 96 or h < 26) else f"[{index}] {(action.get('label') or '')[:20]}"
        box = draw.textbbox((0, 0), text, font=font)
        text_y = max(0, y - (box[3] - box[1]) - 4)
        draw.rectangle([x, text_y, x + (box[2] - box[0]) + 6, text_y + (box[3] - box[1]) + 4],
                       fill=(255, 245, 200))
        draw.text((x + 3, text_y + 2), text, fill=(140, 0, 0), font=font)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=72)
    return {"screenshot": base64.b64encode(out.getvalue()).decode(),
            "labelled": len(numbered), "size": [image.width, image.height]}


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
    try:
        _browser.act(action, _page, text=text if action.get("kind") == "fill" else None)
    except StalePage:
        # The page moved while the model decided. Hand back the current action table instead of failing,
        # so the next decision runs on what is actually on screen (the harness loop does the same).
        observation = _observe()
        observation["stale"] = True
        return observation
    # A date picker or menu renders after the click returns. The harness's own post-input wait only covers
    # combobox autocompletes, so without this the observation misses whatever the click just opened.
    _browser.settle(quiet=0.15, limit=1.0)
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
