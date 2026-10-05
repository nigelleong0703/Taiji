"""Drive a Linux desktop in a self-hosted E2B sandbox (Debian/XFCE, Chromium, LibreOffice) for Taiji.

The sandbox is the environment, not the user's machine: every session starts a fresh microVM from
the `desktop-debian-taiji` template (~/Desktop/agent-computer, built with accessibility enabled), so
nothing on the host is touched and tasks start from a clean state. The same environment contract as
the macOS desktop adapter -- observe / act / screenshot_image plus S2-side preparation (list_windows,
focus_app, open_url) -- so S1 sees the trained page shape for any window:

    AT-SPI element {path, role, label, value, actions, editable, frame}   (sandbox_atspi.py, in the VM)
        ->  our {id: "e<n>", node, kind: click|fill, label, role, value}
    our action id  ->  AT-SPI action / set_text (click + type fallback) / key / scroll

Web pages are not read through AT-SPI: Chromium's own UI (address bar, tabs, bookmarks) would sit in
the same list as the page and S1 was trained on DOM snapshots. A relay in the VM (sandbox_cdp_relay.py)
exposes Chromium's DevTools endpoint through the E2B proxy, and the browser harness the training rows
were built with (examples/browser/harness/jev_ultrafast, the same snapshot.js) drives a page window over
it; that page is the target "web". Desktop apps (LibreOffice, Mousepad, terminal) stay on AT-SPI.

Configuration (environment): TAIJI_E2B_ENV, the .env holding E2B_API_KEY / E2B_API_URL /
E2B_SANDBOX_URL (default ~/Desktop/agent-computer/.env); TAIJI_SANDBOX_TEMPLATE (default
desktop-debian-taiji); TAIJI_SANDBOX_ID to attach to a running sandbox instead of creating one;
TAIJI_SANDBOX_KEEP=1 to leave the sandbox running after the session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

HERE = Path(__file__).resolve().parent
PAGE_TEXT_CHARS = 6000
MAX_ELEMENTS = int(os.environ.get("TAIJI_SANDBOX_MAX_ELEMENTS", "200"))
SCREEN = (1364, 1024)
SCRIPT = "/tmp/sandbox_atspi.py"
FILL_ROLES = {"entry", "password text", "text", "spin button", "document text", "editbar", "terminal"}
META_ACTIONS = (
    {"id": "wait", "kind": "wait", "label": "Wait for the window to update"},
    {"id": "look", "kind": "look", "label": "Look at the screen (screenshot of the current window)"},
)
KEYS = {"key_enter": "enter", "key_escape": "escape", "key_tab": "tab", "key_down": "down", "key_up": "up"}
# App names S2 may ask for -> how they are started in the VM and how their AT-SPI app is named.
APPS = {
    "chromium": ("chromium", "chromium"), "browser": ("chromium", "chromium"),
    "mousepad": ("mousepad", "mousepad"), "text editor": ("mousepad", "mousepad"),
    "terminal": ("xfce4-terminal", "xfce4-terminal"),
    "writer": ("libreoffice --writer", "soffice"), "calc": ("libreoffice --calc", "soffice"),
    "impress": ("libreoffice --impress", "soffice"),
}
CHROME = {"xfwm4", "xfce4-panel", "xfdesktop"}  # window manager, dock and wallpaper: not task windows

# The sandbox lives on a short lease renewed while the session acts, so a session killed before its
# cleanup runs leaves a VM behind for at most LEASE_S, not an hour.
LEASE_S, RENEW_S = 900, 120
_renewed = 0.0
_sandbox = None
_bus = ""                              # DBUS_SESSION_BUS_ADDRESS=... of the XFCE session
_target: str | None = None             # target_id ("<pid>.<frame>") of the window S1 acts in
_elements: dict[int, dict] = {}        # element number -> AT-SPI element of the latest snapshot
_frame: dict | None = None             # screen rectangle of the target window in that snapshot
# Stable numbering: an element keeps its number across snapshots of a window while its role, name and
# tree path stay the same, so "e71" still means the same date cell after the calendar redraws, and
# failed-attempt counts and field changes compare like with like.
_ids: dict[tuple, int] = {}
_ids_window: str | None = None
WEB = "web"                            # target id of the CDP-driven page window
WEB_KEYS = {"key_enter": ("Enter", "Enter", 13, "\r"), "key_escape": ("Escape", "Escape", 27, None),
            "key_tab": ("Tab", "Tab", 9, "\t"), "key_down": ("ArrowDown", "ArrowDown", 40, None),
            "key_up": ("ArrowUp", "ArrowUp", 38, None)}
_web = None                            # jev_ultrafast Browser on the sandbox's Chromium
_web_page: dict | None = None
_web_actions: dict[str, dict] = {}


def settings() -> dict:
    path = Path(os.environ.get("TAIJI_E2B_ENV", Path.home() / "Desktop/agent-computer/.env")).expanduser()
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    values.update({k: v for k, v in os.environ.items() if k.startswith("E2B_")})
    for key in ("E2B_API_URL", "E2B_SANDBOX_URL"):
        # The SDK's HTTP client resolves names itself and misses Tailscale MagicDNS: use the address.
        host = urlparse(values[key]).hostname
        values[key] = values[key].replace(host, socket.gethostbyname(host), 1)
    return values


def connect():
    from e2b_desktop import Sandbox

    env = settings()
    conn = {"api_key": env["E2B_API_KEY"], "api_url": env["E2B_API_URL"], "sandbox_url": env["E2B_SANDBOX_URL"]}
    if os.environ.get("TAIJI_SANDBOX_ID"):
        sandbox = Sandbox.connect(os.environ["TAIJI_SANDBOX_ID"], timeout=LEASE_S, **conn)
    else:
        sandbox = Sandbox.create(os.environ.get("TAIJI_SANDBOX_TEMPLATE", "desktop-debian-taiji"),
                                 resolution=SCREEN, timeout=LEASE_S, **conn)
    sandbox.files.write(SCRIPT, (HERE / "sandbox_atspi.py").read_text())
    sandbox.files.write("/tmp/sandbox_cdp_relay.py", (HERE / "sandbox_cdp_relay.py").read_text())
    sandbox.commands.run("python3 /tmp/sandbox_cdp_relay.py >/tmp/cdp-relay.log 2>&1 &", background=True, timeout=30)
    pid = sandbox.commands.run("pgrep -x xfce4-session", timeout=30).stdout.split()[0]
    bus = sandbox.commands.run(f"tr '\\0' '\\n' < /proc/{pid}/environ | grep ^DBUS_SESSION_BUS_ADDRESS=",
                               timeout=30).stdout.strip()
    return sandbox, bus


def connect_devtools(sandbox):
    """Point the browser harness at the sandbox's Chromium (BU_CDP_WS) through the relay and proxy."""
    import time
    import urllib.request

    proxy = urlparse(settings()["E2B_SANDBOX_URL"])
    host = f"9223-{sandbox.sandbox_id}.{proxy.hostname}.nip.io:{proxy.port}"
    for _ in range(40):
        try:
            version = json.loads(urllib.request.urlopen(f"http://{host}/json/version", timeout=5).read())
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("Chromium DevTools did not come up in the sandbox")
    os.environ["BU_CDP_WS"] = f"ws://{host}{urlparse(version['webSocketDebuggerUrl']).path}"
    os.environ["BU_NAME"] = f"taiji-sandbox-{sandbox.sandbox_id}"[:64]


def web_browser(url: str):
    """The harness Browser (imported only once BU_CDP_WS names the sandbox)."""
    sys.path.insert(0, str(HERE.parents[1] / "examples" / "browser" / "harness"))
    from jev_ultrafast.browser import Browser

    return Browser(url)


def stop_web():
    global _web
    if _web is not None:
        try:
            _web.close()
        except Exception:
            pass
        _web = None
    if os.environ.get("BU_NAME", "").startswith("taiji-sandbox-"):
        try:
            from browser_harness.admin import restart_daemon

            restart_daemon(os.environ["BU_NAME"])
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_server):
    global _sandbox, _bus
    _sandbox, _bus = await asyncio.to_thread(connect)
    await asyncio.to_thread(connect_devtools, _sandbox)
    try:
        # A view-only live stream for whoever wants to watch the run; its URL goes to a file.
        info = await sandbox_info()
        path = Path(os.environ.get("TAIJI_SANDBOX_STREAM_FILE", Path.home() / ".cache/taiji-sandbox/stream.url"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{info['stream_url']}\n")
    except Exception as error:
        print(f"stream unavailable: {error}", file=sys.stderr)
    try:
        yield {"sandbox": _sandbox}
    finally:
        await asyncio.to_thread(stop_web)
        if os.environ.get("TAIJI_SANDBOX_KEEP") != "1" and not os.environ.get("TAIJI_SANDBOX_ID"):
            try:
                await asyncio.to_thread(_sandbox.kill)
            except Exception:
                pass
        _sandbox = None


server = MCPServer("taiji-sandbox-desktop", lifespan=lifespan)


def sandbox():
    if _sandbox is None:
        raise ToolError("Sandbox connection is not ready")
    return _sandbox


async def vm(*args: str, check: bool = True):
    """Run sandbox_atspi.py in the VM's desktop session; its JSON reply, or a ToolError."""
    from e2b.sandbox.commands.command_handle import CommandExitException

    command = f"{_bus} DISPLAY=:0 python3 {SCRIPT} " + " ".join(shlex.quote(a) for a in args)
    try:
        result = await asyncio.to_thread(sandbox().commands.run, command, timeout=40)
        return json.loads(result.stdout)
    except CommandExitException as error:
        if not check:
            return json.loads(error.stdout or "{}")
        raise ToolError(f"accessibility {args[0]} failed (exit {error.exit_code}): {(error.stdout or error.stderr)[-500:]}")


def kind_of(element: dict) -> str:
    return "fill" if element["role"] in FILL_ROLES and element.get("editable") else "click"


def normalise(snapshot: dict) -> dict:
    """An AT-SPI window snapshot -> the page the runtime hands to the decision loop."""
    global _elements, _frame, _ids_window
    elements, actions = {}, []
    _frame = snapshot.get("frame")
    if snapshot.get("target_id") != _ids_window:
        _ids.clear()
        _ids_window = snapshot.get("target_id")
    for element in snapshot.get("elements", [])[:MAX_ELEMENTS]:
        key = (element["role"], element["label"], tuple(element["path"]))
        number = _ids.setdefault(key, len(_ids) + 1)
        elements[number] = element
        kind = kind_of(element)
        label = (element["label"] or element["role"] or f"element {number}")[:200]
        action = {"id": f"e{number}", "node": number, "kind": kind, "label": label, "role": element["role"]}
        if element.get("value"):
            action["value"] = element["value"][:500]
        for state in ("checked", "selected", "expanded"):
            if state in element.get("states", []):
                action[state] = True
        actions.append(action)
        if kind == "fill":
            actions.append({**action, "id": f"e{number}-open", "kind": "click", "label": f"Open {label}"})
    _elements = elements
    for direction in ("down", "up"):
        actions.append({"id": f"scroll_{direction}", "kind": "scroll", "label": f"Scroll {direction}", "node": None})
    actions.extend({"node": None, **meta} for meta in META_ACTIONS)
    actions.extend({"id": action_id, "kind": "key", "node": None,
                    "label": f"Press {key} on the currently focused control"} for action_id, key in KEYS.items())
    app, title = snapshot.get("app", ""), snapshot.get("title", "")
    return {"url": f"app://{app}/{title}", "title": title or app, "text": snapshot.get("text", "")[:PAGE_TEXT_CHARS],
            "actions": actions, "target_id": snapshot.get("target_id")}


def tool_response(page: dict):
    if os.environ.get("TAIJI_PI_INLINE_STATE") != "1":
        return page
    visible = {k: page.get(k) for k in ("target_id", "url", "title", "text")}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(visible))], structured_content=page)


async def windows() -> list[dict]:
    return [w for w in await vm("windows") if w["app"] not in CHROME]


async def select_target(target_id: str | None) -> str:
    global _target
    if str(target_id) == WEB:
        if _web is None:
            raise ToolError("No page is open; call open_url first.")
        _target = WEB
    elif target_id is not None and str(target_id) != _target:
        if not any(w["target_id"] == str(target_id) for w in await windows()):
            raise ToolError(f"Window {target_id} is not open; list windows before choosing a target.")
        _target = str(target_id)
    if _target is None:
        raise ToolError("No window selected; call list_windows, focus_app or open_url first.")
    return _target


async def activate(snapshot: dict):
    """Raise the window: Chromium exposes a page's content to AT-SPI only in its active window.
    The sandbox desktop belongs to the agent, so taking focus there disturbs no one."""
    title = snapshot.get("title")
    if title:
        await asyncio.to_thread(sandbox().commands.run,
                                f"xdotool search --onlyvisible --name {shlex.quote('^' + re.escape(title) + '$')} "
                                f"windowactivate --sync >/dev/null 2>&1 || true", timeout=20)
        await asyncio.sleep(0.6)


def web_observe() -> dict:
    """The page in the trained DOM shape (as examples/browser_server.py), plus the key controls."""
    global _web_page, _web_actions
    page = _web.observe(screenshot=False)
    _web_page = page
    _web_actions = {str(a["id"]): a for a in page.get("actions", [])}
    actions = [{k: a[k] for k in ("id", "kind", "label", "role", "node", "value", "current_value",
                                  "checked", "selected", "expanded", "delta", "href") if k in a}
               for a in _web_actions.values()]
    actions.extend({"id": action_id, "kind": "key", "node": None, "label": f"Press {key[0]} on the focused element"}
                   for action_id, key in WEB_KEYS.items())
    return {"url": page.get("url"), "title": page.get("title"), "text": (page.get("text") or "")[:PAGE_TEXT_CHARS],
            "actions": actions, "target_id": WEB}


def web_key(action_id: str):
    key, code, vk, text = WEB_KEYS[action_id]
    # Enter submits only when the keyDown carries its text ("\r"); a bare rawKeyDown does not.
    down = {"type": "keyDown", "key": key, "code": code, "windowsVirtualKeyCode": vk}
    if text:
        down.update(text=text, unmodifiedText=text)
    else:
        down["type"] = "rawKeyDown"
    _web.call("Input.dispatchKeyEvent", **down)
    _web.call("Input.dispatchKeyEvent", type="keyUp", key=key, code=code, windowsVirtualKeyCode=vk)


def web_act(action_id: str, text: str) -> dict:
    from jev_ultrafast.browser import StalePage

    if action_id in WEB_KEYS:
        web_key(action_id)
    else:
        action = _web_actions.get(action_id)
        if action is None:
            raise ToolError(f"{action_id} is not in the latest page observation; observe before acting.")
        if action.get("kind") == "fill" and not text:
            raise ToolError("A fill action requires text")
        if action.get("kind") == "look":
            return web_observe()
        try:
            _web.act(action, _web_page, text=text if action.get("kind") == "fill" else None)
        except StalePage:
            page = web_observe()
            page["stale"] = True
            return page
    _web.settle(quiet=0.15, limit=1.0)
    return web_observe()


async def renew_lease():
    global _renewed
    import time

    if os.environ.get("TAIJI_SANDBOX_ID") or time.monotonic() - _renewed < RENEW_S:
        return
    _renewed = time.monotonic()
    try:
        await asyncio.to_thread(sandbox().set_timeout, LEASE_S)
    except Exception as error:
        print(f"lease renewal failed: {error}", file=sys.stderr)


async def observe_page() -> dict:
    await renew_lease()
    if _target == WEB:
        if _web is None:
            raise ToolError("No page is open; call open_url first.")
        return await asyncio.to_thread(web_observe)
    target = await select_target(None)
    for attempt in range(5):
        snapshot = await vm("snapshot", target)
        if snapshot.get("active") is False and not snapshot.get("error"):
            await activate(snapshot)
            snapshot = await vm("snapshot", target)
        if snapshot.get("elements"):
            return normalise(snapshot)
        await asyncio.sleep(0.4)  # a window that just opened may not have exposed its tree yet
    if snapshot.get("error"):
        raise ToolError(f"Window {target} is gone; list windows before observing.")
    return normalise(snapshot)


async def settled_page(seconds: float = 10) -> dict:
    """The target window once two consecutive snapshots agree: a page whose title is known can
    still be rendering its content into the accessibility tree."""
    page, previous = await observe_page(), None
    for _ in range(int(seconds / 0.7)):
        signature = (len(page["actions"]), page["text"])
        if signature == previous and page["text"]:
            break
        previous = signature
        await asyncio.sleep(0.7)
        page = await observe_page()
    return page


async def wait_for_window(match, seconds: float = 20) -> dict:
    for _ in range(int(seconds / 0.5)):
        found = [w for w in await windows() if match(w)]
        if found:
            return next((w for w in found if w["active"]), found[-1])
        await asyncio.sleep(0.5)
    raise ToolError("No matching window appeared.")


@server.tool()
async def sandbox_info() -> dict:
    """The sandbox id and a view-only live stream URL (for a human watching the run)."""
    box = sandbox()
    try:
        await asyncio.to_thread(box.stream.start, require_auth=True)
    except Exception:
        pass  # already streaming
    # The SDK names the public e2b.app host; self-hosted, the proxy routes <port>-<id>.<ip>.nip.io.
    public = await asyncio.to_thread(box.stream.get_url, auth_key=box.stream.get_auth_key(), view_only=True)
    proxy = urlparse(settings()["E2B_SANDBOX_URL"])
    url = f"http://6080-{box.sandbox_id}.{proxy.hostname}.nip.io:{proxy.port}/vnc.html?{urlparse(public).query}"
    return {"sandbox_id": box.sandbox_id, "stream_url": url}


@server.tool()
async def list_windows() -> dict:
    """List open application windows in the sandbox; target_id is the window to observe or delegate."""
    listed = [{k: w[k] for k in ("target_id", "app", "title", "active")} for w in await windows()
              if w["app"].lower() != "chromium"]  # web pages are the "web" target, read from the DOM
    if _web is not None:
        listed.insert(0, {"target_id": WEB, "app": "Chromium page", "title": (_web_page or {}).get("title", ""),
                          "active": _target == WEB})
    return {"windows": listed}


@server.tool()
async def focus_app(name: str) -> dict:
    """Start (or reuse) an app in the sandbox -- chromium, mousepad, terminal, writer, calc, impress --
    and make its window the target."""
    global _target
    command, app = APPS.get(name.lower(), (name, name))
    if app == "chromium":
        if _web is None:
            return await open_url("https://www.google.com")
        _target = WEB
        return tool_response(await observe_page())
    existing = [w for w in await windows() if w["app"].lower() == app]
    if not existing or app == "soffice":
        await asyncio.to_thread(sandbox().commands.run, f"{command} >/dev/null 2>&1 &", background=True)
        await asyncio.sleep(1.5)
    want = name.lower().split()[-1]
    window = await wait_for_window(lambda w: w["app"].lower() == app and (app != "soffice" or want in w["title"].lower()))
    _target = window["target_id"]
    return tool_response(await observe_page())


@server.tool()
async def open_url(url: str) -> dict:
    """Open a URL in the sandbox's browser and make that page the target "web" (S2 navigation)."""
    global _web, _target
    if not url.startswith(("http://", "https://")):
        raise ToolError("open_url takes an http(s) URL")
    if _web is None:
        _web = await asyncio.to_thread(web_browser, url)
    else:
        def navigate():
            import time

            _web.call("Page.navigate", url=url)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and _web.evaluate("document.readyState") != "complete":
                time.sleep(0.05)
        await asyncio.to_thread(navigate)
    await asyncio.to_thread(_web.settle, 0.3, 3.0)
    _target = WEB
    return tool_response(await observe_page())


@server.tool()
async def observe(target_id: str | None = None) -> dict:
    """Read the target window's accessibility tree as the current page."""
    await select_target(target_id)
    return tool_response(await observe_page())


async def click_center(element: dict):
    frame = element.get("frame") or {}
    if not frame.get("w"):
        raise ToolError(f"{element['label'] or element['role']} has no action and no screen position.")
    await asyncio.to_thread(sandbox().left_click, frame["x"] + frame["w"] // 2, frame["y"] + frame["h"] // 2)


@server.tool()
async def act(action_id: str, text: str = "", target_id: str | None = None) -> dict:
    """Perform one action by id in the target window and return the window state that follows."""
    target = await select_target(target_id)
    if target == WEB:
        return tool_response(await asyncio.to_thread(web_act, action_id, text))
    box = sandbox()
    if action_id in KEYS:
        await asyncio.to_thread(box.press, KEYS[action_id])
    elif action_id in {"wait", "look"}:
        await asyncio.sleep(0.5)
    elif action_id in {"scroll_down", "scroll_up"}:
        # The wheel scrolls whatever is under the pointer: put it over the target window first.
        if _frame and _frame.get("w"):
            await asyncio.to_thread(box.move_mouse, _frame["x"] + _frame["w"] // 2, _frame["y"] + _frame["h"] // 2)
        await asyncio.to_thread(box.scroll, action_id.removeprefix("scroll_"), 5)
    else:
        base, _, suffix = action_id.partition("-")
        element = _elements.get(int(base[1:])) if base.startswith("e") and base[1:].isdigit() else None
        if element is None:
            raise ToolError(f"{action_id} is not in the latest window snapshot; observe before acting.")
        ref = (target, json.dumps(element["path"]), element["role"], element["label"])
        if text and suffix != "open":
            done = await vm("act", *ref, "set_text", text, check=False)
            if done.get("error") == "stale":
                raise ToolError(f"{action_id} changed since the last snapshot (now {done.get('found')}); observe before acting.")
            if not done.get("ok"):
                await click_center(element)
                await asyncio.to_thread(box.press, ["ctrl", "a"])
                await asyncio.to_thread(box.write, text)
        else:
            done = await vm("act", *ref, "click", check=False)
            if done.get("error") == "stale":
                raise ToolError(f"{action_id} changed since the last snapshot (now {done.get('found')}); observe before acting.")
            if not done.get("ok"):
                await click_center(element)
    await asyncio.sleep(0.4)
    return tool_response(await observe_page())


@server.tool()
async def screenshot_image(target_id: str | None = None):
    """Return the sandbox screen as a native image content block."""
    target = await select_target(target_id)
    if target == WEB:
        shot = await asyncio.to_thread(_web.call, "Page.captureScreenshot", format="jpeg", quality=72)
        return CallToolResult(content=[ImageContent(type="image", data=shot["data"], mimeType="image/jpeg")])
    image = await asyncio.to_thread(sandbox().screenshot)
    return CallToolResult(content=[ImageContent(type="image", data=base64.b64encode(bytes(image)).decode(),
                                                mimeType="image/png")])


if __name__ == "__main__":
    server.run(transport="stdio")
