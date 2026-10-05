"""Runs inside the E2B desktop sandbox: the AT-SPI accessibility tree as JSON, and element actions.

  python3 sandbox_atspi.py windows                 -> [{target_id, app, title, pid, active}]
  python3 sandbox_atspi.py snapshot [target_id]    -> {target_id, app, title, text, elements[...]}
  python3 sandbox_atspi.py act <target_id> <path> <role> <name> <action> [text]

Elements carry their child-index path from the application root; `act` re-resolves the path and
refuses (exit 3) when the role or name there no longer matches, so a stale index is never guessed.
Uploaded by agent_runtime/examples/sandbox_desktop_server.py; standard library + gi.Atspi only.
"""
import json
import re
import sys
import time

import gi

gi.require_version("Atspi", "2.0")
from gi.repository import Atspi  # noqa: E402

INTERACTIVE = {"push button", "toggle button", "check box", "radio button", "menu item", "check menu item",
               "radio menu item", "menu", "combo box", "entry", "password text", "text", "spin button", "slider",
               "link", "page tab", "list item", "tree item", "table cell", "icon", "tool bar button", "toggle",
               "button", "switch", "editbar", "paragraph", "document text", "terminal", "heading", "image"}
# Leaf text only: containers repeat their children's text (with U+FFFC where each child sits).
TEXTUAL = {"label", "static", "heading", "paragraph", "status bar", "text", "document text", "caption",
           "description term", "description value", "terminal"}
EMBEDDED = "\ufffc"
BLOCK = {"heading", "paragraph", "label", "status bar", "caption", "document text", "terminal"}
SCREEN_W, SCREEN_H = 1364, 1024
EDITABLE = {"entry", "password text", "text", "spin button", "document text", "editbar", "paragraph", "terminal"}
# Actions Chromium attaches to almost every node (text runs included): they do not make a control.
NOISE_ACTIONS = {"showContextMenu", "click-ancestor", "scrollToMakeVisible", "scrollToPoint", "setSelection", ""}
MAX_ELEMENTS, MAX_TEXT, MAX_DEPTH, MAX_NODES = 250, 6000, 40, 6000
# Every node is a D-Bus round trip; a spreadsheet exposes millions of cells. Containers are read up
# to MAX_CHILDREN children (cells start top-left, where the view is) and a snapshot stops at BUDGET_S.
MAX_CHILDREN, BUDGET_S = 300, 6.0


def role(node):
    try:
        return node.get_role_name()
    except Exception:
        return ""


def name(node):
    try:
        return (node.get_name() or "").strip()
    except Exception:
        return ""


def states(node):
    try:
        s = node.get_state_set()
    except Exception:
        return set()
    out = set()
    for flag, label in ((Atspi.StateType.FOCUSED, "focused"), (Atspi.StateType.CHECKED, "checked"),
                        (Atspi.StateType.SELECTED, "selected"), (Atspi.StateType.EXPANDED, "expanded"),
                        (Atspi.StateType.ENABLED, "enabled"), (Atspi.StateType.SHOWING, "showing"),
                        (Atspi.StateType.EDITABLE, "editable"), (Atspi.StateType.ACTIVE, "active"),
                        (Atspi.StateType.FOCUSABLE, "focusable"), (Atspi.StateType.SENSITIVE, "sensitive")):
        if s.contains(flag):
            out.add(label)
    return out


def actions(node):
    try:
        action = node.get_action_iface()
        return [action.get_action_name(i) for i in range(action.get_n_actions())] if action else []
    except Exception:
        return []


def text_of(node, limit=400):
    """Text or value of a node. The interface methods are class functions taking the node
    (Atspi.Text.get_text(node, ...)); called on the interface object they raise TypeError."""
    try:
        if node.get_text_iface() is not None:
            n = Atspi.Text.get_character_count(node)
            return (Atspi.Text.get_text(node, 0, min(n, limit)) or "").replace(EMBEDDED, "").strip()
    except Exception:
        pass
    try:
        if node.get_value_iface() is not None:
            return f"{Atspi.Value.get_current_value(node):g}"
    except Exception:
        pass
    return ""


def extents(node):
    try:
        e = node.get_extents(Atspi.CoordType.SCREEN)
        return {"x": e.x, "y": e.y, "w": e.width, "h": e.height}
    except Exception:
        return None


def children(node, limit=MAX_CHILDREN):
    try:
        return [node.get_child_at_index(i) for i in range(min(node.get_child_count(), limit))]
    except Exception:
        return []


def frames():
    desktop = Atspi.get_desktop(0)
    out = []
    for ai, app in enumerate(children(desktop, 1000)):
        if app is None:
            continue
        for fi, frame in enumerate(children(app)):
            if frame is None or role(frame) not in {"frame", "window", "dialog", "alert", "file chooser"}:
                continue
            st = states(frame)
            try:
                pid = app.get_process_id()
            except Exception:
                pid = None
            # pid-based, so another app starting or closing does not renumber this window
            out.append({"target_id": f"{pid}.{fi}", "app": name(app), "title": name(frame), "pid": pid,
                        "active": "active" in st, "node": frame, "path": [ai, fi]})
    return out


def resolve(path):
    node = Atspi.get_desktop(0)
    for index in path:
        node = node.get_child_at_index(index)
        if node is None:
            return None
    return node


def pick(target_id):
    windows = frames()
    if target_id:
        return next((w for w in windows if w["target_id"] == target_id), None)
    active = [w for w in windows if w["active"]]
    return (active or windows or [None])[-1]


def snapshot(target_id):
    window = pick(target_id)
    if window is None:
        return {"error": "no window"}
    elements, text, seen = [], [], 0
    stack = [(window["node"], window["path"], 0, False)]
    deadline = time.monotonic() + BUDGET_S
    while stack and seen < MAX_NODES and time.monotonic() < deadline:
        node, path, depth, in_page = stack.pop()
        seen += 1
        r, st = role(node), states(node)
        # Web content inside Chromium carries no SHOWING state; there the screen position decides.
        if not in_page and "showing" not in st and depth > 1:
            continue
        in_page = in_page or r == "document web"
        label, value = name(node), text_of(node)
        acts = [a for a in actions(node) if a not in NOISE_ACTIONS]
        box = extents(node)
        visible = box is None or (box["w"] > 1 and box["h"] > 1)
        if in_page and box is not None:
            visible = visible and box["y"] < SCREEN_H and box["y"] + box["h"] > 0 and box["x"] < SCREEN_W
            if not visible and r != "document web":
                continue
        editable = r in EDITABLE and "editable" in st
        interactive = visible and (editable or bool(acts) and r not in TEXTUAL
                                   or r in INTERACTIVE and r not in TEXTUAL and "focusable" in st)
        if interactive and len(elements) < MAX_ELEMENTS and (label or value or editable):
            elements.append({"path": path, "role": r, "label": label, "value": value if value != label else "",
                             "actions": acts, "editable": editable,
                             "states": sorted(st & {"focused", "checked", "selected", "expanded"}),
                             "enabled": "sensitive" in st or "enabled" in st, "frame": box})
        if visible and not interactive and r in TEXTUAL:
            try:
                container = node.get_child_count() > 0
            except Exception:
                container = False
            if container:
                # the children carry this node's text (it shows U+FFFC where each child sits)
                if r in BLOCK and text:
                    text.append("\n")
            elif value or label:
                piece = value or label
                if not text or piece != text[-1]:
                    text.append(("\n" if r in BLOCK and text else "") + piece)
        if depth < MAX_DEPTH:
            kids = children(node)
            for i in range(len(kids) - 1, -1, -1):
                if kids[i] is not None:
                    stack.append((kids[i], path + [i], depth + 1, in_page))
    return {"target_id": window["target_id"], "app": window["app"], "title": window["title"],
            "pid": window["pid"], "active": window["active"], "frame": extents(window["node"]), "text": re.sub(r"[ \t]*\n[ \t\n]*", "\n", re.sub(r" {2,}", " ", " ".join(text))).strip()[:MAX_TEXT], "elements": elements, "truncated": bool(stack)}


def act(target_id, path, expected_role, expected_name, action, text=""):
    node = resolve(json.loads(path))
    if node is None or role(node) != expected_role or name(node) != expected_name:
        print(json.dumps({"error": "stale", "found": [role(node), name(node)] if node else None}))
        sys.exit(3)
    if action == "set_text":
        if node.get_editable_text_iface() is not None and Atspi.EditableText.set_text_contents(node, text):
            print(json.dumps({"ok": True, "via": "editable_text"}))
            return
        print(json.dumps({"ok": False, "via": "editable_text"}))
        sys.exit(4)
    if action == "focus":
        try:
            ok = node.get_component_iface().grab_focus()
        except Exception:
            ok = False
        print(json.dumps({"ok": bool(ok), "via": "grab_focus"}))
        return
    names = actions(node)
    for preferred in (action, "click", "press", "activate", "jump", "toggle", "open"):
        if preferred in names:
            ok = node.get_action_iface().do_action(names.index(preferred))
            print(json.dumps({"ok": bool(ok), "via": preferred}))
            return
    print(json.dumps({"ok": False, "via": None, "frame": extents(node)}))
    sys.exit(5)


def main():
    command = sys.argv[1]
    if command == "windows":
        print(json.dumps([{k: v for k, v in w.items() if k not in {"node", "path"}} for w in frames()]))
    elif command == "snapshot":
        print(json.dumps(snapshot(sys.argv[2] if len(sys.argv) > 2 else None), ensure_ascii=False))
    elif command == "act":
        act(*sys.argv[2:8])
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
