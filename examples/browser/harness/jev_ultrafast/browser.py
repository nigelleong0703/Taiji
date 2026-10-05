"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

# Derived from the running Chrome's major version so the UA tracks the installed browser.
def _headed_identity():
    version = cdp("Browser.getVersion")["product"].split("/")[-1]
    ua = (f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
          f"(KHTML, like Gecko) Chrome/{version} Safari/537.36")
    metadata = {
        "platform": "macOS", "platformVersion": "15.5.0", "platformArch": "arm",
        "uaFullVersion": version, "architecture": "arm", "model": "", "mobile": False,
        "bitness": "64", "wow64": False,
    }
    return ua, metadata

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        # Own window, not a background tab: Chrome throttles background tabs to ~2 fps despite focus emulation.
        self.target = cdp("Target.createTarget", url="about:blank", newWindow=True)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        # Headless Chrome advertises `HeadlessChrome` in its UA, which bot filters (Trip.com's
        # whaleguard among them) answer with HTTP 432 and an empty page. Present the same UA the
        # headed browser sends so observed pages are the real ones.
        ua, metadata = _headed_identity()
        self.call("Emulation.setUserAgentOverride", userAgent=ua, userAgentMetadata=metadata)
        # Keep rAF/menus rendering when the owned window is behind the user's, without stealing their tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        input_happened = False
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            input_happened = True
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass
        if input_happened:
            # A control can open a widget that renders well after the input: Google Flights' date picker
            # paints its day grid and its Next/Previous arrows about a second later. Reading the page in
            # that window shows a half-open picker, so the model sees no dates and no way to page. Wait for
            # the page to stop changing first; a settled page returns almost immediately.
            self.settle(quiet=0.15, limit=1.5)
        # A navigating document is a timing problem, not a decision: poll until a real page loads.
        deadline = time.monotonic() + 5
        while True:
            try:
                return browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except StalePage:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def fresh(self, page, action=None, page_only=False):
        """Is a decision about `page` still valid? Inputs compare the target and its nearby context (typing into an
        unchanged field is safe while prices elsewhere update); page_only (DONE/BLOCKED, which change nothing)
        compares document, URL, scroll, viewport and form values; anything else compares the whole visible page."""
        if page_only:
            return self.evaluate("(() => { const c=window.__jevFast; return c ? c.pageKey() : null; })()") \
                == page["page_key"]
        if action is not None and action["kind"] in {"click", "select", "fill"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def settle(self, quiet=0.3, limit=3.0):
        """Wait until the visible page stops changing (two reads `quiet` apart agree), at most `limit` seconds.
        Read-only: for pages that load prices or results asynchronously and void every decision meanwhile."""
        started, last = time.monotonic(), self.evaluate(MARKER)
        while time.monotonic() - started < limit:
            time.sleep(quiet)
            current = self.evaluate(MARKER)
            if current == last:
                return True
            last = current
        return False

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text})
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            resolver = """(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")"
            target = evaluate(resolver)
            if target is None and kind in {"click", "fill"}:
                # A date picker is often a horizontally scrolling strip: the month the goal needs is laid
                # out past the clip window, so its centre point hits the clipping container instead of the
                # cell and no click lands. Scroll to it the way a person would, then resolve again.
                evaluate("""(action => {
                  const e=window.__jevFast?.nodes.get(action.node);
                  if (!e?.isConnected) return false;
                  e.scrollIntoView({block:'nearest', inline:'center'});
                  return true;
                })(""" + json.dumps(action) + ")")
                target = evaluate(resolver)
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                # A node we observed can be laid out but covered by another container: Google's date
                # picker puts the whole trailing column (every Sunday) under the search-form layer, so
                # a coordinate click is refused and the goal date can never be chosen. The node id is
                # ours, never model-generated, so fall back to dispatching the click on the node itself.
                landed = evaluate("""(action => {
                  const e=window.__jevFast?.nodes.get(action.node);
                  if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                      !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return false;
                  const r=e.getBoundingClientRect();
                  const opts={bubbles:true,cancelable:true,composed:true,view:window,
                              clientX:r.x+r.width/2,clientY:r.y+r.height/2,button:0,buttons:1};
                  e.dispatchEvent(new PointerEvent('pointerdown', opts));
                  e.dispatchEvent(new MouseEvent('mousedown', opts));
                  e.dispatchEvent(new PointerEvent('pointerup', opts));
                  e.dispatchEvent(new MouseEvent('mouseup', opts));
                  e.click();
                  return true;
                })(""" + json.dumps(action) + ")")
                if not landed:
                    raise StalePage("Target changed or is covered. Observe again.")
                if kind == "fill":
                    call("Input.insertText", text=request["text"])
                return {"executed": action["id"]}
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
