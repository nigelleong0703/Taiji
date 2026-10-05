"""Thin live front end for a Taiji agent run: embedded browser plus a step-by-step action log.

Run the agent in-process and stream its trace events over Server-Sent Events. The page shows the
live browser as a screencast frame, the exact request sent to S1, and every action the agent took.
The prompt and step limit are settable from the page before the run starts.

Usage: python -m tools.live_ui --url https://www.trip.com/flights/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("live_ui.html")

# The connected agent outlives one task: the page, the MCP browser and the transcript stay, so a
# follow-up message continues the conversation instead of starting over.
SESSION: dict = {"agent": None, "clients": None}

# One loop for the whole server: the S1/S2 httpx clients are created inside it and cannot be reused from
# a different loop, so a follow-up run must not get a fresh one.
LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, daemon=True).start()


async def _close_session() -> None:
    clients = SESSION.get("clients")
    SESSION["agent"] = SESSION["clients"] = None
    if not clients:
        return
    registry, s1, s2 = clients
    for close in (s1.close, s2.close, registry.close):
        try:
            await close()
        except Exception:
            pass


class Run:
    """One agent run, with a rolling event log and the latest browser frame."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.subscribers: list[threading.Event] = []
        self.frame: bytes | None = None
        self.result: dict | None = None
        self.running = False

    def _wake(self):
        for waiter in self.subscribers:
            waiter.set()

    def emit(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)
            self._wake()

    def set_frame(self, data: bytes) -> None:
        with self.lock:
            self.frame = data
            self._wake()

    def wait(self, timeout: float = 1.0):
        waiter = threading.Event()
        with self.lock:
            self.subscribers.append(waiter)
        try:
            waiter.wait(timeout)
            with self.lock:
                return list(self.events), self.frame, self.result
        finally:
            with self.lock:
                if waiter in self.subscribers:
                    self.subscribers.remove(waiter)

    def snapshot(self):
        with self.lock:
            return list(self.events), self.frame, self.result


class _LiveArgs:
    """Argparse-shaped carrier so the CLI's _run needs no changes."""

    def __init__(self, body: dict):
        self.goal = body.get("goal", "")
        self.mcp_config = ROOT / "examples" / "browser-mcp.json"
        self.s1_threshold = float(body.get("s1_threshold") or 0.48)
        # 0 means "no step limit": the run ends when the model finishes or the stall guard stops it.
        self.max_steps = int(body["max_steps"]) if body.get("max_steps") is not None else 0
        self.yes = True
        self.url = body.get("url") or None
        self.trace_file = None
        self.compact_s1_context = bool(body.get("compact_s1_context"))
        self.audit_dir = ROOT / "audit"
        self.no_audit = False


def _screencast(run: Run, stop: threading.Event, want_host: str | None = None) -> None:
    """Mirror the agent's own Chrome window into the page.

    The browser is owned by the MCP server subprocess, so this attaches to the same CDP target
    read-only and pushes JPEG frames; it never issues input.
    """
    harness = ROOT.parent / "examples" / "browser" / "harness"
    import sys
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    try:
        from browser_harness.helpers import cdp
    except ImportError:
        run.emit({"event": "browser_error", "error": "browser_harness not importable",
                  "at_ms": 0, "duration_ms": 0, "status": "error"})
        return
    session = None
    target = None
    while not stop.wait(1.0):
        try:
            pages = [t for t in cdp("Target.getTargets")["targetInfos"]
                     if t.get("type") == "page" and t.get("url") not in ("about:blank", "")]
            # Mirror the agent's own window. Chrome usually holds other tabs (earlier runs, a viewer), and
            # the run's page is not necessarily the newest one, so the pick is rechecked every frame.
            preferred = [t for t in pages if want_host and want_host in (t.get("url") or "")]
            chosen = (preferred or pages)[-1] if (preferred or pages) else None
            if chosen is not None and chosen["targetId"] != target:
                target, session = chosen["targetId"], None
            if session is None:
                if target is None:
                    continue
                session = cdp("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
            data = cdp("Page.captureScreenshot", session_id=session, format="jpeg", quality=70)["data"]
            frame = base64.b64decode(data)
            # A page that closes or is still blank returns a near-empty frame; keep the last real one so
            # the panel does not go black the moment the run ends.
            if len(frame) > 3000:
                run.set_frame(frame)
        except Exception:
            session = None
            if stop.wait(2.0):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Live browser + agent action log for Taiji.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--url", default=None, help="open this page first")
    parser.add_argument("--prompt", default=None, help="prefill the goal box")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    page = PAGE.read_text(encoding="utf-8")
    run = Run()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (page.replace("__DEFAULT_URL__", args.url or "")
                            .replace("__DEFAULT_PROMPT__", args.prompt or "")
                            .replace("__DEFAULT_STEPS__", str(args.max_steps)))
                self._send(200, body.encode(), "text/html; charset=utf-8")
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                sent = 0
                deadline = time.monotonic() + 1800
                try:
                    while time.monotonic() < deadline:
                        events, frame, result = run.wait(1.0)
                        payload = {"n": len(events), "events": events[sent:],
                                   "done": result is not None, "frame": len(frame or b"")}
                        sent = len(events)
                        self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                        self.wfile.flush()
                        if result is not None:
                            final = {"result": result, "done": True}
                            self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
                            self.wfile.flush()
                            break
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path == "/frame.jpg":
                _events, frame, _result = run.snapshot()
                self._send(200, frame or b"", "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/start":
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            if run.running:
                return self._send(409, b'{"error":"already running"}', "application/json")
            run.running = True
            stop = threading.Event()
            want_host = urlparse(args.url).hostname if args.url else None
            threading.Thread(target=_screencast, args=(run, stop, want_host), daemon=True).start()

            def worker():
                async def execute():
                    from taiji_agent.cli import build_agent

                    live_args = _LiveArgs(body)
                    agent = SESSION.get("agent") if not body.get("fresh") else None
                    if agent is not None:
                        agent.on_event = run.emit
                        return agent, await agent.run(live_args.goal, url=live_args.url, fresh=False)
                    await _close_session()
                    agent, clients = await build_agent(live_args, on_event=run.emit)
                    SESSION["agent"], SESSION["clients"] = agent, clients
                    return agent, await agent.run(live_args.goal, url=live_args.url)

                try:
                    agent, result = asyncio.run_coroutine_threadsafe(execute(), LOOP).result()
                    run.result = {"status": result.status, "message": result.message,
                                  "elapsed_ms": result.elapsed_ms, "steps": result.steps,
                                  "trace": result.trace, "turn": getattr(agent, "_turn", 1),
                                  "audit": str(agent.audit_path) if agent.audit_path else None}
                except SystemExit:
                    pass
                except Exception as error:
                    run.emit({"event": "run_error", "error": f"{type(error).__name__}: {error}",
                              "at_ms": 0, "duration_ms": 0, "status": "error"})
                finally:
                    stop.set()
                    with run.lock:
                        if run.result is None:
                            run.result = {"status": "error", "message": "run ended without a result"}
                        run.running = False
                        run._wake()

            threading.Thread(target=worker, daemon=True).start()
            self._send(200, b'{"ok":true}', "application/json")

        def log_message(self, *_):
            pass

    print(f"Taiji live UI on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
