"""CLI for the general MCP-backed Taiji agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .mcp_registry import MCPRegistry, load_server_config
from .runtime import S1Client, S2Client, TaijiAgent


def _approval(args):
    def approve(tool, arguments):
        if args.yes:
            return True
        print(f"\nRun MCP tool {tool.name} on server {tool.server}?", file=sys.stderr)
        print(json.dumps(arguments, ensure_ascii=False, indent=2), file=sys.stderr)
        return input("Approve? [y/N] ").strip().lower() in {"y", "yes"}

    return approve


def _trace_line(event):
    """One line per decision and tool call, so a long run is observable from the terminal."""
    kind = event["event"]
    if kind == "s1_decision":
        detail = f"{event.get('choice')} p={event.get('confidence')}"
    elif kind == "mcp_tool":
        detail = f"{event.get('tool')} -> {event.get('status')} ({event.get('duration_ms')} ms)"
    elif kind == "s2_request":
        detail = event.get("tool") or "final answer"
    else:
        return None
    print(f"[{event.get('at_ms')} ms] {kind}: {detail}", file=sys.stderr, flush=True)


async def build_agent(args, on_event=None):
    """Connect the MCP servers and build one agent.

    Kept apart from _run so a long-lived caller (the live view) can hold the agent and continue the
    conversation with further run() calls, instead of reconnecting and starting over each time.
    """
    servers = load_server_config(args.mcp_config)
    s1_url, s1_key = os.environ.get("TAIJI_S1_URL"), os.environ.get("TAIJI_S1_API_KEY")
    s2_url, s2_key, s2_model = (os.environ.get(name) for name in
                                ("TAIJI_S2_BASE_URL", "TAIJI_S2_API_KEY", "TAIJI_S2_MODEL"))
    missing = [name for name, value in {
        "TAIJI_S1_URL": s1_url, "TAIJI_S1_API_KEY": s1_key,
        "TAIJI_S2_BASE_URL": s2_url, "TAIJI_S2_API_KEY": s2_key, "TAIJI_S2_MODEL": s2_model,
    }.items() if not value]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))

    registry = await MCPRegistry.connect(servers)
    supports_write = os.environ.get("TAIJI_S1_WRITE", "true").lower() not in {"0", "false", "no"}
    s1 = S1Client(s1_url, s1_key, supports_write=supports_write)
    s2 = S2Client(s2_url, s2_key, s2_model, originator=os.environ.get("TAIJI_S2_ORIGINATOR"),
                  plan_model=os.environ.get("TAIJI_S2_PLAN_MODEL"))
    print("Registered MCP tools:", file=sys.stderr)
    for name, tool in registry.tools.items():
        print(f"  {name}: {tool.description}", file=sys.stderr)
    agent = TaijiAgent(registry, s1, s2, threshold=args.s1_threshold, max_steps=args.max_steps,
                       approve=_approval(args), compact_s1_context=args.compact_s1_context,
                       on_event=on_event or _trace_line,
                       audit_dir=None if getattr(args, "no_audit", False) else getattr(args, "audit_dir", None))
    return agent, (registry, s1, s2)


async def _run(args, on_event=None):
    agent, (registry, s1, s2) = await build_agent(args, on_event=on_event)
    try:
        result = await agent.run(args.goal, url=args.url)
        report = {"status": result.status, "message": result.message, "elapsed_ms": result.elapsed_ms,
                  "steps": result.steps, "trace": result.trace,
                  "audit": str(agent.audit_path) if agent.audit_path else None}
        if args.trace_file:
            args.trace_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.trace_file.with_name(args.trace_file.name + ".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(args.trace_file)
            visible = {"status": result.status, "message": result.message,
                       "elapsed_ms": result.elapsed_ms, "step_count": len(result.steps),
                       "trace_events": len(result.trace), "trace_file": str(args.trace_file),
                       "audit_file": str(agent.audit_path) if agent.audit_path else None}
        else:
            visible = report
        print(json.dumps(visible, ensure_ascii=False, indent=2))
        if result.status != "completed":
            raise SystemExit(1)
    finally:
        await s1.close()
        await s2.close()
        await registry.close()


def main():
    parser = argparse.ArgumentParser(description="Run Taiji's general S1/S2 agent with MCP tools.")
    parser.add_argument("goal", help="task for the agent")
    parser.add_argument("--mcp-config", type=Path, default=Path("mcp.json"),
                        help="MCP server configuration (default: ./mcp.json)")
    parser.add_argument("--s1-threshold", type=float, default=0.48,
                        help="registered-tool selection confidence below which S2 takes the turn")
    parser.add_argument("--max-steps", type=int, default=0, help="optional step cap; 0 (default) has no cap")
    parser.add_argument("--audit-dir", type=Path, default=Path("audit"),
                        help="append-only trail of every decision and tool call (default: ./audit)")
    parser.add_argument("--no-audit", action="store_true", help="do not write the audit trail")
    parser.add_argument("--yes", action="store_true", help="skip per-tool approval prompts")
    parser.add_argument("--url", help="optional convenience: open this URL before the normal agent loop")
    parser.add_argument("--trace-file", type=Path,
                        help="write exact model inputs/outputs, tool results, and per-call timings as JSON")
    parser.add_argument("--compact-s1-context", action="store_true",
                        help="truncate older S1 tool results to reduce repeated input processing")
    args = parser.parse_args()
    if not 0 <= args.s1_threshold <= 1 or args.max_steps < 0:
        parser.error("--s1-threshold must be in [0, 1] and --max-steps must be 0 (no limit) or positive")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
