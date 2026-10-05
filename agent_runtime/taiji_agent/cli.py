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


async def _run(args):
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
    s1, s2 = S1Client(s1_url, s1_key, supports_write=supports_write), S2Client(s2_url, s2_key, s2_model)
    try:
        print("Registered MCP tools:", file=sys.stderr)
        for name, tool in registry.tools.items():
            print(f"  {name}: {tool.description}", file=sys.stderr)
        agent = TaijiAgent(registry, s1, s2, threshold=args.s1_threshold, max_steps=args.max_steps,
                           approve=_approval(args))
        result = await agent.run(args.goal)
        print(json.dumps({"status": result.status, "message": result.message, "steps": result.steps},
                         ensure_ascii=False, indent=2))
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
                        help="below this Taiji confidence, hand the turn to S2")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--yes", action="store_true", help="skip per-tool approval prompts")
    args = parser.parse_args()
    if not 0 <= args.s1_threshold <= 1 or args.max_steps < 1:
        parser.error("--s1-threshold must be in [0, 1] and --max-steps must be positive")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
