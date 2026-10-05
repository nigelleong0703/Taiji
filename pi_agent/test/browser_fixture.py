"""In-memory browser for native SDK execution-contract tests; no actual browser."""
import asyncio
import os
from pathlib import Path

from mcp.server import MCPServer

server = MCPServer("test-browser")
loaded = False


def record_action():
    with Path(os.environ["TAIJI_TEST_ACTION_LOG"]).open("a") as output:
        output.write("executed\n")


@server.tool()
async def observe() -> dict:
    return {"url": "https://example.test", "title": "Test", "text": "Loaded" if loaded else "Loading",
            "tab_id": 20, "actions": [{"id": "wait", "kind": "wait", "label": "Wait", "node": None}]}


@server.tool()
async def act(action_id: str, tab_id: int | None = None) -> dict:
    global loaded
    assert action_id == "wait"
    assert tab_id == 20
    await asyncio.to_thread(record_action)
    loaded = True
    return await observe()


if __name__ == "__main__":
    server.run(transport="stdio")
