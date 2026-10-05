"""Tiny local MCP server to exercise tool discovery and calls without touching real data."""

from mcp.server import MCPServer

server = MCPServer("taiji-demo")
_notes: list[str] = []


@server.tool()
def add_note(text: str) -> dict[str, int]:
    """Add a note to this temporary in-memory demo notebook."""
    _notes.append(text)
    return {"notes": len(_notes)}


@server.tool()
def list_notes() -> list[str]:
    """List notes currently stored in the temporary demo notebook."""
    return list(_notes)


if __name__ == "__main__":
    server.run(transport="stdio")
