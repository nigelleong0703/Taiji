"""General S1/S2 agent runtime backed by registered MCP tools."""

from .mcp_registry import MCPRegistry, load_server_config
from .runtime import AgentResult, S1Client, S2Client, TaijiAgent

__all__ = ["AgentResult", "MCPRegistry", "S1Client", "S2Client", "TaijiAgent", "load_server_config"]
