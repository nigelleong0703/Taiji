"""MCP server connections and a namespaced tool registry for the agent runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"MCP config needs environment variable {name}")
            return os.environ[name]

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_server_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read MCP host config (`mcpServers`) with `${ENV_VAR}` expansion for secrets."""
    raw = json.loads(Path(path).read_text())
    servers = raw.get("mcpServers", raw.get("servers"))
    if not isinstance(servers, dict) or not servers:
        raise ValueError("MCP config must contain a non-empty 'mcpServers' object")
    expanded = _expand_env(servers)
    for name, spec in expanded.items():
        if not isinstance(spec, dict) or not (spec.get("url") or spec.get("command")):
            raise ValueError(f"MCP server {name!r} needs either 'url' or 'command'")
        if spec.get("url") and spec.get("command"):
            raise ValueError(f"MCP server {name!r} cannot set both 'url' and 'command'")
    return expanded


def _function_name(server: str, tool: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}")
    if len(name) > 64:
        suffix = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = f"{name[:55]}_{suffix}"
    return name


@dataclass
class ToolBinding:
    """A model-visible tool name mapped to one MCP server and its original schema."""

    name: str
    server: str
    remote_name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    client: Any

    def openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def validate(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        Draft202012Validator(self.input_schema).validate(arguments)
        return arguments


class MCPRegistry:
    """Owns live MCP clients and routes namespaced calls to their originating servers."""

    def __init__(self, stack: AsyncExitStack, tools: dict[str, ToolBinding]):
        self._stack = stack
        self.tools = tools

    @classmethod
    async def connect(cls, servers: dict[str, dict[str, Any]]) -> MCPRegistry:
        from mcp import Client, StdioServerParameters
        from mcp.client.streamable_http import streamable_http_client

        stack = AsyncExitStack()
        await stack.__aenter__()
        tools: dict[str, ToolBinding] = {}
        try:
            for server_name, spec in servers.items():
                if spec.get("url"):
                    url = spec["url"]
                    headers = spec.get("headers") or {}
                    if headers:
                        import httpx2

                        http = await stack.enter_async_context(httpx2.AsyncClient(headers=headers))
                        transport = streamable_http_client(url, http_client=http)
                        client = await stack.enter_async_context(Client(transport))
                    else:
                        client = await stack.enter_async_context(Client(url))
                else:
                    params = StdioServerParameters(
                        command=spec["command"],
                        args=spec.get("args", []),
                        env=spec.get("env"),
                    )
                    client = await stack.enter_async_context(Client(params))

                cursor = None
                while True:
                    listed = await client.list_tools(cursor=cursor)
                    for remote in listed.tools:
                        alias = _function_name(server_name, remote.name)
                        if alias in tools:
                            raise ValueError(f"MCP tool name collision after normalization: {alias}")
                        schema = remote.input_schema or {"type": "object", "properties": {}}
                        Draft202012Validator.check_schema(schema)
                        annotations = remote.annotations.model_dump(by_alias=True) if remote.annotations else {}
                        description = remote.description or remote.title or remote.name
                        tools[alias] = ToolBinding(
                            name=alias,
                            server=server_name,
                            remote_name=remote.name,
                            description=description,
                            input_schema=schema,
                            annotations=annotations,
                            client=client,
                        )
                    cursor = listed.next_cursor
                    if cursor is None:
                        break
            if not tools:
                raise ValueError("The configured MCP servers exposed no tools")
            return cls(stack, tools)
        except BaseException:
            await stack.aclose()
            raise

    async def close(self) -> None:
        await self._stack.aclose()

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool is not registered: {name}")
        binding = self.tools[name]
        binding.validate(arguments)
        return await binding.client.call_tool(binding.remote_name, arguments)

    def model_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names if names is not None else list(self.tools)
        return [self.tools[name].openai_definition() for name in selected]

    def describe_for_s1(self) -> dict[str, str]:
        """Compact candidates for Taiji; full JSON schemas are reserved for S2 tool calls."""
        result = {}
        for name, tool in self.tools.items():
            properties = tool.input_schema.get("properties", {})
            fields = ", ".join(properties) if isinstance(properties, dict) else ""
            result[name] = f"{tool.description} (arguments: {fields or 'none'})"
        return result

    @staticmethod
    def result_for_model(result: Any, limit: int = 12000) -> str:
        """Render MCP structured or content results as bounded text for model context."""
        if getattr(result, "structured_content", None) is not None:
            value = result.structured_content
        else:
            value = []
            for block in getattr(result, "content", []):
                if hasattr(block, "text"):
                    value.append(block.text)
                elif hasattr(block, "uri"):
                    value.append({"resource": block.uri})
                else:
                    value.append({"type": getattr(block, "type", "content"), "omitted": True})
        text = json.dumps({"is_error": bool(getattr(result, "is_error", False)), "result": value},
                          ensure_ascii=False, default=str)
        if len(text) > limit:
            text = text[:limit] + "… [truncated]"
        return text
