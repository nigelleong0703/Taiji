# Taiji general agent runtime

This is the shared S1/S2 agent loop for registered MCP tools. It connects to MCP servers, lists their tools and JSON Schemas, lets Taiji (S1) make a fast choice when confidence is high, and uses an OpenAI-compatible S2 model for planning and native function calls when the choice is uncertain or the arguments are complex. Tool results return to the same history and loop.

Browser control belongs in this tool registry too: the included browser MCP adapter exposes the existing managed browser as `browser__open`, `browser__observe`, `browser__act`, and `browser__close`. It uses the same runtime, history, approval policy, Taiji decisions, and S2 tool-calling path as other MCP providers.

## Run the local demo

From the repository root, export the variables:

```sh
export TAIJI_S1_URL=http://127.0.0.1:8000
export TAIJI_S1_API_KEY=...
export TAIJI_S2_BASE_URL=https://api.openai.com/v1
export TAIJI_S2_API_KEY=...
export TAIJI_S2_MODEL=gpt-5-mini
```

Start Taiji's existing server in one terminal. In another, install the runtime and start a task:

```sh
uv sync --project agent_runtime
uv run --project agent_runtime taiji-agent \
  --mcp-config agent_runtime/examples/mcp.json \
  "Add a note that the MCP agent runtime is working, then show me the notes"
```

The demo server is launched over MCP stdio and stores notes only in memory. By default, the CLI asks before every tool call. Add `--yes` only when you want the agent to execute calls without interactive approval.

For the browser adapter (Python 3.11+), install its optional dependency and point the runtime at the browser MCP config:

```sh
uv sync --project agent_runtime --extra browser
uv run --project agent_runtime taiji-agent \
  --mcp-config agent_runtime/examples/browser-mcp.json \
  "Open https://example.com and tell me what the page is about"
```

The browser adapter reuses the repo's managed Chrome/CDP implementation. It returns visible page text and opaque action IDs; the model never supplies selectors or coordinates. Browser calls use the same per-tool approval prompts as every other MCP call.

If Taiji S1 is served with `--backend vllm`, its current endpoint supports decisions but not text generation. Set `TAIJI_S1_WRITE=false`; the runtime then sends argument generation directly to S2 without making a failed S1 text request.

## Register your own MCP servers

Use the standard `mcpServers` JSON shape. Each entry uses either a local process (`command`, `args`, optional `env`) or a Streamable HTTP `url` (optional `headers`). Use `${ENV_VAR}` placeholders for credentials; they are expanded from the current process environment and are not written into the config file.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/path/to/allowed/folder"]
    },
    "work_api": {
      "url": "https://tools.example.com/mcp",
      "headers": {"Authorization": "Bearer ${WORK_API_TOKEN}"}
    }
  }
}
```

Tools are namespaced as `server__tool` so separate servers may expose the same original tool name. S2 receives their complete JSON Schemas in standard chat-completions `tools`; Taiji sees concise tool descriptions as dynamic choices. S1 argument generation is limited to flat primitive schemas and is schema-validated before execution. Complex schemas go to S2. The runtime executes one tool at a time and records every call/result. S1 may propose completion, but S2 checks the tool history and can continue calling tools before returning a final answer.

Install and review each MCP server you configure: a server can perform the actions granted by its tools. The default CLI confirmation applies to all tool calls; `--yes` bypasses it. Programmatic users must supply an `approve` callback to `TaijiAgent`; the library default denies calls.

The runtime uses the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for stdio and Streamable HTTP connections.
