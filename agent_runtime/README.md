# Taiji general agent runtime

For the new Pi-native session experiment, see [pi_agent/README.md](../pi_agent/README.md)
and [the session handoff](../docs/handoffs/2026-10-02-pi-session.md). It keeps native
Pi history and execution while allowing S1 browser actions and stable S2 tools
across handoffs. This Python runtime remains the previous implementation.

This is the shared S1/S2 agent loop for registered MCP tools. It connects to MCP servers, lists their tools and JSON Schemas, lets Taiji (S1) make a fast choice when confidence is high, and uses an OpenAI-compatible S2 model for planning and native function calls when the choice is uncertain or the arguments are complex. Tool results return to the same history and loop.

The input is a task, with or without a URL in its text. S1 first chooses from registered tools;
S2 handles uncertainty and complex arguments. `--url` is an optional setup shortcut, not an agent
type. After a browser observation, the browser decision handler provides operation/target heads
and a `USE_TOOLS` operation. Choosing it returns to the registered-tool choices with the current
page in context, so having an open page does not prevent using another capability.

When S1 keeps stalling after a planner subgoal, S2 retains control of recovery across tool calls.
S2 can finish with an answer or call `agent__delegate_s1` with a concrete `subgoal` and a `browser`
or `tools` capability to return control explicitly. A recovery tool call no longer automatically
hands the next step back to the stalled policy. Screenshot tool results are attached as image
messages for S2, rather than sending the base64 payload as text.

`taiji_agent/browser_policy.py` owns the compatibility boundary to the vendored browser policy.
Importing that policy does not connect a CDP browser. Pilot owns browser transport and normalisation;
the runtime owns execution, shared conversation and audit. This is still a specialised browser handler,
not yet a general plugin registry for arbitrary decision handlers.

The runtime browser handler uses the top-two relative gap `(p1-p2)/(p1+p2)` for tie retries. Unlike
an absolute probability gap, this cancels the softmax denominator if candidate scores stay fixed.
It still needs calibration for actual accuracy: adding candidates can also change the model's scores.
`--s1-threshold` applies to registered-tool selection; browser target probabilities are not compared
with this threshold. The trace records tie candidates, before/after probabilities and server timing.
Pilot scroll controls remain available independently of the number of element targets.

Each run sends a unique cache-session ID with S1 decisions. When S1 is served by Transformers with `--shared-prefix` enabled and the state is text-only, the server can reuse the stable goal prefix across turns while processing the changing tool history and question. Run the inference cache parity check before enabling that server flag. S2 receives the growing transcript as usual; prefix-aware S2 servers can reuse its unchanged history prefix automatically.

Browser control belongs in this tool registry too: the included browser MCP adapter exposes the existing managed browser as `browser__open`, `browser__observe`, `browser__act`, `browser__screenshot`, and `browser__close`. It uses the same runtime, history, approval policy, Taiji decisions, and S2 tool-calling path as other MCP providers. `browser__screenshot` returns the viewport with every observed element boxed and numbered, and `TAIJI_S1_SCREENSHOT=1` sends that picture with each browser decision, **The model decides when it needs the picture**: every observation offers a `Look at the screen` operation, and choosing it makes the next decision carry the viewport, so a vision S1 can ground itself without paying for an image it did not ask for. Set `TAIJI_S1_SCREENSHOT=always` to force the picture into every browser decision instead; leave it unset for text-only decisions.

## Run the local demo

From the repository root, export the variables:

```sh
export TAIJI_S1_URL=http://127.0.0.1:8000
export TAIJI_S1_API_KEY=...
export TAIJI_S2_BASE_URL=https://api.openai.com/v1
export TAIJI_S2_API_KEY=...
export TAIJI_S2_MODEL=gpt-5-mini
```

To route S2 through a running OCX (OpenCodeX) proxy using Muse Spark 1.3 Contributor, set:

```sh
export TAIJI_S2_BASE_URL=http://127.0.0.1:10100/v1
export TAIJI_S2_API_KEY=opencodex
export TAIJI_S2_MODEL=opencode-go/muse-spark-1.3-contributor
export TAIJI_S2_ORIGINATOR=opencode
```

The runtime sends `Originator: opencode` with S2 requests. For OpenCode's own model picker, the corresponding model name is `opencodex/opencode-go/muse-spark-1.3-contributor` because OpenCode adds its OCX provider namespace.

Start Taiji's existing server in one terminal. In another, install the runtime and start a task:

```sh
uv sync --project agent_runtime
uv run --project agent_runtime taiji-agent \
  --mcp-config agent_runtime/examples/mcp.json \
  --trace-file /tmp/taiji-agent-trace.json \
  "Add a note that the MCP agent runtime is working, then show me the notes"
```

The trace JSON records each model call's endpoint, exact request body, full and actually sent S1 context, model output, full tool results, and per-call timing. It can contain page contents and task data. S1 context is sent in full by default; add `--compact-s1-context` only when you want older S1 tool results shortened to reduce repeated processing. S2 keeps its full conversation history in either mode.

The demo server is launched over MCP stdio and stores notes only in memory. By default, the CLI asks before every tool call. Add `--yes` only when you want the agent to execute calls without interactive approval.

For the browser adapter (Python 3.11+), install its optional dependency and point the runtime at the browser MCP config:

```sh
uv sync --project agent_runtime --extra browser
uv run --project agent_runtime taiji-agent \
  --mcp-config agent_runtime/examples/browser-mcp.json \
  "Open https://example.com and tell me what the page is about"
```

The browser adapter reuses the repo's managed Chrome/CDP implementation. It returns visible page text and opaque action IDs; the model never supplies selectors or coordinates. Browser calls use the same per-tool approval prompts as every other MCP call.

With the Taiji vLLM source patch installed, `--backend vllm` serves both decision scores and LoRA field writing through one engine. Keep `TAIJI_S1_WRITE=true` (the default) so the runtime can use the local writer. Set it to `false` only for an endpoint that actually lacks text generation. See [`../docs/vllm.md`](../docs/vllm.md) for patch installation and validation.

## Audit trail

Every run writes an append-only trail by default: one JSON object per line, flushed as it happens, so a
run that is killed or crashes still leaves the record behind. (The `--trace-file` JSON is written only
once, at the end.)

```
agent_runtime/audit/20261001T143121Z-0e14df548da1.jsonl
```

- The first line is `run_started`: run id, goal, start URL, the S1 and S2 endpoints, the S2 model, the
  step limit and threshold, and the host and user that ran it.
- Then one line per event: every S1 decision (choice, operation, confidence, probabilities, latency, and
  the exact request the model saw), every tool call (name, arguments, status, duration), every S2 call,
  and every approval denial.
- The last line is `run_finished` with the final status, message, elapsed time, and step count.
- Long payloads (a request carrying the screenshot the model saw) are stored as a length plus a sha256
  digest, so a long-running trail stays small: a five-step browser task is about 70 KB.

Review it with `jq` or `grep`, or render a visual replay:
`python -m tools.trace_html <trace.json> out.html` shows each step's screenshot, the element table the
model chose from, the probabilities, and the timings.

`--audit-dir DIR` moves the trail; `--no-audit` disables it. Tasks started from the live view are
audited into `agent_runtime/audit/` as well.

## Long-running runs

There is no step cap by default (`--max-steps 0`). The run ends when the model answers, reports the
goal infeasible, or encounters an unrecoverable provider/planner error; the user can also stop it.
The stall guard asks for a subgoal and then transfers recovery to S2 if S1 keeps stalling.
It does not independently verify the task or declare it complete.

For a resident agent that takes tasks over time, run the live view and start tasks from its page: the
process and its browser MCP server stay alive between tasks, and every task gets its own audit file.
Calling the loop again on the same agent **continues the conversation** rather than starting over: the page
stays open, the tool history and the transcript carry over, so a follow-up can refer to what just happened
(`agent.run("now open the first result")` after `agent.run("open the travel category", url=...)`).
Pass `fresh=True` to start a new conversation on the same agent. Each turn gets its own audit file and
records its turn number.

In the live view, type the next message and press 开始 again — that continues the conversation. Tick
新会话 to reconnect and start over.

## Watch it live

```sh
cd <repository root>
PYTHONPATH=$PWD/agent_runtime uv run --project agent_runtime python -m tools.live_ui \
  --port 8130 --url "https://www.google.com/travel/flights?hl=en" --max-steps 20
```

Open `http://127.0.0.1:8130`, type the goal, and press 开始. The right column streams every decision
and tool call as it happens; the left column mirrors the agent's browser window; the box at the bottom
shows the final result. The 步数 field takes 0 for no limit.

Two things to know: the harness drives a **headless** Chrome, so there is no browser window on screen to
watch — the mirror is the view; and the mirror keeps the last real frame after a run ends, so the panel
does not go black. Start it from the repository root: the browser MCP server is spawned with a relative
path.

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

A stdio server does not inherit the caller's whole environment: the MCP SDK passes only a small platform
whitelist (PATH, HOME, and similar) plus whatever the config's own `env` block declares. A tuning knob for an
adapter therefore belongs in the config, not in the shell the agent happens to be launched from. The Pilot
adapter's action-space size is the worked example: `TAIJI_PILOT_MAX_ELEMENTS` is set to 320 in
[`examples/pilot-mcp-wide.json`](examples/pilot-mcp-wide.json), which offers every element Pilot snapshots
instead of the default 200, so a decision question can carry more than 255 options. Pair it with
`TAIJI_MAX_CRITERIA` on the S1 server (see [`../hf/macos/Taiji-2B/README.md`](../hf/macos/Taiji-2B/README.md)).
