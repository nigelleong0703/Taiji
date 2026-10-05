import { parseArgs } from "node:util";
import { InteractiveMode } from "@earendil-works/pi-coding-agent";
import { configure } from "./config.js";
import { createRuntime } from "./runtime.js";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const { values } = parseArgs({ options: {
  new: { type: "boolean" }, session: { type: "string" }, prompt: { type: "string" },
  json: { type: "boolean" }, "no-mcp": { type: "boolean" },
} });
const config = configure();
const runtime = await createRuntime(config, { fresh: values.new, sessionFile: values.session, mcp: !values["no-mcp"] });
try {
  if (values.prompt !== undefined) {
    const session = runtime.session;
    const started = performance.now();
    const auditStart = Date.now();
    let s1 = 0, s2 = 0, tools = 0, cacheRead = 0;
    await session.bindExtensions({});
    session.subscribe(event => {
      if (event.type === "message_end" && event.message.role === "assistant") {
        if (event.message.provider === "taiji-s1") s1++;
        else if (event.message.provider === "taiji-proxy") { s2++; cacheRead += event.message.usage.cacheRead; }
      }
      if (event.type === "tool_execution_end") tools++;
      if (values.json) console.log(JSON.stringify(event));
      else if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
        process.stdout.write(event.assistantMessageEvent.delta);
      }
    });
    await session.prompt(values.prompt);
    if (!values.json) process.stdout.write("\n");
    const last = [...session.messages].reverse().find(m => m.role === "assistant");
    if (last && "stopReason" in last && ["error", "aborted"].includes(String(last.stopReason))) {
      throw new Error("errorMessage" in last ? String(last.errorMessage) : String(last.stopReason));
    }
    console.error(`Session: ${session.sessionFile}`);
    const audit = readFileSync(resolve(config.stateDir, "audit.jsonl"), "utf8").trim().split("\n").map(line => JSON.parse(line))
      .filter(e => e.session === session.sessionManager.getSessionId() && e.at >= auditStart);
    const actions = audit.filter(e => e.type === "browser_action_result");
    const envServers = new Set(["browser", ...actions.map(e => e.env)]);
    console.error(JSON.stringify({ elapsed_ms: Math.round(performance.now() - started), s1_responses: s1,
      s2_responses: s2, s2_model_requests: audit.filter(e => e.type === "s2_request").length, tool_calls: tools, reported_cache_read_tokens: cacheRead,
      s1_model_requests: audit.filter(e => e.type === "s1_request").length + audit.filter(e => ["s1_intake", "s1_intake_error"].includes(e.type)).length
        + audit.filter(e => ["s1_decision", "s1_error"].includes(e.type)).reduce((sum, e) => sum + (e.model_calls?.length ?? 0), 0),
      local_actions: Object.fromEntries(["taiji-s1", "taiji-proxy"].map(executor => [executor, {
        attempted: actions.filter(e => e.executor === executor).length,
        executed: actions.filter(e => e.executor === executor && !e.is_error).length,
        observed_page_changes: actions.filter(e => e.executor === executor && !e.is_error && e.page_changed === true).length,
      }])),
      explicit_s2_fallbacks: audit.filter(e => e.type === "tool_dispatch" && e.tool === "taiji_delegate" && e.arguments?.executor === "s2").length,
      general_tool_results: audit.filter(e => e.type === "tool_execution_result" && !envServers.has(e.tool.match(/^mcp__(.+?)__/)?.[1])
        && !["taiji_delegate", "taiji_handoff"].includes(e.tool)).map(e => ({ tool: e.tool, executor: e.executor, is_error: e.is_error })),
    }));
  } else {
    if (!process.stdin.isTTY || !process.stdout.isTTY) throw new Error("Use --prompt for non-interactive runs.");
    await new InteractiveMode(runtime).run();
  }
} finally {
  await runtime.dispose();
}
// Some SDK background handles remain referenced after disposal; a headless CLI turn is over.
if (values.prompt !== undefined) process.exit(0);
