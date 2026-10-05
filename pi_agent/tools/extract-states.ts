/**
 * Replay recorded Pi sessions into the decision states S1 saw, for every environment.
 *
 * Environment states (browser, desktop, ...): every page observation that a model then acted on,
 * rebuilt with the live `envContext`, so rows match what `env_decision.py` received.
 * Tool states: the exact `s1_tool_request` inputs the general tool path recorded in the audit log.
 *
 *   npx tsx tools/extract-states.ts [--audit .state/audit.jsonl] .state/sessions/*.jsonl > states.jsonl
 */
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import type { Message } from "@earendil-works/pi-ai";
import { envContext } from "../src/bridge.js";
import { envNames, environmentsIn, splitTool } from "../src/environments.js";

const args = process.argv.slice(2);
const auditIndex = args.indexOf("--audit");
const auditPath = auditIndex >= 0 ? args.splice(auditIndex, 2)[1] : undefined;
const sessions = new Set<string>();

for (const file of args) {
  const messages: Message[] = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const entry = JSON.parse(line);
    if (entry.type === "session") sessions.add(entry.id);
    if (entry.type === "message" && entry.message?.role !== "system") messages.push(entry.message);
  }
  const envs = environmentsIn(messages);
  for (let i = 1; i < messages.length; i++) {
    const before = messages[i - 1], next = messages[i];
    if (before.role !== "toolResult" || before.isError || next.role !== "assistant") continue;
    const parts = splitTool(before.toolName);
    if (!parts || !envs.has(parts.server) || parts.tool === "screenshot_image" || /^(list|close)/.test(parts.tool)) continue;
    const { page, env, goal, subgoal, doneWhen, history } = envContext(messages.slice(0, i));
    if (!page || !env) continue;
    const call = next.content.find(b => b.type === "toolCall");
    const { screenshot: _image, ...textPage } = page as typeof page & { screenshot?: string };
    const names = envNames(env);
    console.log(JSON.stringify({
      source: { session: basename(file), message_index: i, controller: next.provider === "taiji-proxy" ? "s2" : "s1", kind: "env" },
      request: { env, page: textPage, goal, subgoal, done_when: doneWhen, history,
        capabilities: [names.act, names.observe, names.screenshot].sort() },
      next_call: call?.type === "toolCall" ? { name: call.name, arguments: call.arguments } : null,
    }));
  }
}

if (auditPath) {
  for (const line of readFileSync(auditPath, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const event = JSON.parse(line);
    if (event.type !== "s1_tool_request" || (sessions.size && !sessions.has(event.session))) continue;
    const { type: _type, session, at, ...request } = event;
    console.log(JSON.stringify({ source: { session, at, controller: "s1", kind: "tools" }, tool_request: request }));
  }
}
