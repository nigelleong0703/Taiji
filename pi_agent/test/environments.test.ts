import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeContext, type Message, type Model } from "@earendil-works/pi-ai";
import { envContext, names, s1Provider, zeroCost } from "../src/bridge.js";
import { envNames, environments, splitTool, targetKey } from "../src/environments.js";

const model: Model<"openai-completions"> = { id: "browser-policy", name: "S1", api: "openai-completions",
  provider: "taiji-s1", baseUrl: "http://localhost", reasoning: false, input: ["text", "image"],
  contextWindow: 64000, maxTokens: 1024, cost: zeroCost };
const desktop = envNames("desktop");
const targeted = { type: "object" as const, properties: { target_id: { type: "string" } } };
const tools = [...Object.values(names), "mcp__desktop__focus_app", "mcp__github__search"].map(name => ({ name, description: name,
  parameters: { type: "object" as const, properties: {} } }))
  .concat([desktop.observe, desktop.act, desktop.screenshot].map(name => ({ name, description: name, parameters: targeted })));
const window = { url: "app://TextEdit/Untitled", title: "Untitled", text: "", target_id: "w1",
  actions: [{ id: "e3", node: 3, kind: "fill", label: "Document body", role: "AXTextArea" }] };
const user: Message = { role: "user", content: "Type hello in TextEdit", timestamp: 1 };
const result = (toolName: string, toolCallId: string, body: object): Message => ({ role: "toolResult", toolName, toolCallId,
  content: [{ type: "text", text: JSON.stringify(body) }], isError: false, timestamp: 2 });
function call(id: string, name: string, args: Record<string, any>, provider = "taiji-proxy"): Message {
  return { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }], api: model.api, provider,
    model: model.id, timestamp: 3, stopReason: "toolUse",
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { ...zeroCost, total: 0 } } };
}

test("an environment is any server exposing observe and act", () => {
  assert.deepEqual(environments(tools), ["browser", "desktop"]);
  assert.deepEqual(splitTool("mcp__desktop__focus_app"), { server: "desktop", tool: "focus_app" });
  assert.equal(targetKey(tools.find(t => t.name === desktop.act), "desktop"), "target_id");
  assert.equal(targetKey(tools.find(t => t.name === names.act), "browser"), "tab_id");
});

test("desktop delegation observes the delegated window, then acts there with the environment's own tools", async () => {
  const delegate = call("d", names.delegate, { subgoal: "Type hello", done_when: "Document shows hello", handler: "desktop", target_id: "w1" });
  const messages = [user, delegate, result(names.delegate, "d", {})];
  const first = s1Provider(() => {}, async () => { throw new Error("Must observe the desktop first"); })(model,
    normalizeContext({ messages, tools }));
  const observe = (await first.result()).content[0] as any;
  assert.equal(observe.name, desktop.observe);
  assert.deepEqual(observe.arguments, { target_id: "w1" });

  const observed = [...messages, call("o", desktop.observe, { target_id: "w1" }, "taiji-s1"), result(desktop.observe, "o", window)];
  let input: any;
  const acting = s1Provider(() => {}, async received => { input = received; return { operation: "TYPE_TEXT", choice: "e3", text: "hello" }; })(model,
    normalizeContext({ messages: observed, tools }));
  const act = (await acting.result()).content[0] as any;
  assert.equal(input.env, "desktop");
  assert.deepEqual(input.capabilities, [desktop.act, desktop.observe, desktop.screenshot].sort());
  assert.equal(act.name, desktop.act);
  assert.deepEqual(act.arguments, { action_id: "e3", text: "hello", target_id: "w1" });
});

test("desktop history records no-effect attempts the same way as the browser", () => {
  const messages: Message[] = [user, call("o", desktop.observe, {}), result(desktop.observe, "o", window)];
  for (const id of ["a1", "a2"]) messages.push(call(id, desktop.act, { action_id: "e3" }, "taiji-s1"), result(desktop.act, id, window));
  const context = envContext(messages);
  assert.equal(context.env, "desktop");
  assert.deepEqual(context.history.map(h => [h.action_id, h.page_changed, h.unchanged_attempts]),
    [["read", true, 0], ["e3", false, 1], ["e3", false, 2]]);
});

test("unrelated MCP results neither reset nor replace the environment page", () => {
  const messages: Message[] = [user, call("o", desktop.observe, {}), result(desktop.observe, "o", window),
    call("g", "mcp__github__search", {}), result("mcp__github__search", "g", { items: [] })];
  assert.deepEqual(envContext(messages).page, window);
});

test("a handler naming an absent environment routes to the environment that owns the observed page", async () => {
  // Only the sandbox environment exists; its web page has an https URL, so S2 may say handler=browser.
  const sandbox = envNames("sandbox");
  const only = [...Object.values(names).filter(name => !name.startsWith("mcp__browser__")), "mcp__sandbox__open_url"]
    .map(name => ({ name, description: name, parameters: { type: "object" as const, properties: {} } }))
    .concat([sandbox.observe, sandbox.act, sandbox.screenshot].map(name => ({ name, description: name, parameters: targeted })));
  const page = { url: "https://www.google.com/travel/flights", title: "Flights", text: "", target_id: "web",
    actions: [{ id: "e1", node: 1, kind: "click", label: "Search" }] };
  const messages = [user, call("n", "mcp__sandbox__open_url", { url: page.url }), result("mcp__sandbox__open_url", "n", page),
    call("d", names.delegate, { subgoal: "Search", done_when: "Results visible", handler: "browser", target_id: "web" }),
    result(names.delegate, "d", {})];
  for (const handler of ["browser", "tools"]) {
    const delegated = messages.map(m => m.role === "assistant" && (m.content[0] as any).name === names.delegate
      ? call("d", names.delegate, { subgoal: "Search", done_when: "Results visible", handler, target_id: "web" }) : m);
    let input: any;
    const stream = s1Provider(() => {}, async received => { input = received; return { operation: "CLICK", choice: "e1" }; },
      async () => { throw new Error(`handler=${handler} with an owned target must not take the general tool path`); })(model,
      normalizeContext({ messages: delegated, tools: only }));
    const act = (await stream.result()).content[0] as any;
    assert.equal(input?.env, "sandbox", handler);
    assert.equal(act.name, sandbox.act);
    assert.deepEqual(act.arguments, { action_id: "e1", target_id: "web" });
  }
});
