import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { normalizeContext, type AssistantMessage, type Message, type Model } from "@earendil-works/pi-ai";
import { browserContext, browserExecutionGate, controller, controlTools, names, s1Provider, zeroCost } from "../src/bridge.js";
import { PrefixAudit } from "../src/cache-audit.js";

const model: Model<"openai-completions"> = { id: "browser-policy", name: "S1", api: "openai-completions",
  provider: "taiji-s1", baseUrl: "http://localhost", reasoning: false, input: ["text", "image"],
  contextWindow: 64000, maxTokens: 1024, cost: zeroCost };
const page = { url: "https://example.test", title: "Example", text: "Next", actions: [{ id: "r19", node: 19, kind: "click", label: "Next" }] };
const user: Message = { role: "user", content: "Open the next page", timestamp: 1 };
const observed: Message = { role: "toolResult", toolName: names.observe, toolCallId: "observe", content: [{ type: "text", text: JSON.stringify(page) }], isError: false, timestamp: 2 };
const tools = Object.values(names).map(name => ({ name, description: name, parameters: { type: "object" as const, properties: {} } }));

test("S1 emits one native tool call from the observed table, without S2", async () => {
  const audit: any[] = [];
  const stream = s1Provider(e => audit.push(e), async input => {
    assert.deepEqual((input as any).page, page);
    return { operation: "CLICK", choice: "r19", raw_answers: { unused_head: "r999" } };
  })(model, normalizeContext({ messages: [user, observed], tools }));
  const result = await stream.result();
  assert.equal(result.stopReason, "toolUse");
  assert.equal(result.content.length, 1);
  assert.deepEqual(result.content[0], { type: "toolCall", id: (result.content[0] as any).id, name: names.act, arguments: { action_id: "r19" } });
  assert.equal(audit.at(-1).type, "s1_tool");
});

test("unobserved target and failed action transfer to S2 without another mutation", async () => {
  const stream = s1Provider(() => {}, async () => ({ operation: "CLICK", choice: "r999" }))(model, normalizeContext({ messages: [user, observed], tools }));
  assert.equal((await stream.result()).content[0].type, "toolCall");
  assert.equal(((await stream.result()).content[0] as any).name, names.handoff);
  const failed: Message = { ...observed, toolName: names.act, isError: true };
  const afterError = s1Provider(() => {}, async () => { throw new Error("Must not compute another mutation"); })(model, normalizeContext({ messages: [user, observed, failed], tools }));
  assert.match(((await afterError.result()).content[0] as any).arguments.reason, /unknown/);
});

test("LOOK is model-selected and images expire after a new observation", async () => {
  const stream = s1Provider(() => {}, async () => ({ operation: "LOOK", choice: "look" }))(model, normalizeContext({ messages: [user, observed], tools }));
  assert.equal(((await stream.result()).content[0] as any).name, names.screenshot);
  const shot: Message = { ...observed, toolName: names.screenshot, content: [{ type: "image", data: "base64", mimeType: "image/png" }] };
  assert.equal(browserContext([user, observed, shot]).page?.screenshot, "base64");
  assert.equal(browserContext([user, observed, shot, observed]).page?.screenshot, undefined);
});

test("S2 keeps control until successful delegation; new user starts new intake", () => {
  const handoff: Message = { ...observed, toolName: names.handoff };
  const delegate: Message = { ...observed, toolName: names.delegate };
  assert.equal(controller([user, handoff, observed]), "s2");
  assert.equal(controller([user, handoff, delegate, observed]), "s1");
  assert.equal(controller([user, handoff, { ...delegate, isError: true }]), "s2");
  assert.equal(controller([user, handoff, user]), undefined);
});

test("aborting S1 produces an aborted response with no tool call", async () => {
  const cancel = new AbortController();
  const stream = s1Provider(() => {}, async () => { cancel.abort(); throw new Error("cancelled"); })(model,
    normalizeContext({ messages: [user, observed], tools }), { signal: cancel.signal });
  const result = await stream.result();
  assert.equal(result.stopReason, "aborted"); assert.deepEqual(result.content, []);
});

test("cache audit detects prefix rewrites and tool schema changes", () => {
  const audit = new PrefixAudit();
  const payload = { tools, messages: [{ role: "system", content: "fixed" }, { role: "user", content: "task" }] };
  audit.inspect(payload);
  const next = audit.inspect({ ...payload, messages: [...payload.messages, { role: "assistant", content: "done" }] });
  assert.equal(next.tools_stable, true); assert.equal(next.system_stable, true); assert.equal(next.previous_messages_preserved, true);
  const changed = audit.inspect({ tools: [], messages: [{ role: "system", content: "changed" }] });
  assert.equal(changed.tools_stable, false); assert.equal(changed.system_stable, false); assert.equal(changed.previous_messages_preserved, false);
});

test("complete inline MCP output is used when display is compact; malformed output invalidates page", () => {
  const path = resolve(mkdtempSync(resolve(tmpdir(), "taiji-page-")), "page.txt");
  writeFileSync(path, JSON.stringify(page));
  const truncated = { ...observed, content: [{ type: "text", text: "Compact page state" }], structuredContent: { structuredContent: page }, details: { server: "browser" } } as any;
  assert.deepEqual(browserContext([user, truncated]).page, page);
  assert.equal(browserContext([user, observed, { ...truncated, structuredContent: undefined }]).page, undefined);
});

test("active follow-up keeps prior results as context rather than repeating completed goals", () => {
  const answer: Message = { role: "assistant", content: [{ type: "text", text: "Previous book price was £45.17" }],
    api: model.api, provider: model.provider, model: model.id, timestamp: 3, stopReason: "stop",
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { ...zeroCost, total: 0 } } };
  const goal = browserContext([user, observed, answer, { ...user, content: "Now open the book details" }]).goal;
  assert.match(goal, /do not redo completed work/);
  assert.match(goal, /Previous book price was £45.17/);
  assert.match(goal, /Active user request \(authoritative\):\nNow open the book details/);
});

test("new user turn observes afresh before using historical element refs", async () => {
  const stream = s1Provider(() => {}, async () => { throw new Error("Old page must not be used"); })(model,
    normalizeContext({ messages: [user, observed, { ...user, content: "Click the next link" }], tools }));
  assert.equal(((await stream.result()).content[0] as any).name, names.observe);
});

test("prefix fingerprints survive a process/session resume without storing message contents", () => {
  const path = resolve(mkdtempSync(resolve(tmpdir(), "taiji-prefix-")), "prefix.json");
  const first = { tools, messages: [{ role: "system", content: "fixed" }, { role: "user", content: "private task" }] };
  new PrefixAudit(path).inspect(first);
  const result = new PrefixAudit(path).inspect({ ...first, messages: [...first.messages, { role: "assistant", content: "answer" }] });
  assert.equal(result.previous_messages_preserved, true);
  assert.equal(result.tools_stable, true);
});

test("model can request fresh state or S2 directly", async () => {
  for (const [operation, tool] of [["READ", names.observe], ["ASK_S2", names.handoff]]) {
    const stream = s1Provider(() => {}, async input => {
      assert.ok((input as any).capabilities.includes(names.screenshot));
      return { operation, choice: operation.toLowerCase() };
    })(model, normalizeContext({ messages: [user, observed], tools }));
    assert.equal(((await stream.result()).content[0] as any).name, tool);
  }
});

test("failed repetition is evidence; it cannot override the model's chosen next operation", async () => {
  const messages: Message[] = [user, observed];
  for (let i = 0; i < 3; i++) {
    const assistant: any = { role: "assistant", content: [{ type: "toolCall", id: `repeat${i}`, name: names.act,
      arguments: { action_id: "r19" } }], api: model.api, provider: model.provider, model: model.id,
      timestamp: i + 3, stopReason: "toolUse", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { ...zeroCost, total: 0 } } };
    messages.push(assistant, { ...observed, toolName: names.act, toolCallId: `repeat${i}` });
  }
  const stream = s1Provider(() => {}, async input => {
    assert.equal((input as any).history.at(-1).unchanged_attempts, 3);
    assert.equal((input as any).history.at(-1).page_changed, false);
    return { operation: "LOOK", choice: "look" };
  })(model, normalizeContext({ messages, tools }));
  assert.equal(((await stream.result()).content[0] as any).name, names.screenshot);
});

function assistantCall(id: string, name: string, args: Record<string, any>, provider = "taiji-proxy"): Message {
  return { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }],
    api: model.api, provider, model: model.id, timestamp: 3, stopReason: "toolUse",
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { ...zeroCost, total: 0 } } };
}

test("S2 local execution requires an explicit fallback; nested calls cannot bypass it", () => {
  const action = assistantCall("act", names.act, { action_id: "wait" });
  assert.equal(browserExecutionGate([user, action], names.act, "act")?.block, true);
  assert.equal(browserExecutionGate([user, assistantCall("code", "codemode", {})], names.act, "code/0")?.block, true);
  assert.equal(browserExecutionGate([user, assistantCall("act", names.act, {}, "taiji-s1")], names.act, "act"), undefined);
  const fallback = assistantCall("fallback", names.delegate, { executor: "s2", subgoal: "Recover", reason: "S1 cannot select this control" });
  const result: Message = { ...observed, toolName: names.delegate, toolCallId: "fallback" };
  assert.equal(controller([user, fallback, result]), "s2");
  assert.equal(browserExecutionGate([user, fallback, result, action], names.act, "act"), undefined);
  assert.equal(browserExecutionGate([user, fallback, result, user, action], names.act, "act")?.block, true);
  assert.equal(browserExecutionGate([user, fallback, { ...result, isError: true }, action], names.act, "act")?.block, true);
});

test("delegation carries phase completion and preserves failed attempts for recovery", async () => {
  const action = assistantCall("a", names.act, { action_id: "r19" }, "taiji-s1");
  const actionResult: Message = { ...observed, toolName: names.act, toolCallId: "a" };
  const delegate = assistantCall("d", names.delegate, { subgoal: "Open Next", done_when: "New article text is visible" });
  const messages = [user, observed, action, actionResult, delegate, { ...observed, toolName: names.delegate, toolCallId: "d" }];
  const projected = browserContext(messages);
  assert.equal(projected.subgoal, "Open Next");
  assert.equal(projected.doneWhen, "New article text is visible");
  assert.equal(projected.history[0].unchanged_attempts, 1);
  const stream = s1Provider(() => {}, async input => {
    assert.equal((input as any).done_when, projected.doneWhen);
    return { operation: "DONE" };
  })(model, normalizeContext({ messages, tools }));
  assert.match(((await stream.result()).content[0] as any).arguments.reason, /delegated phase/);
  const tool = controlTools().find(t => t.name === names.delegate)!;
  const bad = await tool.execute("bad", { subgoal: "Recover", done_when: "Recovery observed", executor: "s2" }, undefined as any, undefined as any, undefined as any);
  assert.equal((bad as any).isError, true);
});

test("delegated tab is observed before decisions and binds subsequent local actions", async () => {
  const delegate = assistantCall("d", names.delegate, { subgoal: "Submit Search", done_when: "Loading starts", tab_id: 20 });
  const result: Message = { ...observed, toolName: names.delegate, toolCallId: "d" };
  const wrongTab: Message = { ...observed, content: [{ type: "text", text: JSON.stringify({ ...page, tab_id: 10 }) }] };
  const messages = [user, wrongTab, delegate, result];
  const switching = s1Provider(() => {}, async () => { throw new Error("Must select the delegated tab first"); })(model,
    normalizeContext({ messages, tools }));
  assert.deepEqual(((await switching.result()).content[0] as any).arguments, { tab_id: 20 });
  const correctTab: Message = { ...observed, content: [{ type: "text", text: JSON.stringify({ ...page, tab_id: 20 }) }] };
  const acting = s1Provider(() => {}, async () => ({ operation: "CLICK", choice: "r19" }))(model,
    normalizeContext({ messages: [...messages, correctTab], tools }));
  assert.deepEqual(((await acting.result()).content[0] as any).arguments, { action_id: "r19", tab_id: 20 });
});


test("harness input guard returns cause and final state inline without replacing history", async () => {
  const path = resolve(mkdtempSync(resolve(tmpdir(), "taiji-handoff-")), "page.json");
  writeFileSync(path, JSON.stringify({ ...page, tab_id: 20, text: "evidence ".repeat(3000) }));
  const full = { ...observed, structuredContent: { structuredContent: { ...page, tab_id: 20, text: "Final price £56.88; stock 6" } } } as any;
  const inputLimit = { source: "browser_projection", unit: "characters", actual: 27000, limit: 20000 };
  const events: any[] = [];
  const messages = [user, full];
  const before = JSON.stringify(messages);
  const response = await s1Provider(e => events.push(e), async () => {
    throw Object.assign(new Error("Harness input guard blocked S1 before inference"), { input_limit: inputLimit, model_calls: [] });
  })(model, normalizeContext({ messages, tools })).result();
  const call = response.content[0] as any;
  assert.equal(call.name, names.handoff);
  assert.equal(call.arguments.kind, "input_limit");
  assert.deepEqual(call.arguments.state.input_limit, inputLimit);
  assert.equal(call.arguments.state.tab_id, 20);
  assert.equal(call.arguments.state.page.text, 'Final price £56.88; stock 6');
  assert.equal(call.arguments.state.observation_refs, undefined);
  assert.ok(!JSON.stringify(call.arguments).includes('evidence evidence'));
  assert.equal(JSON.stringify(messages), before);
  assert.equal(events.find(e => e.type === 's1_error').model_calls.length, 0);
  const tool = controlTools().find(t => t.name === names.handoff)!;
  const result = await tool.execute('h', call.arguments, undefined as any, undefined as any, undefined as any);
  assert.deepEqual(JSON.parse((result.content[0] as any).text), call.arguments);
});

test("model help and phase completion remain distinct from harness failures", async () => {
  for (const [operation, kind] of [['ASK_S2', 'model_help'], ['DONE', 'phase_done']]) {
    const response: AssistantMessage = await s1Provider(() => {}, async () => ({ operation }))(model,
      normalizeContext({ messages: [user, observed], tools })).result();
    const args: any = (response.content[0] as any).arguments;
    assert.equal(args.kind, kind);
    assert.equal(args.state.input_limit, undefined);
  }
});


test("S1 retains the entire task operation chain across replanning, including READ", async () => {
  const messages: Message[] = [user, observed];
  for (let i = 0; i < 25; i++) {
    messages.push(assistantCall(`a${i}`, names.act, { action_id: "r19" }, "taiji-s1"),
      { ...observed, toolName: names.act, toolCallId: `a${i}` });
  }
  messages.push(assistantCall("read", names.observe, {}, "taiji-s1"),
    { ...observed, toolName: names.observe, toolCallId: "read" });
  messages.push(assistantCall("phase", names.delegate, { subgoal: "Recover", done_when: "Target is visible" }),
    { ...observed, toolName: names.delegate, toolCallId: "phase" });
  const context = browserContext(messages);
  assert.equal(context.history.length, 26);
  assert.equal(context.history[0].unchanged_attempts, 1);
  assert.equal(context.history[24].unchanged_attempts, 25);
  assert.equal(context.history[25].kind, "observe");
  const stream = s1Provider(() => {}, async input => {
    assert.deepEqual((input as any).history, context.history);
    return { operation: "ASK_S2" };
  })(model, normalizeContext({ messages, tools }));
  await stream.result();
});
