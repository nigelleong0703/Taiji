import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { configure, packageDir, repoDir } from "../src/config.js";
import { createRuntime } from "../src/runtime.js";
import { names } from "../src/bridge.js";

test("S2 owns intake, keeps its prefix on follow-up, reports cache usage", async () => {
  const payloads: any[] = [];
  let intakes = 0;
  const server = createServer(async (request, response) => {
    if (request.url === "/v1/models") {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ data: [{ id: "test-s2", context_window: 1048576,
        max_output_tokens: 131072, reasoning_efforts: [{ value: "low" }] }] }));
      return;
    }
    let text = "";
    for await (const chunk of request) text += chunk;
    const body = JSON.parse(text);
    if (request.url === "/v1/systemone") {
      const choice = ++intakes === 1 ? "s1" : "s2";
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ answers: { rank: { type: "choice", choice,
        probabilities: { s1: choice === "s1" ? 1 : 0, s2: choice === "s2" ? 1 : 0 }, confidence: 1 } } }));
      return;
    }
    assert.equal(request.url, "/v1/chat/completions");
    payloads.push(body);
    response.setHeader("content-type", "text/event-stream");
    response.write(`data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "test-s2",
      choices: [{ index: 0, delta: { role: "assistant", content: "READY" }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "test-s2",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 60000, completion_tokens: 1, total_tokens: 60001, prompt_tokens_details: { cached_tokens: 80 } } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise<void>(done => server.listen(0, "127.0.0.1", done));
  const port = (server.address() as { port: number }).port;
  const base = `http://127.0.0.1:${port}`;
  const state = mkdtempSync(resolve(tmpdir(), "taiji-pi-test-"));
  const old = { ...process.env };
  process.env.TAIJI_PI_STATE_DIR = state;
  process.env.TAIJI_S1_URL = base;
  process.env.TAIJI_S2_BASE_URL = base + "/v1";
  process.env.TAIJI_S2_MODEL = "test-s2";
  const runtime = await createRuntime(configure(), { fresh: true, mcp: false });
  try {
    await runtime.session.bindExtensions({});
    assert.equal(runtime.session.model?.contextWindow, 1048576);
    await runtime.session.prompt("Open a browser page");
    assert.ok(!runtime.session.messages.some(m => m.role === "toolResult" && m.toolName === names.handoff));
    assert.ok(!runtime.session.messages.some(m => m.role === "assistant" && m.provider === "taiji-s1"));
    assert.equal(runtime.session.getLastAssistantText(), "READY");
    await runtime.session.prompt("Just respond from our previous conversation");
    assert.equal(payloads.length, 2);
    assert.ok(!runtime.session.sessionManager.getBranch().some(entry => entry.type === "compaction"), "60K usage must not compact a 1M session");
    assert.deepEqual(payloads[1].tools, payloads[0].tools);
    assert.deepEqual(payloads[1].messages.slice(0, payloads[0].messages.length), payloads[0].messages);
    const audit = readFileSync(resolve(state, "audit.jsonl"), "utf8").trim().split("\n").map(line => JSON.parse(line));
    assert.equal(audit.filter(e => e.type === "s1_intake").length, 0);
    assert.equal(audit.filter(e => e.type === "s2_prefix").at(-1).previous_messages_preserved, true);
    assert.equal(audit.filter(e => e.type === "s2_usage").at(-1).usage.cacheRead, 80);
  } finally {
    await runtime.dispose();
    server.closeAllConnections();
    await new Promise<void>(done => server.close(() => done()));
    process.env = old;
  }
});

test("native Pi keeps S1 ownership after background completion until explicit phase DONE", async () => {
  const payloads: any[] = [];
  let localDecisions = 0;
  const server = createServer(async (request, response) => {
    response.setHeader("content-type", "application/json");
    if (request.url === "/v1/models") {
      response.end(JSON.stringify({ data: [{ id: "test-s2", context_window: 1048576,
        max_output_tokens: 131072, reasoning_efforts: [{ value: "low" }] }] })); return;
    }
    let text = "";
    for await (const chunk of request) text += chunk;
    const body = JSON.parse(text);
    if (request.url === "/v1/systemone") {
      if (body.questions.operation) localDecisions++;
      const operation = localDecisions <= 3 ? "WAIT" : "DONE";
      const answers = Object.fromEntries(Object.entries(body.questions).map(([name, question]: [string, any]) => {
        const choice = name === "rank" ? "s2" : name === "dispatch" ? "LOCAL" : operation;
        return [name, { choice, confidence: 1, probabilities: Object.fromEntries(Object.keys(question.criteria).map(key => [key, Number(key === choice)])) }];
      }));
      response.end(JSON.stringify({ model: "test-s1", answers })); return;
    }
    payloads.push(body);
    const call = payloads.length === 1 ? { name: names.act, arguments: JSON.stringify({ action_id: "wait", tab_id: 20 }) }
      : payloads.length === 2 ? { name: names.delegate, arguments: JSON.stringify({ subgoal: "Wait until loaded", done_when: "Loaded is visible", tab_id: 20 }) } : undefined;
    response.setHeader("content-type", "text/event-stream");
    const delta = call ? { role: "assistant", tool_calls: [{ index: 0, id: `call${payloads.length}`, type: "function", function: call }] }
      : { role: "assistant", content: "READY" };
    for (const [part, finish] of [[delta, null], [{}, call ? "tool_calls" : "stop"]]) {
      response.write(`data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "test-s2",
        choices: [{ index: 0, delta: part, finish_reason: finish }] })}\n\n`);
    }
    response.end("data: [DONE]\n\n");
  });
  await new Promise<void>(done => server.listen(0, "127.0.0.1", done));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const state = mkdtempSync(resolve(tmpdir(), "taiji-pi-execution-"));
  const old = { ...process.env };
  Object.assign(process.env, { TAIJI_PI_STATE_DIR: state, TAIJI_S1_URL: base,
    TAIJI_S2_BASE_URL: base + "/v1", TAIJI_S2_MODEL: "test-s2" });
  const config = configure();
  const actionLog = resolve(state, "actions.log");
  writeFileSync(config.mcpPath, JSON.stringify({ mcpServers: { browser: {
    command: resolve(repoDir, "agent_runtime/.venv/bin/python"), args: [resolve(packageDir, "test/browser_fixture.py")],
    env: { TAIJI_TEST_ACTION_LOG: actionLog }, exposure: "direct",
  } } }));
  const runtime = await createRuntime(config, { fresh: true });
  try {
    await runtime.session.bindExtensions({});
    await runtime.session.prompt("Load and report this page");
    assert.equal(runtime.session.getLastAssistantText(), "READY");
    assert.ok(readFileSync(actionLog, "utf8").trim().split("\n").every(line => line === "executed"), "Only delegated S1 actions execute");
    const audit = readFileSync(resolve(state, "audit.jsonl"), "utf8").trim().split("\n").map(line => JSON.parse(line));
    assert.equal(audit.filter(e => e.type === "execution_contract").length, 1);
    const actions = audit.filter(e => e.type === "browser_action_result" && !e.is_error);
    assert.equal(actions.length, 3, "S1 continues all three actions even after background S2 is ready");
    assert.ok(!audit.some(e => e.type === "s2_background_applied"));
    const done = audit.findIndex(e => e.type === "s1_decision" && e.operation === "DONE");
    const consulted = audit.findIndex(e => e.type === "s2_background_consulted");
    assert.ok(done >= 0 && consulted > done, "background advice is consumed only after S1 DONE");
    assert.ok(audit.findIndex(e => e.type === "s2_background_usage") < done);
    assert.ok(payloads[3].messages.some((m: any) => JSON.stringify(m).includes("Advisory only")), "fresh S2 sees the final handoff plus explicitly stale advice");
    assert.equal(actions[0].executor, "taiji-s1");
    assert.equal(actions[0].page_changed, true);
    assert.equal(payloads.length, 4);
    assert.deepEqual(payloads[0].tools, payloads[3].tools);
  } finally {
    await runtime.dispose();
    server.closeAllConnections();
    await new Promise<void>(done => server.close(() => done()));
    process.env = old;
  }
});

test('native Pi assigns terminal work directly to S2 without S1 intake', async () => {
  let commandWrites = 0, s2Calls = 0;
  const server = createServer(async (request, response) => {
    response.setHeader('content-type', 'application/json');
    if (request.url === '/v1/models') {
      response.end(JSON.stringify({ data: [{ id: 'test-s2', context_window: 1048576, max_output_tokens: 8192, reasoning_efforts: [{ value: 'low' }] }] })); return;
    }
    let text = '';
    for await (const chunk of request) text += chunk;
    const body = JSON.parse(text);
    if (request.url === '/v1/systemone') {
      const q = body.questions.rank;
      const criteria = q.criteria;
      assert.ok(!Object.values(criteria).some(value => String(value).startsWith('run_command:')));
      const choice = 's1' in criteria ? 's1' : 'ASK_S2';
      assert.ok(choice);
      response.end(JSON.stringify({ answers: { rank: { choice, confidence: 1,
        probabilities: Object.fromEntries(Object.keys(criteria).map(k => [k, Number(k === choice)])) } } })); return;
    }
    if (body.model === 's1') { commandWrites++; throw new Error('S1 writer must not run'); }
    s2Calls++;
    response.setHeader('content-type', 'text/event-stream');
    const call = s2Calls === 1 ? { name: 'run_command', arguments: JSON.stringify({ command: 'printf pi-native-command-ok' }) } : undefined;
    if (!call) assert.ok(JSON.stringify(body.messages).includes('pi-native-command-ok'));
    const delta = call ? { role: 'assistant', tool_calls: [{ index: 0, id: 'cmd', type: 'function', function: call }] }
      : { role: 'assistant', content: 'pi-native-command-ok' };
    for (const [part, finish] of [[delta, null], [{}, call ? 'tool_calls' : 'stop']]) {
      response.write(`data: ${JSON.stringify({ id: 'test', object: 'chat.completion.chunk', created: 1, model: 'test-s2',
        choices: [{ index: 0, delta: part, finish_reason: finish }] })}\n\n`);
    }
    response.end('data: [DONE]\n\n');
  });
  await new Promise<void>(done => server.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const state = mkdtempSync(resolve(tmpdir(), 'taiji-pi-command-'));
  const old = { ...process.env };
  Object.assign(process.env, { TAIJI_PI_STATE_DIR: state, TAIJI_S1_URL: base, TAIJI_S2_BASE_URL: base + '/v1', TAIJI_S2_MODEL: 'test-s2' });
  const runtime = await createRuntime(configure(), { fresh: true, mcp: false });
  try {
    await runtime.session.bindExtensions({});
    await runtime.session.prompt('Use run_command to produce a small diagnostic output, then report the result.');
    const results = runtime.session.messages.filter(m => m.role === 'toolResult' && m.toolName === 'run_command');
    assert.equal(results.length, 1);
    assert.equal((results[0] as any).isError, false);
    assert.equal(runtime.session.getLastAssistantText(), 'pi-native-command-ok');
    assert.equal(commandWrites, 0); assert.equal(s2Calls, 2);
    assert.ok(!runtime.session.messages.some(m => m.role === 'toolResult' && m.toolName === names.handoff));
    const audit = readFileSync(resolve(state, 'audit.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
    assert.equal(audit.filter(e => e.type === 's1_request').length, 0);
    assert.equal(audit.find(e => e.type === 'tool_dispatch' && e.tool === 'run_command').executor, 'taiji-proxy');
    assert.equal(audit.find(e => e.type === 'tool_execution_result' && e.tool === 'run_command').is_error, false);
  } finally {
    await runtime.dispose(); server.closeAllConnections();
    await new Promise<void>(done => server.close(() => done())); process.env = old;
  }
});
