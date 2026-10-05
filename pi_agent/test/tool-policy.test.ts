import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeContext, type Message } from '@earendil-works/pi-ai';
import { Type } from 'typebox';
import { toolDecision, executionContext, toolEvidence, toolState } from '../src/tool-policy.js';
import { s1Provider, names } from '../src/bridge.js';

const user: Message = { role: 'user', content: 'Count files in the working directory', timestamp: 1 };
const tools = [{ name: 'run_command', description: 'Execute a shell command', parameters: Type.Object({ command: Type.String(), timeout: Type.Optional(Type.Number()) }) },
  { name: names.handoff, description: 'Ask S2', parameters: Type.Object({ reason: Type.String() }) }];
const messages = normalizeContext({ messages: [user], tools }).messages;
const ranked = (choice: string) => async () => ({ choice, probabilities: {}, confidence: 1, elapsed_ms: 1, server_ms: 1 });

test('S1 excludes terminal tools and hands command work to S2 without writing arguments', async () => {
  const selection = await toolDecision(messages, () => {}, undefined, async request => {
    assert.ok(!request.options.some(option => option.label.startsWith('run_command:')));
    return ranked('ASK_S2')();
  }, async () => { throw new Error('S1 must not write commands'); });
  assert.equal(selection.name, names.handoff);
});

test('MCP arguments use the discovered schema, including nested objects and enums', async () => {
  const schema = Type.Object({ query: Type.String(), filter: Type.Object({ kind: Type.Union([Type.Literal('file'), Type.Literal('folder')]) }) });
  const context = normalizeContext({ messages: [user], tools: [{ name: 'mcp__files__search', description: 'Search files', parameters: schema }] });
  const args = { query: 'report', filter: { kind: 'file' } };
  const selected = await toolDecision(context.messages, () => {}, undefined, ranked('t0'), async () => JSON.stringify(args));
  assert.deepEqual(selected.arguments, args);
  await assert.rejects(toolDecision(context.messages, () => {}, undefined, ranked('t0'), async () => JSON.stringify({ ...args, filter: { kind: 'unsupported' } })), /Validation failed/);
});

test('parameterless tools do not call the writer; help and finish are model choices', async () => {
  const empty = normalizeContext({ messages: [user], tools: [{ name: 'mcp__files__status', description: 'Status', parameters: Type.Object({}) }] });
  const noWrite = async () => { throw new Error('writer must not run'); };
  assert.deepEqual(await toolDecision(empty.messages, () => {}, undefined, ranked('t0'), noWrite), { name: 'mcp__files__status', arguments: {} });
  assert.equal((await toolDecision(messages, () => {}, undefined, ranked('ASK_S2'), noWrite)).name, names.handoff);
  assert.deepEqual(await toolDecision(messages, () => {}, undefined, ranked('FINISH'), async () => 'Observed 12 files.'), { answer: 'Observed 12 files.' });
});

test('general tool results retain failure/output evidence and follow-up constraints', () => {
  const result: Message = { role: 'toolResult', toolCallId: 'cmd', toolName: 'run_command', content: [{ type: 'text', text: 'not found' }], isError: true, timestamp: 2 };
  const state = executionContext([...messages, result]);
  assert.equal((state.observations[0] as any).is_error, true);
  assert.equal((state.observations[0] as any).text, 'not found');
  const followup: Message = { ...user, content: 'Now show that number' };
  assert.ok(executionContext([...messages, result, followup]).prior_context.includes(String(user.content)));
});

test('general provider runs without browser observation and can end its own turn', async () => {
  const model: any = { api: 'openai-completions', provider: 'taiji-s1', id: 'test' };
  const context = normalizeContext({ messages: [user], tools });
  const browser = async () => { throw new Error('No browser operation expected'); };
  const call = await s1Provider(() => {}, browser, async () => ({ name: 'run_command', arguments: { command: 'pwd' } }))(model, context).result();
  assert.equal((call.content[0] as any).name, names.handoff);
  const done = await s1Provider(() => {}, browser, async () => ({ answer: 'Done from observed output.' }))(model, context).result();
  assert.equal(done.stopReason, 'stop');
  assert.equal((done.content[0] as any).text, 'Done from observed output.');
});


test('handoff packets are not duplicated as tool observations in S1 execution context', () => {
  const handoff: Message = { role: 'toolResult', toolCallId: 'h', toolName: names.handoff,
    content: [{ type: 'text', text: 'duplicated page evidence '.repeat(1000) }], isError: false, timestamp: 2 };
  assert.deepEqual(executionContext([...messages, handoff]).observations, []);
});


test('general S1 tool execution retains all task calls and outcomes beyond six results', () => {
  const history: any[] = [user];
  for (let i = 0; i < 18; i++) {
    history.push({ role: 'assistant', content: [{ type: 'toolCall', id: `c${i}`, name: 'lookup', arguments: { index: i } }] },
      { role: 'toolResult', toolCallId: `c${i}`, toolName: 'lookup', content: [{ type: 'text', text: `result ${i}` }], isError: i === 0 });
  }
  const context = executionContext(history);
  assert.equal(context.observations.length, 18);
  assert.equal(context.omitted_observations, 0);
  assert.deepEqual(context.observations[0], { tool: 'lookup', arguments: { index: 0 }, is_error: true,
    text: 'result 0', omitted_chars: 0, output_truncated: false, full_output_path: undefined });
});

test('tool options carry their own failure evidence; the state uses the environment shape', async () => {
  const call = (id: string, args: Record<string, any>): Message => ({ role: 'assistant', content: [{ type: 'toolCall', id, name: 'mcp__files__search', arguments: args }],
    api: 'openai-completions', provider: 'taiji-s1', model: 's1', timestamp: 2, stopReason: 'toolUse',
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  const result = (id: string, text: string, isError = false): Message => ({ role: 'toolResult', toolCallId: id, toolName: 'mcp__files__search',
    content: [{ type: 'text', text }], isError, timestamp: 3 });
  const tool = { name: 'mcp__files__search', description: 'Search files', parameters: Type.Object({ query: Type.String() }) };
  const history = [user, call('a', { query: 'x' }), result('a', 'none'), call('b', { query: 'x' }), result('b', 'none')];
  const observations = executionContext(normalizeContext({ messages: history, tools: [tool] }).messages).observations;
  assert.equal(toolEvidence(observations, 'mcp__files__search'), ' (called 2x in this task; last result repeated the previous call exactly)');
  assert.equal(toolEvidence(observations, 'mcp__files__other'), '');
  let request: any;
  await toolDecision(normalizeContext({ messages: history, tools: [tool] }).messages, () => {}, undefined,
    async input => { request = input; return ranked('ASK_S2')(); });
  assert.match(request.options[0].label, /called 2x/);
  assert.equal(request.state.page.url, 'tools://');
  const { tool: _name, ...observed } = observations[1] as any;
  assert.deepEqual(request.state.recent_actions[1], { action: 'mcp__files__search', kind: 'tool', ...observed });
});
