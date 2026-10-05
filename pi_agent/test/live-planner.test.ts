import { test } from 'node:test';
import assert from 'node:assert/strict';
import { type AssistantMessage, type Message } from '@earendil-works/pi-ai';
import { LivePlanner, replayResponse } from '../src/live-planner.js';
import { controller } from '../src/bridge.js';

const reply: AssistantMessage = { role: 'assistant', api: 'openai-completions', provider: 'taiji-proxy', model: 's2', timestamp: 99,
  content: [{ type: 'text', text: 'Finished from observed results' }], stopReason: 'stop',
  usage: { input: 10, output: 3, cacheRead: 0, cacheWrite: 0, totalTokens: 13,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
const messages: Message[] = [{ role: 'user', content: 'Task', timestamp: 1 }];
const tick = () => new Promise(resolve => setImmediate(resolve));

test('a ready background reply cannot interrupt S1; only handoff consumes an advisory', async () => {
  let finish!: (value: AssistantMessage) => void;
  let calls = 0;
  const planner = new LivePlanner(async snapshot => { calls++; assert.deepEqual(snapshot, messages); return new Promise(resolve => { finish = resolve; }); }, () => {});
  assert.equal(planner.poll('phase', messages), undefined);
  for (let i = 0; i < 25; i++) assert.equal(planner.poll('phase', messages), undefined);
  assert.equal(calls, 1, 'not one S2 call per S1 action');
  finish(reply); await tick();
  for (let i = 0; i < 25; i++) assert.equal(planner.poll('phase', messages), undefined);
  assert.equal(calls, 1);
  assert.equal(planner.takeForHandoff('phase')?.response, reply);
  assert.equal(planner.takeForHandoff('phase'), undefined);
  assert.equal(await replayResponse(reply).result(), reply);
  planner.reset();
});

test('a new phase or user cancellation discards old asynchronous decisions', async () => {
  const finishes: ((value: AssistantMessage) => void)[] = [];
  const signals: AbortSignal[] = [];
  const planner = new LivePlanner(async (_messages, signal) => { signals.push(signal); return new Promise(resolve => finishes.push(resolve)); }, () => {});
  planner.poll('old', messages); planner.poll('new', messages);
  assert.equal(signals[0].aborted, true);
  finishes[0](reply); await tick();
  assert.equal(planner.poll('new', messages), undefined);
  planner.reset(); assert.equal(signals[1].aborted, true);
  finishes[1](reply); await tick();
});

test('an applied S2 decision keeps ownership until its next delegation', () => {
  const prior: any = { role: 'toolResult', toolCallId: 'd', toolName: 'taiji_delegate', content: [], isError: false, timestamp: 2 };
  const action: any = { ...reply, content: [{ type: 'toolCall', id: 'nav', name: 'mcp__browser__navigate', arguments: {} }] };
  const result: any = { ...prior, toolName: 'mcp__browser__navigate', toolCallId: 'nav' };
  assert.equal(controller([...messages, prior, action, result]), 's2');
});


test('S2 can resume the current phase and reason again after further execution results', async () => {
  let calls = 0;
  const planner = new LivePlanner(async () => { calls++; return reply; }, () => {});
  planner.poll('phase', messages); await tick();
  assert.equal(planner.poll('phase', messages), undefined);
  assert.equal(planner.takeForHandoff('phase')?.response, reply);
  const result: any = { role: 'toolResult', toolCallId: 'continue', toolName: 'taiji_continue', content: [], isError: false, timestamp: 100 };
  const phase: any = { ...result, toolName: 'taiji_delegate', toolCallId: 'phase' };
  assert.equal(controller([...messages, phase, reply, result]), 's1');
  assert.equal(controller([...messages, reply, result]), 's2', 'cannot resume a nonexistent phase');
  assert.equal(planner.poll('phase', [...messages, result]), undefined);
  assert.equal(calls, 2);
  planner.reset();
});


test('handoff cancels unfinished planning and never consumes a different phase', async () => {
  let signal!: AbortSignal;
  const planner = new LivePlanner(async (_messages, current) => { signal = current; return new Promise(() => {}); }, () => {});
  planner.poll('phase', messages);
  assert.equal(planner.takeForHandoff('phase'), undefined);
  assert.equal(signal.aborted, true);
  const ready = new LivePlanner(async () => reply, () => {});
  ready.poll('old', messages); await tick();
  assert.equal(ready.takeForHandoff('new'), undefined);
});
