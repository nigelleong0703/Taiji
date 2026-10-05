import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeContext, type Message } from '@earendil-works/pi-ai';
import { Type } from 'typebox';
import { s2ExecutionTool, terminalExecutionGate } from '../src/execution-policy.js';

test('S2 owns native terminal, code and schema-discovered command tools; ordinary MCP remains with S1', () => {
  for (const name of ['bash', 'run_command', 'write', 'edit', 'codemode', 'mcp__terminal__exec']) assert.ok(s2ExecutionTool(name));
  assert.ok(s2ExecutionTool('mcp__custom__invoke', { name: 'mcp__custom__invoke', description: '', parameters: Type.Object({ command: Type.String() }) }));
  assert.equal(s2ExecutionTool('mcp__files__search', { name: 'mcp__files__search', description: '', parameters: Type.Object({ query: Type.String() }) }), false);
});

test('S1 direct and nested terminal calls are blocked while S2 executes freely', () => {
  const call: any = { role: 'assistant', provider: 'taiji-s1', content: [{ type: 'toolCall', id: 'parent', name: 'codemode', arguments: {} }] };
  const messages = normalizeContext({ messages: [call as Message], tools: [{ name: 'mcp__custom__invoke', description: '', parameters: Type.Object({ script: Type.String() }) }] }).messages;
  assert.ok(terminalExecutionGate(messages, 'bash', 'parent'));
  assert.ok(terminalExecutionGate(messages, 'mcp__custom__invoke', 'parent/child'));
  assert.equal(terminalExecutionGate(messages, 'mcp__files__search', 'parent/child'), undefined);
  assert.equal(terminalExecutionGate([{ ...call, provider: 'taiji-proxy' }], 'bash', 'parent'), undefined);
});
