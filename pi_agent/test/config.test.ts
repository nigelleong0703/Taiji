import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { configure, repoDir } from '../src/config.js';

test('private S1 endpoint config loads without overriding explicit environment settings', () => {
  const state = mkdtempSync(resolve(tmpdir(), 'taiji-s1-config-'));
  writeFileSync(resolve(state, 's1.env'), 'export TAIJI_S1_URL=http://local-s1:8010\nexport TAIJI_S1_API_KEY=test-key\nTAIJI_S2_MODEL=must-not-load\n');
  const env: NodeJS.ProcessEnv = { TAIJI_PI_STATE_DIR: state };
  configure(env);
  assert.equal(env.TAIJI_S1_URL, 'http://local-s1:8010');
  assert.equal(env.TAIJI_S1_API_KEY, 'test-key');
  assert.equal(env.TAIJI_S2_MODEL, undefined);
  assert.equal(env.TAIJI_PI_CWD, repoDir);
  const custom: NodeJS.ProcessEnv = { TAIJI_PI_STATE_DIR: state, TAIJI_S1_URL: 'http://explicit:8010', TAIJI_S1_API_KEY: 'explicit-key' };
  configure(custom);
  assert.equal(custom.TAIJI_S1_URL, 'http://explicit:8010');
  assert.equal(custom.TAIJI_S1_API_KEY, 'explicit-key');
});
