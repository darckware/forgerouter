import assert from 'node:assert/strict';
import test from 'node:test';
import { hermesAgentConfig } from '../src/agentClientConfigs.ts';

test('Hermes config leaves context length to endpoint autodetection', () => {
  const config = hermesAgentConfig('http://router.test/v1', 'secret-key');

  assert.match(config, /default: forgerouter\/auto/);
  assert.match(config, /base_url: http:\/\/router\.test\/v1/);
  assert.match(config, /api_key: secret-key/);
  assert.doesNotMatch(config, /context_length:/);
});
