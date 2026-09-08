import assert from 'node:assert/strict';
import test from 'node:test';
import { modelIdsForCostClass, allModelIdsForCostClass, nextAgentModelsForToggle } from '../src/agentModelGroups.ts';

const registry = [
  {
    name: 'free-cloud',
    enabled: true,
    access_type: 'api_key',
    cost_type: 'free',
    models: [
      { id: 'free/healthy', enabled: true },
      { id: 'free/down', enabled: true },
      { id: 'free/disabled', enabled: false },
    ],
  },
  {
    name: 'paid-cloud',
    enabled: true,
    access_type: 'subscription',
    cost_type: 'paid',
    models: [{ id: 'paid/healthy', enabled: true }],
  },
  {
    name: 'ollama',
    enabled: true,
    access_type: 'local',
    cost_type: 'free',
    models: [{ id: 'local/healthy', enabled: true }],
  },
  {
    name: 'disabled-provider',
    enabled: false,
    access_type: 'api_key',
    cost_type: 'free',
    models: [{ id: 'free/provider-disabled', enabled: true }],
  },
];

const health = {
  'free/healthy': 'healthy',
  'free/down': 'unhealthy',
  'free/disabled': 'healthy',
  'paid/healthy': 'healthy',
  'local/healthy': 'healthy',
  'free/provider-disabled': 'healthy',
};

test('cost toggles include all catalog models of that class even if offline', () => {
  assert.deepEqual(modelIdsForCostClass(registry, health, 'free'), ['free/healthy', 'free/down']);
});

test('local access wins over the provider cost flag', () => {
  assert.deepEqual(modelIdsForCostClass(registry, health, 'local'), ['local/healthy']);
  assert.deepEqual(modelIdsForCostClass(registry, health, 'paid'), ['paid/healthy']);
});

test('group toggle turns any active subset off, then turns the whole group on', () => {
  assert.deepEqual(
    nextAgentModelsForToggle(['free/healthy', 'unrelated/model'], ['free/healthy', 'free/second']),
    ['unrelated/model'],
  );
  assert.deepEqual(
    nextAgentModelsForToggle(['unrelated/model'], ['free/healthy', 'free/second']),
    ['unrelated/model', 'free/healthy', 'free/second'],
  );
});

test('allModelIdsForCostClass returns all models regardless of provider enabled state', () => {
  assert.deepEqual(allModelIdsForCostClass(registry, 'free'), [
    'free/healthy',
    'free/down',
    'free/disabled',
    'free/provider-disabled',
  ]);
});

test('group toggle with allModelsInClass removes any model in that class', () => {
  assert.deepEqual(
    nextAgentModelsForToggle(
      ['free/healthy', 'free/provider-disabled', 'unrelated/model'],
      ['free/healthy'],
      ['free/healthy', 'free/down', 'free/disabled', 'free/provider-disabled'],
    ),
    ['unrelated/model'],
  );
});
