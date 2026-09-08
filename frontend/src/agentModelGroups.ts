type Model = { id: string; enabled: boolean };
type Provider = {
  enabled?: boolean;
  access_type?: 'subscription' | 'api_key' | 'local';
  cost_type?: 'free' | 'paid';
  models: Model[];
};

export type CostClass = 'free' | 'paid' | 'local';

export function modelIdsForCostClass(
  registry: Provider[],
  healthByModelOrTarget: Record<string, string> | CostClass,
  targetOrNothing?: CostClass,
): string[] {
  const target: CostClass = (typeof targetOrNothing === 'string'
    ? targetOrNothing
    : healthByModelOrTarget) as CostClass;
  return registry.flatMap((provider) => {
    if (provider.enabled === false) return [];
    const costClass: CostClass = provider.access_type === 'local'
      ? 'local'
      : provider.cost_type === 'paid'
        ? 'paid'
        : 'free';
    if (costClass !== target) return [];
    return provider.models
      .filter((model) => model.enabled !== false)
      .map((model) => model.id);
  });
}

export function allModelIdsForCostClass(
  registry: Provider[],
  target: CostClass,
): string[] {
  return registry.flatMap((provider) => {
    const costClass: CostClass = provider.access_type === 'local'
      ? 'local'
      : provider.cost_type === 'paid'
        ? 'paid'
        : 'free';
    if (costClass !== target) return [];
    return provider.models.map((model) => model.id);
  });
}

export function nextAgentModelsForToggle(
  currentModels: string[],
  groupModels: string[],
  allModelsInClass?: string[],
): string[] {
  const current = new Set(currentModels);
  const toRemove = new Set([...groupModels, ...(allModelsInClass ?? [])]);
  const anyOn = groupModels.some((id) => current.has(id)) || (allModelsInClass?.some((id) => current.has(id)) ?? false);
  return anyOn
    ? currentModels.filter((id) => !toRemove.has(id))
    : [...new Set([...currentModels, ...groupModels])];
}
