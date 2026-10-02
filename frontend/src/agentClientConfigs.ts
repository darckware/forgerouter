export function hermesAgentConfig(baseUrl: string, keyToken: string): string {
  return `model:\n  provider: forgerouter\n  default: forgerouter/auto\nproviders:\n  forgerouter:\n    base_url: ${baseUrl}\n    default_model: forgerouter/auto\n    transport: chat_completions\n    api_key: ${keyToken}`;
}

// Snippets target tools running on this host. The dashboard's public origin
// may be a proxy domain that local coding tools should never depend on.
export function agentClientBaseUrl(kind: 'root' | 'v1'): string {
  const origin = 'http://localhost:2100';
  return kind === 'root' ? origin : `${origin}/v1`;
}
