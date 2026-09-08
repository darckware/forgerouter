export function hermesAgentConfig(baseUrl: string, keyToken: string): string {
  return `model:\n  provider: forgerouter\n  default: forgerouter/auto\nproviders:\n  forgerouter:\n    base_url: ${baseUrl}\n    default_model: forgerouter/auto\n    transport: chat_completions\n    api_key: ${keyToken}`;
}
