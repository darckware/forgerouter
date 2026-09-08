import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import ReCAPTCHA from 'react-google-recaptcha';
import * as Select from '@radix-ui/react-select';
import { Activity, AlertTriangle, ArrowLeft, AudioLines, Bot, Boxes, Brain, Check, CheckCircle2, CheckCheck, ChevronDown, ChevronUp, Code, Copy, CopyPlus, DollarSign, ExternalLink, Eye, EyeOff, HeartPulse, ImagePlus, Info, KeyRound, Layers, LayoutDashboard, Link2, Loader2, LogOut, MessageSquare, Monitor, Moon, Network, PanelLeftClose, PanelLeftOpen, Pause, Pencil, Plus, Power, PowerOff, RefreshCw, Route, Save, Scissors, Send, ShieldCheck, Shuffle, SignalHigh, SignalLow, SignalMedium, SlidersHorizontal, Sun, Terminal, Trash2, Type, User, UsersRound, Wrench, X } from 'lucide-react';
import './style.css';
import { selectedSubscriptionPlan, selectedSubscriptionPlanName, subscriptionPlanAuthUrl } from './subscriptionPlans';
import { modelIdsForCostClass, allModelIdsForCostClass, nextAgentModelsForToggle, type CostClass } from './agentModelGroups';
import { hermesAgentConfig } from './agentClientConfigs';

type ProviderHealth = { provider: string; model_id: string; tier: number; status: string; http_code: number | null; latency_ms: number | null; error_message: string | null; checked_at: string | null; };
type RouteEvent = { route_id: number; request_id: string; model_id: string | null; required_capability: string; status: string; error_type: string | null; created_at: string | null; total_tokens: number | null; cost: number; reference_cost: number | null; agent: string | null; demand: string | null; prompt_preview: string | null; messages_dropped: number | null; context_window: number | null; context_budget: number | null; context_action: 'none' | 'summarized' | 'truncated' | 'rejected' | null; context_candidates_skipped: number; };
type UsageDay = { day: string; messages: number; tokens: number; cost: number; reference_cost: number };
type UsageModel = { model_id: string; messages: number; tokens: number; cost: number; reference_cost: number; pct_total: number };
type UsageDemand = { demand: string; messages: number; tokens: number; cost: number; reference_cost: number; pct_total: number };
type AgentUsage = { agent: string; totals: { messages: number; tokens: number; cost: number; reference_cost: number }; daily: UsageDay[]; by_model: UsageModel[] };
type UsageTotals = { messages: number; tokens: number; cost: number; reference_cost: number; tokens_raw?: number; tokens_saved?: number; pct_saved?: number; code_downgrades?: number };
type Usage = { days: number; totals: UsageTotals; daily: UsageDay[]; by_model: UsageModel[]; by_demand?: UsageDemand[]; by_agent?: AgentUsage[] };
type UsageMetric = 'messages' | 'tokens' | 'cost' | 'reference_cost';
type MonthlyTotals = { messages: number; tokens: number; cost: number; reference_cost: number };
type YearlyAgentUsage = { agent: string; months: Record<string, MonthlyTotals>; totals: MonthlyTotals };
type YearlyUsage = { year: number; by_agent: YearlyAgentUsage[] };
type YearlyDemandUsage = { demand: string; months: Record<string, MonthlyTotals>; totals: MonthlyTotals };
type YearlyUsageByDemand = { year: number; by_demand: YearlyDemandUsage[] };
type ProviderReady = { provider: string; tier: number; enabled: boolean; api_key_env: string; api_key_required: boolean; api_key_configured: boolean; api_key_source?: string; api_key_masked?: string; models: string[]; };
type HealthInfo = { status: string; http_code: number | null; latency_ms: number | null; error: string | null };
type RegistryModel = { id: string; provider_model: string; capabilities: string[]; enabled: boolean; healthy?: boolean; score?: number; context_length?: number | null; health?: string; health_detail?: HealthInfo };
type DiscoveredModel = { id: string; score: number; free: boolean | null; capabilities: string[]; health: HealthInfo | null };
type AccessType = 'subscription' | 'api_key' | 'local';
type RegistryProvider = { name: string; tier: number; base_url: string; api_key_env: string; enabled: boolean; models: RegistryModel[]; api_key?: string; api_key_set?: boolean; api_key_masked?: string; access_type?: AccessType; cost_type?: 'free' | 'paid'; api_format?: 'openai' | 'anthropic'; auth_method?: string; auth_config?: { extra_headers?: Record<string, string> } };
type SubscriptionPlan = { name: string; display_name: string; plan_hint: string; base_url: string; auth_method: string; token_hint: string; auth_url: string; extra_headers: Record<string, string> };
type PlayMessage = { role: 'user' | 'assistant'; content: string; meta?: string; images?: string[] };
type Attachment = { id: string; url: string; name: string };

type AgentDaily = { day: string; messages: number; tokens: number; cost: number; reference_cost: number };
type AgentInfo = { name: string; avatar_data_url?: string | null; enabled: boolean; created_at: string | null; description?: string; aux_tasks?: boolean; kind?: 'agent' | 'service'; api_key_masked: string; messages: number; tokens: number; cost: number; reference_cost: number; budget_limit_usd: number | null; budget_action: string; month_spend: number; daily: AgentDaily[]; models: string[]; models_off?: string[]; config_path?: string; config_format?: string; config_key?: string; restart_service?: string };

type DemandData = { demands: string[]; info: Record<string, string>; routes: Record<string, string[]>; defaults: Record<string, string[]>; virtual_models: string[] };

type WatchdogStatus = { healthy_enabled: number | null; total_enabled: number | null; min_healthy: number | null; cooldown_seconds: number | null; degraded: boolean; last_check_at: string | null; last_scan_at: string | null };

type StatusFilter = 'all' | 'healthy' | 'unhealthy';
type Page = 'agents' | 'overview' | 'messages' | 'routing' | 'tasks' | 'playground' | 'pricing' | 'users' | 'profiles';
type PriceModel = { public_id: string; provider_model: string; priced: boolean; input_cost_per_token: number | null; output_cost_per_token: number | null; source: string | null };

// Canonical capability catalog from the PRD.
const CAPABILITIES = ['text', 'code', 'reasoning', 'tool_call', 'vision', 'embedding', 'audio'] as const;
const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const CAP_ICONS: Record<string, React.ReactNode> = {
  text: <Type size={12} />,
  code: <Code size={12} />,
  reasoning: <Brain size={12} />,
  tool_call: <Wrench size={12} />,
  vision: <Eye size={12} />,
  embedding: <Boxes size={12} />,
  audio: <AudioLines size={12} />,
};

function CapIcons({ caps }: { caps: string[] }) {
  return <span className="capIcons">{caps.filter((cap) => CAP_ICONS[cap]).map((cap) => <i key={cap} title={cap}>{CAP_ICONS[cap]}</i>)}</span>;
}

// Color/icon pairing for each demand class — mirrors the Overview "Model groups" panel.
const DEMAND_GROUP_STYLE: Record<string, { accent: string; icon: React.ReactNode }> = {
  simple: { accent: 'accent-green', icon: <SignalLow size={13} /> },
  standard: { accent: 'accent-blue', icon: <SignalMedium size={13} /> },
  complex: { accent: 'accent-amber', icon: <SignalHigh size={13} /> },
  reasoning: { accent: 'accent-violet', icon: <Brain size={13} /> },
  vision: { accent: 'accent-pink', icon: <Eye size={13} /> },
  audio: { accent: 'accent-teal', icon: <AudioLines size={13} /> },
  code: { accent: 'accent-orange', icon: <Code size={13} /> },
};

function DemandTag({ demand }: { demand: string | null }) {
  if (!demand) return <span className="muted">-</span>;
  const style = DEMAND_GROUP_STYLE[demand];
  return (
    <span className={`demandTag ${style ? style.accent.replace('accent-', 'text-') : ''}`}>
      {style?.icon}{demand}
    </span>
  );
}

const FREE_PROVIDER_PRESETS: { name: string; label: string; base_url: string; website_url: string; key_url: string; api_format?: 'openai' | 'anthropic'; hint: string }[] = [
  { name: 'huggingface', label: 'Hugging Face Inference API', base_url: 'https://api-inference.huggingface.co/v1', website_url: 'https://huggingface.co', key_url: 'https://huggingface.co/settings/tokens', hint: 'Free serverless inference for open-source models with User Access Token (hf_...)' },
  { name: 'modelscope', label: 'ModelScope (Aliyun / Alibaba)', base_url: 'https://api-inference.modelscope.cn/v1', website_url: 'https://modelscope.cn', key_url: 'https://modelscope.cn/my/myaccesstoken', hint: 'Free daily tokens for Qwen, DeepSeek and GLM models with ModelScope SDK Token' },
  { name: 'siliconflow', label: 'SiliconFlow (SiliconCloud)', base_url: 'https://api.siliconflow.cn/v1', website_url: 'https://siliconflow.cn', key_url: 'https://cloud.siliconflow.cn/account/ak', hint: 'Generous free tier for Qwen, DeepSeek, GLM, TeleChat open models' },
  { name: 'together', label: 'Together AI', base_url: 'https://api.together.xyz/v1', website_url: 'https://www.together.ai', key_url: 'https://api.together.xyz/settings/api-keys', hint: 'OpenAI-compatible serverless endpoints for Llama, Mistral, Qwen' },
  { name: 'fireworks', label: 'Fireworks AI', base_url: 'https://api.fireworks.ai/inference/v1', website_url: 'https://fireworks.ai', key_url: 'https://fireworks.ai/api-keys', hint: 'Fast inference for Llama 3.3, Qwen 2.5 Coder, DeepSeek' },
  { name: 'hyperbolic', label: 'Hyperbolic', base_url: 'https://api.hyperbolic.xyz/v1', website_url: 'https://hyperbolic.xyz', key_url: 'https://app.hyperbolic.xyz/settings', hint: 'Decentralized fast inference for DeepSeek-R1/V3, Llama 3.3, Qwen 2.5' },
  { name: 'deepinfra', label: 'DeepInfra', base_url: 'https://api.deepinfra.com/v1/openai', website_url: 'https://deepinfra.com', key_url: 'https://deepinfra.com/dash/api_keys', hint: 'Serverless inference for Llama 3.3, DeepSeek, Qwen' },
  { name: 'groq', label: 'Groq', base_url: 'https://api.groq.com/openai/v1', website_url: 'https://groq.com', key_url: 'https://console.groq.com/keys', hint: 'Ultra-fast LPU inference for Llama 3.3, Llama 3.1, Mixtral' },
  { name: 'cerebras', label: 'Cerebras', base_url: 'https://api.cerebras.ai/v1', website_url: 'https://cerebras.ai', key_url: 'https://cloud.cerebras.ai', hint: 'World-record fast inference for Llama 3.1/3.3' },
  { name: 'sambanova', label: 'SambaNova Cloud', base_url: 'https://api.sambanova.ai/v1', website_url: 'https://sambanova.ai', key_url: 'https://cloud.sambanova.ai/apis', hint: 'Fast SN40L inference for Llama 3.3 70B & 405B' },
  { name: 'github-models', label: 'GitHub Models', base_url: 'https://models.github.ai/inference', website_url: 'https://github.com/marketplace/models', key_url: 'https://github.com/settings/tokens', hint: 'Free tier for personal GitHub accounts via Personal Access Token' },
  { name: 'gemini-studio', label: 'Google Gemini Studio', base_url: 'https://generativelanguage.googleapis.com/v1beta/openai', website_url: 'https://aistudio.google.com', key_url: 'https://aistudio.google.com/app/apikey', hint: 'Generous free tier for Gemini 2.5/3 Flash & Pro' },
  { name: 'cohere', label: 'Cohere', base_url: 'https://api.cohere.ai/compatibility/v1', website_url: 'https://cohere.com', key_url: 'https://dashboard.cohere.com/api-keys', hint: 'Free trial key for Command R & R+ models' },
  { name: 'mistral', label: 'Mistral AI', base_url: 'https://api.mistral.ai/v1', website_url: 'https://mistral.ai', key_url: 'https://console.mistral.ai/api-keys', hint: 'La Plateforme API for Mistral Small, Codestral' },
  { name: 'cloudflare', label: 'Cloudflare Workers AI', base_url: 'https://api.cloudflare.com/client/v4/accounts/.../ai/v1', website_url: 'https://ai.cloudflare.com', key_url: 'https://dash.cloudflare.com/profile/api-tokens', hint: 'Free daily neurons on Cloudflare Workers AI' },
  { name: 'openrouter', label: 'OpenRouter', base_url: 'https://openrouter.ai/api/v1', website_url: 'https://openrouter.ai', key_url: 'https://openrouter.ai/keys', hint: 'Free tier for :free models (Llama, Gemini, Qwen)' },
  { name: 'Kilo', label: 'Kilo Gateway', base_url: 'https://api.kilo.ai/api/gateway', website_url: 'https://kilo.ai', key_url: 'https://kilo.ai', hint: 'Free gateway for open models' },
  { name: 'opencode zen', label: 'OpenCode Zen', base_url: 'https://opencode.ai/zen/v1', website_url: 'https://opencode.ai', key_url: 'https://opencode.ai', hint: 'Free Zen endpoints for community models' },
  { name: 'llm7', label: 'LLM7', base_url: 'https://api.llm7.io/v1', website_url: 'https://llm7.io', key_url: 'https://llm7.io', hint: 'Free OpenAI-compatible community gateway' },
  { name: 'Empero', label: 'Empero', base_url: 'https://free.empero.org/v1', website_url: 'https://empero.org', key_url: 'https://empero.org', hint: 'Free community LLM endpoints' },
  { name: 'zai_api', label: 'Z.AI / Zhipu GLM', base_url: 'https://api.z.ai/api/coding/paas/v4', website_url: 'https://z.ai', key_url: 'https://z.ai', hint: 'GLM coding endpoints' },
  { name: 'ovhcloud', label: 'OVHcloud AI Endpoints', base_url: 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1', website_url: 'https://www.ovhcloud.com', key_url: 'https://www.ovh.com/manager', hint: 'Free tier for Llama 3.3 and open models (EU sovereign/GDPR)' },
  { name: 'novita', label: 'Novita AI', base_url: 'https://api.novita.ai/openai/v1', website_url: 'https://novita.ai', key_url: 'https://novita.ai/settings/key-management', hint: 'Fast inference for open-source LLMs with free trial credits' },
  { name: 'pollinations', label: 'Pollinations.ai', base_url: 'https://text.pollinations.ai/openai', website_url: 'https://pollinations.ai', key_url: 'https://pollinations.ai', hint: 'Free OpenAI-compatible community endpoint for text and code (no key or dummy key)' },
];

/** Lookup table of provider website links and API key portals. */
const KNOWN_PROVIDER_LINKS: Record<string, { website: string; key_portal?: string }> = {
  huggingface: { website: 'https://huggingface.co', key_portal: 'https://huggingface.co/settings/tokens' },
  modelscope: { website: 'https://modelscope.cn', key_portal: 'https://modelscope.cn/my/myaccesstoken' },
  siliconflow: { website: 'https://siliconflow.cn', key_portal: 'https://cloud.siliconflow.cn/account/ak' },
  together: { website: 'https://www.together.ai', key_portal: 'https://api.together.xyz/settings/api-keys' },
  fireworks: { website: 'https://fireworks.ai', key_portal: 'https://fireworks.ai/api-keys' },
  hyperbolic: { website: 'https://hyperbolic.xyz', key_portal: 'https://app.hyperbolic.xyz/settings' },
  deepinfra: { website: 'https://deepinfra.com', key_portal: 'https://deepinfra.com/dash/api_keys' },
  ovhcloud: { website: 'https://www.ovhcloud.com', key_portal: 'https://www.ovh.com/manager' },
  novita: { website: 'https://novita.ai', key_portal: 'https://novita.ai/settings/key-management' },
  pollinations: { website: 'https://pollinations.ai' },
  groq: { website: 'https://groq.com', key_portal: 'https://console.groq.com/keys' },
  cerebras: { website: 'https://cerebras.ai', key_portal: 'https://cloud.cerebras.ai' },
  sambanova: { website: 'https://sambanova.ai', key_portal: 'https://cloud.sambanova.ai/apis' },
  'github-models': { website: 'https://github.com/marketplace/models', key_portal: 'https://github.com/settings/tokens' },
  gemini_studio: { website: 'https://aistudio.google.com', key_portal: 'https://aistudio.google.com/app/apikey' },
  'gemini-studio': { website: 'https://aistudio.google.com', key_portal: 'https://aistudio.google.com/app/apikey' },
  cohere: { website: 'https://cohere.com', key_portal: 'https://dashboard.cohere.com/api-keys' },
  mistral: { website: 'https://mistral.ai', key_portal: 'https://console.mistral.ai/api-keys' },
  cloudflare: { website: 'https://ai.cloudflare.com', key_portal: 'https://dash.cloudflare.com/profile/api-tokens' },
  openrouter: { website: 'https://openrouter.ai', key_portal: 'https://openrouter.ai/keys' },
  Kilo: { website: 'https://kilo.ai' },
  'opencode zen': { website: 'https://opencode.ai' },
  llm7: { website: 'https://llm7.io' },
  Empero: { website: 'https://empero.org' },
  zai_api: { website: 'https://z.ai' },
  'claude-code': { website: 'https://anthropic.com' },
  'google-antigravity': { website: 'https://cloud.google.com' },
  'openai-codex': { website: 'https://chatgpt.com' },
  nvidia: { website: 'https://build.nvidia.com', key_portal: 'https://build.nvidia.com' },
};

function getProviderLinks(name: string, baseUrl?: string) {
  const match = KNOWN_PROVIDER_LINKS[name] || FREE_PROVIDER_PRESETS.find((p) => p.name === name || p.label.toLowerCase().includes(name.toLowerCase()));
  if (match) return { website: 'website' in match ? match.website : match.website_url, key_portal: 'key_portal' in match ? match.key_portal : match.key_url };
  try {
    if (baseUrl && baseUrl.startsWith('http')) {
      const u = new URL(baseUrl);
      return { website: `${u.protocol}//${u.host}` };
    }
  } catch {
    // fallback
  }
  return {};
}

const EMPTY_PROVIDER: RegistryProvider = { name: '', tier: 3, base_url: '', api_key_env: '', api_key: '', enabled: true, models: [], access_type: 'api_key', cost_type: 'free', api_format: 'openai', auth_config: {} };


const PAGES: { id: Page; label: string; section: string; icon: React.ReactNode }[] = [
  { id: 'agents', label: 'Agents', section: 'Monitoring', icon: <Bot size={15} /> },
  { id: 'overview', label: 'Overview', section: 'Monitoring', icon: <LayoutDashboard size={15} /> },
  { id: 'messages', label: 'Messages', section: 'Monitoring', icon: <MessageSquare size={15} /> },
  { id: 'routing', label: 'Routing', section: 'Manage', icon: <SlidersHorizontal size={15} /> },
  { id: 'tasks', label: 'Tasks', section: 'Manage', icon: <Layers size={15} /> },
  { id: 'playground', label: 'Playground', section: 'Manage', icon: <Terminal size={15} /> },
  { id: 'pricing', label: 'LLM Pricing', section: 'Manage', icon: <DollarSign size={15} /> },
  { id: 'users', label: 'Users', section: 'Administration', icon: <UsersRound size={15} /> },
  { id: 'profiles', label: 'Access Profiles', section: 'Administration', icon: <ShieldCheck size={15} /> },
];

/** Modules a profile can grant to non-admin users. Users/Profiles management
 * stays admin-only (like ForgeHub), so it's not part of the matrix. */
const PROFILE_MODULES: Page[] = ['agents', 'overview', 'messages', 'routing', 'tasks', 'playground', 'pricing'];

// Recommended virtual model for each built-in task — paste these ids into
// your agent/client's own model settings. forgerouter/auto classifies the
// request on the fly regardless.
const DEFAULT_TASK_MAP: { task: string; hint: string; model: string }[] = [
  { task: 'Main model', hint: 'primary session model', model: 'forgerouter/auto' },
  { task: 'Vision', hint: 'image analysis', model: 'forgerouter/standard' },
  { task: 'Web Extract', hint: 'page summarization', model: 'forgerouter/simple' },
  { task: 'Compression', hint: 'context compaction', model: 'forgerouter/simple' },
  { task: 'Skills Hub', hint: 'skill search', model: 'forgerouter/simple' },
  { task: 'Approval', hint: 'smart auto-approve', model: 'forgerouter/simple' },
  { task: 'MCP', hint: 'MCP tool routing', model: 'forgerouter/standard' },
  { task: 'Title Gen', hint: 'session titles', model: 'forgerouter/simple' },
  { task: 'TTS Audio Tags', hint: 'Gemini TTS tag insertion', model: 'forgerouter/audio' },
  { task: 'Triage Specifier', hint: 'Kanban spec fleshing', model: 'forgerouter/standard' },
  { task: 'Kanban Decomposer', hint: 'task decomposition', model: 'forgerouter/complex' },
  { task: 'Profile Describer', hint: 'auto profile descriptions', model: 'forgerouter/simple' },
  { task: 'Curator', hint: 'skill-usage review', model: 'forgerouter/standard' },
];

async function copyText(value: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    // clipboard API unavailable (older browsers / non-secure context)
    const area = document.createElement('textarea');
    area.value = value;
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  }
}

function formatTokens(value: number | null): string {
  if (!value) return '0';
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return `${value}`;
}

function formatCost(value: number): string {
  if (!value) return '$0.00';
  if (value < 0.01) return '< $0.01';
  return `$${value.toFixed(2)}`;
}

function formatMetricValue(value: number, metric: UsageMetric): string {
  if (metric === 'cost' || metric === 'reference_cost') return formatCost(value);
  if (metric === 'tokens') return formatTokens(value);
  return `${value}`;
}

function formatLatency(ms: number | null): string {
  if (!ms) return '-';
  const seconds = ms / 1000;
  return `${seconds.toFixed(seconds < 1 ? 2 : 1)}s`;
}

function formatContextLength(tokens: number | null | undefined): string {
  if (!tokens) return '-';
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 ? 1 : 0)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return String(tokens);
}

function formatDate(iso: string | null): string {
  if (!iso) return '-';
  // dd/mm/yyyy hh:mm, keeping the server-reported (America/Sao_Paulo) time as-is.
  return `${iso.slice(8, 10)}/${iso.slice(5, 7)}/${iso.slice(0, 4)} ${iso.slice(11, 16)}`;
}

// Time left before the watchdog's own cooldown (its loop guard) allows another
// real rescan — last_scan_at + cooldown_seconds, clamped at 0 once it's elapsed
// (the next 60s tick, not this countdown, is what actually fires it).
function watchdogNextRetryLabel(watchdog: WatchdogStatus): string {
  if (!watchdog.last_scan_at || !watchdog.cooldown_seconds) return 'soon';
  const elapsedMs = Date.now() - new Date(watchdog.last_scan_at).getTime();
  const remaining = Math.max(0, Math.round(watchdog.cooldown_seconds - elapsedMs / 1000));
  if (remaining === 0) return 'next check';
  return remaining >= 60 ? `${Math.ceil(remaining / 60)}min` : `${remaining}s`;
}

function formatDay(day: string | undefined): string {
  if (!day) return '';
  return `${day.slice(8, 10)}/${day.slice(5, 7)}/${day.slice(0, 4)}`;
}

// structuredClone is missing in older browsers; the registry is plain JSON, so this is safe.
function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function Logo({ size = 26 }: { size?: number }) {
  // ForgeRouter mark: three model routes forged into one endpoint.
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-label="ForgeRouter">
      <defs><linearGradient id="frg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#8b5cf6" /><stop offset="1" stopColor="#2dd4bf" /></linearGradient></defs>
      <rect x="2" y="2" width="60" height="60" rx="14" fill="#0c0c0f" stroke="url(#frg)" strokeWidth="2.5" />
      <path d="M14 20H26C34 20 34 32 42 32H50" fill="none" stroke="url(#frg)" strokeWidth="4" strokeLinecap="round" />
      <path d="M14 32H50" fill="none" stroke="url(#frg)" strokeWidth="4" strokeLinecap="round" opacity=".85" />
      <path d="M14 44H26C34 44 34 32 42 32" fill="none" stroke="url(#frg)" strokeWidth="4" strokeLinecap="round" opacity=".7" />
      <circle cx="14" cy="20" r="3.5" fill="#8b5cf6" />
      <circle cx="14" cy="32" r="3.5" fill="#a78bfa" />
      <circle cx="14" cy="44" r="3.5" fill="#c4b5fd" />
      <circle cx="50" cy="32" r="4.5" fill="#2dd4bf" />
    </svg>
  );
}

type PermFlags = { can_view: boolean; can_query: boolean; can_write: boolean; can_delete: boolean };
type AuthUser = {
  username: string;
  must_change_password: boolean;
  is_admin?: boolean;
  full_name?: string | null;
  email?: string | null;
  avatar_data_url?: string | null;
  permissions?: Record<string, PermFlags>;
};
type AdminUser = {
  user_id: number; username: string; full_name: string | null; email: string | null;
  is_admin: boolean; is_active: boolean; must_change_password: boolean; profile_id: number | null;
  profile_name: string | null; created_at: string | null; avatar_data_url: string | null;
};
type AdminProfile = {
  profile_id: number; name: string; description: string | null;
  permissions: Array<PermFlags & { module: string }>; users_count: number; created_at: string | null;
};
type SystemVersion = {
  app_version: string;
  git_sha: string;
  git_commit_url: string | null;
  build_date: string;
  postgres_version: string | null;
  latest_migration_bundled: string | null;
  github_repo_url: string;
};

// ---------------------------------------------------------------------------
// Auth screens: same visual language as the ForgeHub login (two-column layout,
// animated contextual backdrop, glass form card), rebuilt with plain CSS +
// SMIL since this frontend has no Tailwind/framer-motion. The backdrop tells
// ForgeRouter's own story: many provider nodes routed through one hub to a
// single endpoint — the same idea as the Logo mark.
// ---------------------------------------------------------------------------

/** Deterministic rising-dot field (module-level so it never reshuffles). */
const AUTH_RISERS = Array.from({ length: 16 }, (_, i) => ({
  left: `${(i * 137.5) % 100}%`, // golden-angle spread, no clumping
  size: 2 + ((i * 7) % 3),
  duration: 8 + ((i * 13) % 7),
  delay: (i * 1.9) % 9,
}));

/** Routing graph: provider nodes → relays → router hub → one endpoint.
 * Pulses travel the edges via SMIL animateMotion (no JS animation lib). */
function RouteGraph() {
  const nodes = {
    providers: [
      { x: 180, y: 240 },
      { x: 180, y: 380 },
      { x: 180, y: 520 },
      { x: 180, y: 660 },
    ],
    relays: [
      { x: 520, y: 310 },
      { x: 520, y: 590 },
    ],
    hub: { x: 780, y: 450 },
    endpoint: { x: 1180, y: 450 },
  };
  const edges: Array<[{ x: number; y: number }, { x: number; y: number }]> = [
    [nodes.providers[0], nodes.relays[0]],
    [nodes.providers[1], nodes.relays[0]],
    [nodes.providers[2], nodes.relays[1]],
    [nodes.providers[3], nodes.relays[1]],
    [nodes.relays[0], nodes.hub],
    [nodes.relays[1], nodes.hub],
    [nodes.hub, nodes.endpoint],
  ];
  const pulses = [
    { edge: edges[0], dur: 3.6, begin: 0 },
    { edge: edges[2], dur: 4.2, begin: 1.3 },
    { edge: edges[3], dur: 3.2, begin: 2.6 },
    { edge: edges[4], dur: 2.8, begin: 0.8 },
    { edge: edges[5], dur: 3.0, begin: 2.0 },
  ];
  return (
    <svg className="authGraph" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {edges.map(([a, b], i) => (
        <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="rgba(167,139,250,0.16)" strokeWidth={i === 6 ? 2 : 1} />
      ))}
      {nodes.providers.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r={4} fill="rgba(167,139,250,0.45)" />
      ))}
      {nodes.relays.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r={3.5} fill="rgba(139,92,246,0.4)" />
      ))}
      {/* Router hub: breathing violet ring (the Logo's forge point) */}
      <circle cx={nodes.hub.x} cy={nodes.hub.y} r={9} fill="none" stroke="rgba(139,92,246,0.55)" strokeWidth={1.5}>
        <animate attributeName="r" values="8;11;8" dur="4s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;1;0.5" dur="4s" repeatCount="indefinite" />
      </circle>
      {/* Single endpoint: teal ring, the one API every route converges to */}
      <circle cx={nodes.endpoint.x} cy={nodes.endpoint.y} r={7} fill="none" stroke="rgba(45,212,191,0.6)" strokeWidth={1.5}>
        <animate attributeName="opacity" values="0.4;1;0.4" dur="3s" repeatCount="indefinite" />
      </circle>
      <circle cx={nodes.endpoint.x} cy={nodes.endpoint.y} r={3} fill="rgba(45,212,191,0.7)" />
      {pulses.map(({ edge: [a, b], dur, begin }, i) => (
        <circle key={i} r={3} fill="#a78bfa" style={{ filter: 'drop-shadow(0 0 4px rgba(167,139,250,0.9))' }}>
          <animateMotion dur={`${dur}s`} begin={`${begin}s`} repeatCount="indefinite" path={`M${a.x} ${a.y} L${b.x} ${b.y}`} />
          <animate attributeName="opacity" values="0;1;1;0" dur={`${dur}s`} begin={`${begin}s`} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Hub → endpoint carries the answer: teal pulse, slightly faster */}
      <circle r={3.5} fill="#2dd4bf" style={{ filter: 'drop-shadow(0 0 5px rgba(45,212,191,0.9))' }}>
        <animateMotion dur="2.4s" repeatCount="indefinite" path={`M${nodes.hub.x} ${nodes.hub.y} L${nodes.endpoint.x} ${nodes.endpoint.y}`} />
        <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

function AuthBackdrop() {
  return (
    <div className="authBackdrop" aria-hidden="true">
      <div className="authGridLayer" />
      <div className="authWatermark"><Logo size={560} /></div>
      <RouteGraph />
      <div className="authOrb orbA" />
      <div className="authOrb orbB" />
      <div className="authOrb orbC" />
      {AUTH_RISERS.map((r, i) => (
        <span
          key={i}
          className="authRiser"
          style={{ left: r.left, width: r.size, height: r.size, animationDuration: `${r.duration}s`, animationDelay: `${r.delay}s` }}
        />
      ))}
      <div className="authVignette" />
    </div>
  );
}

const AUTH_PILLARS = [
  {
    icon: <Network size={18} />,
    title: 'One OpenAI-compatible API',
    text: 'A single /v1 endpoint routing across local Ollama, Groq, OpenRouter and Mistral.',
  },
  {
    icon: <HeartPulse size={18} />,
    title: 'Health-based routing',
    text: 'Real chat probes catch silent failures — unhealthy models never receive traffic.',
  },
  {
    icon: <Shuffle size={18} />,
    title: 'Automatic fallback',
    text: 'Failures cascade down the tier chain; rate-limited models cool down and re-enter.',
  },
];

/** Two-column auth shell shared by the login and first-access screens. */
function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="authViewport">
      <AuthBackdrop />
      <div className="authColumns">
        <div className="authBrandPanel">
          <div className="authBrand"><Logo size={38} /><span><p className="eyebrow">LLM Gateway</p><strong>ForgeRouter</strong></span></div>
          <div className="authHero">
            <p className="authBadge"><span className="badgeDot" />LLM routing control plane</p>
            <h1>Route every request to the best available model.</h1>
            <div className="authPillars">
              {AUTH_PILLARS.map((p) => (
                <div key={p.title} className="authPillar">
                  <span className="pillarIcon">{p.icon}</span>
                  <span>
                    <p className="pillarTitle">{p.title}</p>
                    <p className="pillarText">{p.text}</p>
                  </span>
                </div>
              ))}
            </div>
          </div>
          <p className="authFootLine mono">ForgeRouter · /v1/chat/completions · providers → health → tiers → fallback</p>
        </div>
        <div className="authFormPanel">
          <div className="authFormInner">{children}</div>
        </div>
      </div>
    </div>
  );
}

// Unset until the key pair is provisioned for this domain (see CLAUDE.md's
// reCAPTCHA note) -- the widget below only renders once a site key exists,
// so login keeps working exactly as before in the meantime.
const RECAPTCHA_SITE_KEY = import.meta.env.VITE_RECAPTCHA_SITE_KEY as string | undefined;

function LoginScreen({ onLogin }: { onLogin: (token: string, user: AuthUser, password: string) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recaptchaToken, setRecaptchaToken] = useState<string | null>(null);
  const recaptchaRef = useRef<ReCAPTCHA>(null);

  async function submit() {
    if (pending || !username.trim() || !password) return;
    if (RECAPTCHA_SITE_KEY && !recaptchaToken) return;
    setPending(true);
    setError(null);
    try {
      const res = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password, recaptcha_token: recaptchaToken }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message ?? 'Login failed');
      onLogin(
        data.token,
        { username: data.username, must_change_password: data.must_change_password, is_admin: data.is_admin, full_name: data.full_name, email: data.email, avatar_data_url: data.avatar_data_url, permissions: data.permissions },
        password,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
      recaptchaRef.current?.reset();
      setRecaptchaToken(null);
    } finally {
      setPending(false);
    }
  }

  return (
    <AuthLayout>
      <div className="authCard glass">
        <div className="authCardBrand"><Logo size={44} /><strong>ForgeRouter</strong></div>
        <h2>Welcome back</h2>
        <p className="muted authSub">Sign in to the routing dashboard</p>
        {error && <div className="alert">{error}</div>}
        <label>Username
          <span className="inputWrap">
            <User size={14} />
            <input autoFocus value={username} onChange={(e) => setUsername(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }} placeholder="Enter your username" />
          </span>
        </label>
        <label>Password
          <span className="inputWrap">
            <KeyRound size={14} />
            <input type={showPassword ? 'text' : 'password'} value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }} />
            <button type="button" className="inputEye" tabIndex={-1} onClick={() => setShowPassword((s) => !s)}>
              {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </span>
        </label>
        {RECAPTCHA_SITE_KEY && (
          <div className="authRecaptcha">
            <ReCAPTCHA
              ref={recaptchaRef}
              sitekey={RECAPTCHA_SITE_KEY}
              theme="dark"
              onChange={(token) => setRecaptchaToken(token)}
              onExpired={() => setRecaptchaToken(null)}
            />
          </div>
        )}
        <button
          className="button"
          disabled={pending || !username.trim() || !password || (Boolean(RECAPTCHA_SITE_KEY) && !recaptchaToken)}
          onClick={() => void submit()}
        >
          {pending ? 'Signing in…' : 'Sign in'}
        </button>
      </div>
      <p className="authRestrict">Restricted area — administrator access only.</p>
    </AuthLayout>
  );
}

function ChangeCredentialsScreen({ token, username, initialPassword, onDone }: { token: string; username: string; initialPassword: string; onDone: (username: string) => void }) {
  const [currentPassword, setCurrentPassword] = useState(initialPassword);
  const [newUsername, setNewUsername] = useState(username);
  const [newPassword, setNewPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (pending) return;
    if (newPassword.trim().length < 4) { setError('The new password needs at least 4 characters.'); return; }
    if (newPassword !== confirm) { setError('Password confirmation does not match.'); return; }
    setPending(true);
    setError(null);
    try {
      const res = await fetch('/auth/change-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ current_password: currentPassword, new_username: newUsername.trim(), new_password: newPassword.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message ?? 'Failed to change the login');
      onDone(data.username);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change the login');
    } finally {
      setPending(false);
    }
  }

  return (
    <AuthLayout>
      <div className="authCard glass">
        <div className="authCardBrand"><Logo size={44} /><strong>ForgeRouter</strong></div>
        <h2>First access — change your login</h2>
        <p className="muted authHint">The default credentials must be replaced before using the dashboard.</p>
        {error && <div className="alert">{error}</div>}
        <label>Current password<input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} /></label>
        <label>New username<input value={newUsername} onChange={(e) => setNewUsername(e.target.value)} /></label>
        <label>New password<input type="password" autoFocus value={newPassword} onChange={(e) => setNewPassword(e.target.value)} /></label>
        <label>Confirm new password<input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }} /></label>
        <button className="button" disabled={pending || !newPassword || !confirm} onClick={() => void submit()}>{pending ? 'Saving…' : 'Save new login'}</button>
      </div>
    </AuthLayout>
  );
}

type FetchJson = (path: string, init?: RequestInit) => Promise<any>;

type ThemePref = 'light' | 'dark' | 'system';

/** Reads a picked photo, center-crops it square and downscales to 256px,
 * returning a compact data: URL — avatars live inline in the users row
 * (like ForgeHub's avatar_data_url), so they must stay small. */
function readAvatarFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const size = 256;
      const min = Math.min(img.width, img.height);
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) { URL.revokeObjectURL(url); reject(new Error('Canvas unavailable')); return; }
      ctx.drawImage(img, (img.width - min) / 2, (img.height - min) / 2, min, min, 0, 0, size, size);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL('image/jpeg', 0.85));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('The selected file is not a valid image')); };
    img.src = url;
  });
}

/** Circular avatar: the user's photo when set, otherwise their initial. */
function UserAvatar({ username, avatarUrl, size }: { username: string; avatarUrl?: string | null; size?: 'sm' | 'big' }) {
  return (
    <span className={`userAvatar${size ? ` ${size}` : ''}`}>
      {avatarUrl ? <img src={avatarUrl} alt="" /> : username.charAt(0).toUpperCase()}
    </span>
  );
}

function AgentAvatar({ name, avatarUrl, size = 'sm' }: {
  name: string;
  avatarUrl?: string | null;
  size?: 'xs' | 'sm' | 'md' | 'lg';
}) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [avatarUrl]);
  const showImage = Boolean(avatarUrl) && !failed;
  return (
    <span className={`agentAvatar ${size}`} aria-hidden="true">
      {showImage ? <img src={avatarUrl ?? ''} alt="" onError={() => setFailed(true)} /> : name.charAt(0).toUpperCase() || <Bot size={14} />}
    </span>
  );
}

function AgentIdentity({ name, avatarUrl, size = 'sm', className = '' }: {
  name: string;
  avatarUrl?: string | null;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  className?: string;
}) {
  return (
    <span className={`agentIdentity${className ? ` ${className}` : ''}`}>
      <AgentAvatar name={name} avatarUrl={avatarUrl} size={size} />
      <span className="agentIdentityName">{name}</span>
    </span>
  );
}

function AgentSelect({ agents, value, onValueChange, includeAll = false, label, placeholder = 'Select agent' }: {
  agents: AgentInfo[];
  value: string;
  onValueChange: (value: string) => void;
  includeAll?: boolean;
  label: string;
  placeholder?: string;
}) {
  const selected = agents.find((agent) => agent.name === value);
  return (
    <Select.Root value={value || undefined} onValueChange={onValueChange}>
      <Select.Trigger className="agentSelectTrigger" aria-label={label} title={label}>
        {value === 'all' && includeAll ? (
          <span className="agentIdentity"><span className="agentAvatar sm aggregate"><UsersRound size={14} /></span><span className="agentIdentityName">All agents</span></span>
        ) : selected ? (
          <AgentIdentity name={selected.name} avatarUrl={selected.avatar_data_url} />
        ) : (
          <span className="muted">{placeholder}</span>
        )}
        <Select.Icon><ChevronDown size={14} /></Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="agentSelectContent" position="popper" sideOffset={6} collisionPadding={10}>
          <Select.Viewport className="agentSelectViewport">
            {includeAll && (
              <Select.Item className="agentSelectItem" value="all">
                <Select.ItemText><span className="agentIdentity"><span className="agentAvatar sm aggregate"><UsersRound size={14} /></span><span className="agentIdentityName">All agents</span></span></Select.ItemText>
                <Select.ItemIndicator><Check size={13} /></Select.ItemIndicator>
              </Select.Item>
            )}
            {agents.map((agent) => (
              <Select.Item className="agentSelectItem" key={agent.name} value={agent.name}>
                <Select.ItemText><AgentIdentity name={agent.name} avatarUrl={agent.avatar_data_url} /></Select.ItemText>
                <Select.ItemIndicator><Check size={13} /></Select.ItemIndicator>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

/** Photo picker used by the account modal and the admin user form. Supports
 * both the native file dialog and drag-and-drop — dragging a file onto the
 * avatar works even when the OS file-open dialog itself is unusable (e.g. a
 * broken/collapsed Windows Explorer sidebar with no folder navigation). */
function AvatarPicker({ username, value, onChange, onError }: {
  username: string;
  value: string | null;
  onChange: (next: string | null) => void;
  onError: (message: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const dragDepth = useRef(0);

  async function applyFile(file: File | null | undefined) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      onError('That file is not an image.');
      return;
    }
    try {
      onChange(await readAvatarFile(file));
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Could not read the image');
    }
  }

  async function handlePick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    await applyFile(file);
  }

  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault();
    dragDepth.current += 1;
    setDragOver(true);
  }
  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
  }
  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragOver(false);
  }
  async function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    dragDepth.current = 0;
    setDragOver(false);
    await applyFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div
      className={`avatarPicker${dragOver ? ' dragOver' : ''}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={(e) => void handleDrop(e)}
    >
      <div className="avatarPickerRow">
        <UserAvatar username={username || '?'} avatarUrl={value} size="big" />
        <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => void handlePick(e)} />
        <div className="actions">
          <button type="button" className="button secondary" onClick={() => inputRef.current?.click()}><ImagePlus size={14} /> {value ? 'Change photo' : 'Add photo'}</button>
          {value && <button type="button" className="iconButton danger" title="Remove photo" onClick={() => onChange(null)}><Trash2 size={13} /></button>}
        </div>
      </div>
      <p className="avatarDropHint">{dragOver ? 'Drop to use this photo' : 'or drag an image here'}</p>
    </div>
  );
}

/** Dark is the app's native look, so it's the default; `light` swaps the
 * neutral CSS variables (see :root/html.light in style.css). Auth screens
 * pin their own dark tokens and ignore this. */
function applyTheme(pref: ThemePref) {
  const dark = pref === 'dark' || (pref === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('light', !dark);
}

const THEME_OPTIONS: Array<{ id: ThemePref; label: string; icon: React.ReactNode }> = [
  { id: 'light', label: 'Light', icon: <Sun size={14} /> },
  { id: 'dark', label: 'Dark', icon: <Moon size={14} /> },
  { id: 'system', label: 'System', icon: <Monitor size={14} /> },
];

/** Avatar menu in the sidebar footer — same process as ForgeHub's
 * UserSettingsMenu: account editing, password change, theme choice. */
function UserSettingsMenu({ authUser, fetchJson, onUserUpdated, themePref, onThemeChange, collapsed, autoRefresh, onToggleAutoRefresh, onLogout }: {
  authUser: AuthUser;
  fetchJson: FetchJson;
  onUserUpdated: (user: AuthUser) => void;
  themePref: ThemePref;
  onThemeChange: (pref: ThemePref) => void;
  collapsed: boolean;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
  onLogout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [modal, setModal] = useState<'account' | 'password' | 'about' | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [fullName, setFullName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemVersion, setSystemVersion] = useState<SystemVersion | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (event: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open]);

  function openAccount() {
    setFullName(authUser.full_name ?? '');
    setUsername(authUser.username);
    setEmail(authUser.email ?? '');
    setAvatarPreview(authUser.avatar_data_url ?? null);
    setError(null);
    setModal('account');
    setOpen(false);
  }

  function openPassword() {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setError(null);
    setModal('password');
    setOpen(false);
  }

  function openAbout() {
    setSystemVersion(null);
    setVersionError(null);
    setModal('about');
    setOpen(false);
    fetchJson('/admin/system/version')
      .then((data) => setSystemVersion(data as SystemVersion))
      .catch((err) => setVersionError(err instanceof Error ? err.message : 'Failed to load system version'));
  }

  async function saveAccount() {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      const me = await fetchJson('/auth/me', {
        method: 'PATCH',
        body: JSON.stringify({ full_name: fullName, username, email, avatar_data_url: avatarPreview ?? '' }),
      });
      onUserUpdated(me as AuthUser);
      setModal(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save account');
    } finally {
      setPending(false);
    }
  }

  async function savePassword() {
    if (pending) return;
    if (newPassword.trim().length < 4) { setError('The new password needs at least 4 characters.'); return; }
    if (newPassword !== confirmPassword) { setError('Password confirmation does not match.'); return; }
    setPending(true);
    setError(null);
    try {
      // new_username '' keeps the current username (see /auth/change-password).
      await fetchJson('/auth/change-password', {
        method: 'POST',
        body: JSON.stringify({ current_password: currentPassword, new_username: '', new_password: newPassword.trim() }),
      });
      setModal(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change the password');
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="userMenuWrap" ref={wrapRef}>
      <button className={`userRowTrigger${collapsed ? ' collapsed' : ''}`} onClick={() => setOpen((v) => !v)} title="Account & settings">
        <UserAvatar username={authUser.username} avatarUrl={authUser.avatar_data_url} />
        {!collapsed && (
          <span className="userMeta">
            <p className="userName">{authUser.full_name || authUser.username}</p>
            {authUser.is_admin ? (
              <p className="userRole"><ShieldCheck size={10} /> Admin</p>
            ) : (
              <p className="userRole muted">User</p>
            )}
          </span>
        )}
      </button>
      {open && (
        <div className="userMenu">
          <button onClick={openAccount}><User size={14} /> Conta</button>
          <button onClick={openPassword}><KeyRound size={14} /> Alterar senha</button>
          <button onClick={openAbout}><Info size={14} /> Sobre</button>
          <p className="userMenuSection">Tema</p>
          {THEME_OPTIONS.map((option) => (
            <button key={option.id} onClick={() => onThemeChange(option.id)}>
              {option.icon} {option.label}
              {themePref === option.id && <Check size={13} className="activeTheme" />}
            </button>
          ))}
          <div className="userMenuDivider" />
          <button
            onClick={() => { onToggleAutoRefresh(); setOpen(false); }}
            title={autoRefresh ? 'Auto-refresh on (every 5s) — click to turn the sync off' : 'Auto-refresh off — data only reloads on demand; click to turn the sync back on'}
          >
            {autoRefresh ? <RefreshCw size={14} /> : <Pause size={14} />} {autoRefresh ? 'Sync: on' : 'Sync: off'}
          </button>
          <div className="userMenuDivider" />
          <button className="userMenuLogout" onClick={() => { setOpen(false); onLogout(); }}>
            <LogOut size={14} /> Sair
          </button>
        </div>
      )}
      {modal === 'account' && createPortal(
        <div className="modalOverlay" onClick={() => setModal(null)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <h2>Conta</h2>
            {error && <div className="alert">{error}</div>}
            <AvatarPicker username={username} value={avatarPreview} onChange={setAvatarPreview} onError={setError} />
            <label>Username<input value={username} onChange={(e) => setUsername(e.target.value)} /></label>
            <label>Full name<input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Shown in the sidebar" /></label>
            <label>E-mail<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" /></label>
            <div className="modalActions">
              <button className="button secondary" onClick={() => setModal(null)}>Cancel</button>
              <button className="button" disabled={pending || !username.trim()} onClick={() => void saveAccount()}>
                {pending ? <Loader2 size={15} className="spin" /> : <Save size={15} />} Save
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
      {modal === 'password' && createPortal(
        <div className="modalOverlay" onClick={() => setModal(null)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <h2>Alterar senha</h2>
            {error && <div className="alert">{error}</div>}
            <label>Current password<input type="password" autoFocus value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} /></label>
            <label>New password<input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} /></label>
            <label>Confirm new password<input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} /></label>
            <div className="modalActions">
              <button className="button secondary" onClick={() => setModal(null)}>Cancel</button>
              <button className="button" disabled={pending || !currentPassword || !newPassword || !confirmPassword} onClick={() => void savePassword()}>
                {pending ? <Loader2 size={15} className="spin" /> : <Save size={15} />} Save
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
      {modal === 'about' && createPortal(
        <div className="modalOverlay" onClick={() => setModal(null)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <h2>Sobre</h2>
            {versionError && <div className="alert">{versionError}</div>}
            {!systemVersion && !versionError && <p className="muted">Loading…</p>}
            {systemVersion && (
              <dl className="versionInfo">
                <dt>Versão</dt>
                <dd>{systemVersion.app_version}</dd>
                <dt>Commit</dt>
                <dd>
                  {systemVersion.git_commit_url ? (
                    <a href={systemVersion.git_commit_url} target="_blank" rel="noreferrer">{systemVersion.git_sha}</a>
                  ) : systemVersion.git_sha}
                </dd>
                <dt>Build</dt>
                <dd>{systemVersion.build_date}</dd>
                <dt>PostgreSQL</dt>
                <dd>{systemVersion.postgres_version ?? 'unavailable'}</dd>
                <dt>Última migration na imagem</dt>
                <dd>{systemVersion.latest_migration_bundled ?? 'unknown'}</dd>
                <dt>Repositório</dt>
                <dd><a href={systemVersion.github_repo_url} target="_blank" rel="noreferrer">GitHub</a></dd>
              </dl>
            )}
            <div className="modalActions">
              <button className="button secondary" onClick={() => setModal(null)}>Close</button>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

/** Users administration — ForgeHub's Users page process: admin-only CRUD,
 * new users always start with a forced password change, last active admin
 * can't be demoted/deactivated/deleted (also enforced server-side). */
function UsersAdminPage({ fetchJson, currentUsername }: { fetchJson: FetchJson; currentUsername: string }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [profiles, setProfiles] = useState<AdminProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<AdminUser | 'new' | null>(null);
  const [form, setForm] = useState({ username: '', full_name: '', email: '', password: '', is_admin: false, is_active: true, profile_id: '' });
  const [avatar, setAvatar] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function load() {
    try {
      const [u, p] = await Promise.all([fetchJson('/auth/users'), fetchJson('/auth/profiles')]);
      setUsers(u.users ?? []);
      setProfiles(p.profiles ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users');
    }
  }
  useEffect(() => { void load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  function openEditor(target: AdminUser | 'new') {
    setError(null);
    setEditing(target);
    if (target === 'new') {
      setForm({ username: '', full_name: '', email: '', password: '', is_admin: false, is_active: true, profile_id: '' });
      setAvatar(null);
    } else {
      setForm({
        username: target.username,
        full_name: target.full_name ?? '',
        email: target.email ?? '',
        password: '',
        is_admin: target.is_admin,
        is_active: target.is_active,
        profile_id: target.profile_id ? String(target.profile_id) : '',
      });
      setAvatar(target.avatar_data_url);
    }
  }

  async function save() {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      if (editing === 'new') {
        await fetchJson('/auth/users', {
          method: 'POST',
          body: JSON.stringify({
            username: form.username,
            password: form.password,
            full_name: form.full_name,
            email: form.email,
            is_admin: form.is_admin,
            profile_id: form.profile_id ? Number(form.profile_id) : null,
            avatar_data_url: avatar,
          }),
        });
      } else if (editing) {
        const body: Record<string, unknown> = {
          username: form.username,
          full_name: form.full_name,
          email: form.email,
          is_admin: form.is_admin,
          is_active: form.is_active,
          avatar_data_url: avatar ?? '',
        };
        if (form.password) body.password = form.password;
        if (form.profile_id) body.profile_id = Number(form.profile_id);
        else body.clear_profile = true;
        await fetchJson(`/auth/users/${editing.user_id}`, { method: 'PATCH', body: JSON.stringify(body) });
      }
      setEditing(null);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save user');
    } finally {
      setPending(false);
    }
  }

  async function remove(target: AdminUser) {
    if (!window.confirm(`Delete user "${target.username}"? This cannot be undone.`)) return;
    setError(null);
    try {
      await fetchJson(`/auth/users/${target.user_id}`, { method: 'DELETE' });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete user');
    }
  }

  return (
    <>
      <header className="pageHeader">
        <div>
          <h1>Users</h1>
          <p className="subtitle">Dashboard accounts. New users must change their password on first sign-in; access is granted via profiles unless the user is an administrator.</p>
        </div>
        <div className="actions">
          <button className="button" onClick={() => openEditor('new')}><Plus size={15} /> New user</button>
        </div>
      </header>
      {error && <div className="alert">{error}</div>}

      {editing !== null && (
        <section className="panel editor">
          <div className="panelHeader"><h2>{editing === 'new' ? 'New user' : `Edit ${editing.username}`}</h2></div>
          <div className="form">
            <AvatarPicker username={form.username} value={avatar} onChange={setAvatar} onError={setError} />
            <div className="formGrid" style={{ marginTop: 14 }}>
              <label>Username<input value={form.username} onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))} /></label>
              <label>Full name<input value={form.full_name} onChange={(e) => setForm((f) => ({ ...f, full_name: e.target.value }))} /></label>
              <label>E-mail<input type="email" value={form.email} onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))} placeholder="user@example.com" /></label>
              <label>{editing === 'new' ? 'Password' : 'New password (blank = keep current)'}
                <input type="password" value={form.password} onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))} placeholder={editing === 'new' ? 'min. 4 characters' : ''} />
              </label>
              <label>Access profile
                <select className="chip select big" value={form.profile_id} onChange={(e) => setForm((f) => ({ ...f, profile_id: e.target.value }))}>
                  <option value="">— none —</option>
                  {profiles.map((p) => <option key={p.profile_id} value={p.profile_id}>{p.name}</option>)}
                </select>
              </label>
            </div>
            <div className="formActions">
              <label className="check"><input type="checkbox" checked={form.is_admin} onChange={(e) => setForm((f) => ({ ...f, is_admin: e.target.checked }))} />Administrator (bypasses profiles)</label>
              {editing !== 'new' && (
                <label className="check"><input type="checkbox" checked={form.is_active} onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.checked }))} />Active</label>
              )}
              <span className="spacer" />
              <button className="button secondary" onClick={() => setEditing(null)}>Cancel</button>
              <button className="button" disabled={pending || !form.username.trim() || (editing === 'new' && form.password.length < 4)} onClick={() => void save()}>
                {pending ? <Loader2 size={15} className="spin" /> : <Save size={15} />} Save
              </button>
            </div>
          </div>
        </section>
      )}

      <Panel title={<><span className="panelIcon accent-violet"><UsersRound /></span>Accounts</>} meta={`${users.length} user${users.length === 1 ? '' : 's'}`}>
        <div className="row head usersRow"><span>Username</span><span>Full name</span><span>E-mail</span><span>Profile</span><span>Role</span><span>Status</span><span>Created</span><span /></div>
        {users.map((u) => (
          <div key={u.user_id} className="row usersRow">
            <span className="userCell">
              <UserAvatar username={u.username} avatarUrl={u.avatar_data_url} size="sm" />
              <span className="mono">{u.username}{u.username === currentUsername && <em className="muted small"> (you)</em>}</span>
            </span>
            <span>{u.full_name ?? '—'}</span>
            <span className="muted small">{u.email ?? '—'}</span>
            <span>{u.profile_name ?? (u.is_admin ? '—' : <em className="muted">none</em>)}</span>
            <span>{u.is_admin ? <i className="status healthy">Admin</i> : <i className="status unknown">User</i>}</span>
            <span>{u.is_active ? <i className="status healthy">Active</i> : <i className="status unhealthy">Inactive</i>}</span>
            <span className="muted small">{formatDate(u.created_at)}</span>
            <span className="rowActions">
              <button className="iconButton" title="Edit" onClick={() => openEditor(u)}><Pencil size={13} /></button>
              <button className="iconButton danger" title={u.username === currentUsername ? 'You cannot delete yourself' : 'Delete'} disabled={u.username === currentUsername} onClick={() => void remove(u)}><Trash2 size={13} /></button>
            </span>
          </div>
        ))}
      </Panel>
    </>
  );
}

/** Access Profiles administration — ForgeHub's profile process: a named set
 * of per-module view/query/write/delete flags, assigned to non-admin users. */
function ProfilesAdminPage({ fetchJson }: { fetchJson: FetchJson }) {
  const emptyMatrix = () => Object.fromEntries(PROFILE_MODULES.map((m) => [m, { can_view: false, can_query: false, can_write: false, can_delete: false }])) as Record<string, PermFlags>;
  const [profiles, setProfiles] = useState<AdminProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<AdminProfile | 'new' | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [matrix, setMatrix] = useState<Record<string, PermFlags>>(emptyMatrix());
  const [pending, setPending] = useState(false);

  async function load() {
    try {
      const data = await fetchJson('/auth/profiles');
      setProfiles(data.profiles ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load profiles');
    }
  }
  useEffect(() => { void load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  function openEditor(target: AdminProfile | 'new') {
    setError(null);
    setEditing(target);
    if (target === 'new') {
      setName(''); setDescription(''); setMatrix(emptyMatrix());
    } else {
      setName(target.name);
      setDescription(target.description ?? '');
      const next = emptyMatrix();
      for (const perm of target.permissions) {
        if (next[perm.module]) next[perm.module] = { can_view: perm.can_view, can_query: perm.can_query, can_write: perm.can_write, can_delete: perm.can_delete };
      }
      setMatrix(next);
    }
  }

  function toggle(module: string, flag: keyof PermFlags) {
    setMatrix((prev) => {
      const next = { ...prev, [module]: { ...prev[module], [flag]: !prev[module][flag] } };
      // Query/write/delete imply the page is visible; dropping view drops the rest.
      if (flag !== 'can_view' && next[module][flag]) next[module].can_view = true;
      if (flag === 'can_view' && !next[module].can_view) next[module] = { can_view: false, can_query: false, can_write: false, can_delete: false };
      return next;
    });
  }

  async function save() {
    if (pending) return;
    setPending(true);
    setError(null);
    const permissions = PROFILE_MODULES.map((module) => ({ module, ...matrix[module] }));
    try {
      if (editing === 'new') {
        await fetchJson('/auth/profiles', { method: 'POST', body: JSON.stringify({ name, description, permissions }) });
      } else if (editing) {
        await fetchJson(`/auth/profiles/${editing.profile_id}`, { method: 'PATCH', body: JSON.stringify({ name, description, permissions }) });
      }
      setEditing(null);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setPending(false);
    }
  }

  async function remove(target: AdminProfile) {
    const inUse = target.users_count ? ` ${target.users_count} user(s) will fall back to no profile.` : '';
    if (!window.confirm(`Delete profile "${target.name}"?${inUse}`)) return;
    setError(null);
    try {
      await fetchJson(`/auth/profiles/${target.profile_id}`, { method: 'DELETE' });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete profile');
    }
  }

  return (
    <>
      <header className="pageHeader">
        <div>
          <h1>Access Profiles</h1>
          <p className="subtitle">Named permission sets for non-admin users: which pages they see (view) and what they can do there (query, write, delete). Administrators bypass profiles.</p>
        </div>
        <div className="actions">
          <button className="button" onClick={() => openEditor('new')}><Plus size={15} /> New profile</button>
        </div>
      </header>
      {error && <div className="alert">{error}</div>}

      {editing !== null && (
        <section className="panel editor">
          <div className="panelHeader"><h2>{editing === 'new' ? 'New profile' : `Edit ${editing.name}`}</h2></div>
          <div className="form">
            <div className="formGrid">
              <label>Name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="Operators" /></label>
              <label>Description<input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Read-only monitoring access" /></label>
            </div>
            <h3>Permissions</h3>
            <div className="row head permRow"><span>Module</span><span>View</span><span>Query</span><span>Write</span><span>Delete</span></div>
            {PROFILE_MODULES.map((module) => (
              <div key={module} className="row permRow">
                <span className="permModule">{PAGES.find((p) => p.id === module)?.label ?? module}</span>
                {(['can_view', 'can_query', 'can_write', 'can_delete'] as const).map((flag) => (
                  <span key={flag}><input type="checkbox" checked={matrix[module][flag]} onChange={() => toggle(module, flag)} /></span>
                ))}
              </div>
            ))}
            <div className="formActions">
              <span className="spacer" />
              <button className="button secondary" onClick={() => setEditing(null)}>Cancel</button>
              <button className="button" disabled={pending || !name.trim()} onClick={() => void save()}>
                {pending ? <Loader2 size={15} className="spin" /> : <Save size={15} />} Save
              </button>
            </div>
          </div>
        </section>
      )}

      <Panel title={<><span className="panelIcon accent-teal"><ShieldCheck /></span>Profiles</>} meta={`${profiles.length} profile${profiles.length === 1 ? '' : 's'}`}>
        <div className="row head profilesRow"><span>Name</span><span>Description</span><span>Visible modules</span><span>Users</span><span>Created</span><span /></div>
        {profiles.map((p) => (
          <div key={p.profile_id} className="row profilesRow">
            <span>{p.name}</span>
            <span className="muted">{p.description ?? '—'}</span>
            <span className="caps">
              {p.permissions.filter((perm) => perm.can_view).map((perm) => <i key={perm.module} className="cap">{PAGES.find((pg) => pg.id === perm.module)?.label ?? perm.module}</i>)}
              {!p.permissions.some((perm) => perm.can_view) && <em className="muted small">none</em>}
            </span>
            <span>{p.users_count}</span>
            <span className="muted small">{formatDate(p.created_at)}</span>
            <span className="rowActions">
              <button className="iconButton" title="Edit" onClick={() => openEditor(p)}><Pencil size={13} /></button>
              <button className="iconButton danger" title="Delete" onClick={() => void remove(p)}><Trash2 size={13} /></button>
            </span>
          </div>
        ))}
        {!profiles.length && <div className="row"><span className="muted">No profiles yet — non-admin users see nothing until one is assigned.</span></div>}
      </Panel>
    </>
  );
}

// Client-side token generator for the agent registration form (same shape the server generates).
// The agent name is baked in as a PREFIX (<slug>_<random>), not a suffix — masked displays
// only ever show the first few characters of a secret, so the name has to be the part that
// survives truncation for a key pasted into the wrong agent's config to look wrong at a glance.
function generateAgentKey(name: string = ''): string {
  const bytes = new Uint8Array(30);
  crypto.getRandomValues(bytes);
  const base64 = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  const slug = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'agent';
  return `${slug}_${base64}`;
}

// Ready-to-paste connection snippets for coding tools that can be pointed at
// ForgeRouter instead of (or alongside) their default backend. Each snippet is
// rendered with a placeholder/masked key on screen — the real key only ever
// enters the clipboard string, via a fetch to the reveal endpoint at copy
// time, mirroring copyAgentKey's rule that a plaintext secret never touches
// React state or the DOM. `embedsSecret: false` (Codex) means the block
// itself never carries the key at all — Codex reads it from an env var named
// by `env_key`, so only the separate `secret` line needs the real value.
type AgentClientId = 'claude-code' | 'codex' | 'hermes-agent' | 'antigravity-cli';
type AgentClientDef = {
  id: AgentClientId;
  label: string;
  hint: string;
  filename: string;
  baseUrlKind: 'root' | 'v1'; // Anthropic wire: base URL is the API root, SDK appends /v1/messages. OpenAI-style (Codex, Hermes Agent): base URL already ends in /v1.
  embedsSecret: boolean;
  block: (baseUrl: string, keyToken: string) => string;
  secret?: { label: string; line: (key: string) => string };
};
const AGENT_CLIENTS: AgentClientDef[] = [
  {
    id: 'claude-code',
    label: 'Claude Code',
    hint: 'Merge into ~/.claude/settings.json (or settings.local.json). Claude Code sends ANTHROPIC_AUTH_TOKEN as a Bearer header, matching how ForgeRouter reads agent keys.',
    filename: 'settings.json',
    baseUrlKind: 'root',
    embedsSecret: true,
    block: (baseUrl, keyToken) => JSON.stringify({ env: { ANTHROPIC_BASE_URL: baseUrl, ANTHROPIC_AUTH_TOKEN: keyToken } }, null, 2),
  },
  {
    id: 'codex',
    label: 'Codex CLI',
    hint: 'Append to ~/.codex/config.toml. Requires wire_api = "responses" — Codex dropped its Chat Completions fallback in v0.138. The key never enters this file; Codex reads it from the env var below via env_key.',
    filename: 'config.toml',
    baseUrlKind: 'v1',
    embedsSecret: false,
    block: (baseUrl) => `[model_providers.forgerouter]\nname = "ForgeRouter"\nbase_url = "${baseUrl}"\nwire_api = "responses"\nenv_key = "FORGEROUTER_API_KEY"\n\nmodel_provider = "forgerouter"\nmodel = "forgerouter/auto"`,
    secret: { label: 'env var (shell profile or systemd EnvironmentFile)', line: (key) => `FORGEROUTER_API_KEY=${key}` },
  },
  {
    id: 'hermes-agent',
    label: 'Hermes Agent',
    hint: 'Matches the profile contract in /root/.hermes/profiles/<name>/config.yaml. Context length is discovered from ForgeRouter /v1/models so it follows the safe window advertised for forgerouter/auto.',
    filename: 'config.yaml',
    baseUrlKind: 'v1',
    embedsSecret: true,
    block: hermesAgentConfig,
  },
  {
    id: 'antigravity-cli',
    label: 'Antigravity CLI',
    hint: 'Merge into ~/.gemini/antigravity-cli/settings.json or export in your shell profile. Antigravity CLI / AGY connects to OpenAI/Anthropic-compatible endpoints via standard base URL and bearer token.',
    filename: 'settings.json',
    baseUrlKind: 'v1',
    embedsSecret: true,
    block: (baseUrl, keyToken) => JSON.stringify({
      env: {
        OPENAI_BASE_URL: baseUrl,
        OPENAI_API_KEY: keyToken,
      },
      model: "forgerouter/auto"
    }, null, 2),
    secret: { label: 'env vars (export in ~/.bashrc)', line: (key) => `export OPENAI_BASE_URL="http://localhost:2100/v1"\nexport OPENAI_API_KEY="${key}"` },
  },
];

/** Trusted SSO handoff from ForgeHub: the embedding page passes a freshly
 * minted session token in the URL fragment (#sso=…) — fragments never reach
 * server logs. Consumed once and scrubbed from the address bar; the token
 * is then validated by the normal /auth/me boot check like any session. */
function consumeSsoToken(): string | null {
  const match = window.location.hash.match(/[#&]sso=([^&]+)/);
  if (!match) return null;
  history.replaceState(null, '', window.location.pathname + window.location.search);
  return decodeURIComponent(match[1]);
}

function App() {
  const [authToken, setAuthToken] = useState(() => {
    const sso = consumeSsoToken();
    if (sso) {
      localStorage.setItem('forgerouter_session', sso);
      return sso;
    }
    return localStorage.getItem('forgerouter_session') ?? '';
  });
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [loginPassword, setLoginPassword] = useState('');
  const [page, setPage] = useState<Page>('agents');
  const [providers, setProviders] = useState<ProviderHealth[]>([]);
  const [watchdog, setWatchdog] = useState<WatchdogStatus | null>(null);
  const [routes, setRoutes] = useState<RouteEvent[]>([]);
  const [readiness, setReadiness] = useState<ProviderReady[]>([]);
  const [registry, setRegistry] = useState<RegistryProvider[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [usageMetric, setUsageMetric] = useState<UsageMetric>('messages');
  const [yearlyUsage, setYearlyUsage] = useState<YearlyUsage | null>(null);
  const [yearlyDemandUsage, setYearlyDemandUsage] = useState<YearlyUsageByDemand | null>(null);
  const [archiving, setArchiving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<string | null>(null);
  const [agentFilter, setAgentFilter] = useState(localStorage.getItem('forgerouter_agent') ?? 'all');
  const [agentKeys, setAgentKeys] = useState<Record<string, string>>({});
  const [dataLoaded, setDataLoaded] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [capabilityFilter, setCapabilityFilter] = useState<string>('all');
  const [modelSearch, setModelSearch] = useState('');
  const [editing, setEditing] = useState<RegistryProvider | null>(null);
  const [editingOriginalName, setEditingOriginalName] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [discovering, setDiscovering] = useState(false);
  const [validating, setValidating] = useState<string | null>(null);
  const [subscriptionPlans, setSubscriptionPlans] = useState<SubscriptionPlan[]>([]);
  const [msgSearch, setMsgSearch] = useState('');
  const [msgStatus, setMsgStatus] = useState<'all' | 'success' | 'failed'>('all');
  const [expandedRoute, setExpandedRoute] = useState<number | null>(null);
  const [playModel, setPlayModel] = useState('auto');
  const [playMsgs, setPlayMsgs] = useState<PlayMessage[]>([]);
  const [playInput, setPlayInput] = useState('');
  const [playSending, setPlaySending] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [connectOpen, setConnectOpen] = useState(false);
  const [newAgentName, setNewAgentName] = useState('');
  const [newAgentDescription, setNewAgentDescription] = useState('');
  const [newAgentKind, setNewAgentKind] = useState<'agent' | 'service'>('agent');
  const [newAgentAvatar, setNewAgentAvatar] = useState<string | null>(null);
  const [togglingGroup, setTogglingGroup] = useState<string | null>(null);
  const [togglingKind, setTogglingKind] = useState<string | null>(null);
  const [promptModal, setPromptModal] = useState<{ title: string; value: string; resolve: (value: string | null) => void } | null>(null);
  // Promise-based drop-in for window.prompt() — same call signature (title,
  // default value, resolves to the trimmed-by-caller string or null on
  // cancel) so every existing call site only needs `await` added, but backed
  // by the app's own modal instead of the browser's native dialog.
  function showPrompt(title: string, defaultValue = ''): Promise<string | null> {
    return new Promise((resolve) => setPromptModal({ title, value: defaultValue, resolve }));
  }
  const [confirmModal, setConfirmModal] = useState<{ title: string; message: string; confirmLabel: string; danger: boolean; resolve: (value: boolean) => void } | null>(null);
  // Promise-based drop-in for window.confirm() — same as showPrompt above,
  // backed by the app's own modal instead of the browser's native dialog.
  // danger=true styles the confirm button red (Trash2 icon) for a
  // destructive action instead of the default RefreshCw/primary styling.
  function showConfirm(title: string, message: string, confirmLabel = 'Confirm', danger = false): Promise<boolean> {
    return new Promise((resolve) => setConfirmModal({ title, message, confirmLabel, danger, resolve }));
  }
  const [newAgentKey, setNewAgentKey] = useState(generateAgentKey);
  const [creatingAgent, setCreatingAgent] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [agentModelsDraft, setAgentModelsDraft] = useState<string[]>([]);
  const [agentModelSearch, setAgentModelSearch] = useState('');
  const [savingAgentModels, setSavingAgentModels] = useState(false);
  const [agentClientId, setAgentClientId] = useState<AgentClientId>('claude-code');
  const [taskMap, setTaskMap] = useState<{ task: string; hint: string; model: string }[]>(DEFAULT_TASK_MAP);
  const [pricingModels, setPricingModels] = useState<PriceModel[]>([]);
  const [pricingMeta, setPricingMeta] = useState<{ priced_count: number; total_count: number; last_synced: string | null }>({ priced_count: 0, total_count: 0, last_synced: null });
  const [pricingSyncing, setPricingSyncing] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [refreshingHealth, setRefreshingHealth] = useState(false);
  const [associatingHealthy, setAssociatingHealthy] = useState(false);
  const [pricingSearch, setPricingSearch] = useState('');
  const [demandData, setDemandData] = useState<DemandData | null>(null);
  const [demandDrafts, setDemandDrafts] = useState<Record<string, string[]>>({});
  // Chains being reordered right now: while any is in edit mode, the 5s auto-refresh
  // is paused — a reload would overwrite the operator's unsaved ordering.
  const [editingDemands, setEditingDemands] = useState<string[]>([]);
  const editingDemandsRef = useRef<string[]>([]);
  // Master switch for the 5s auto-refresh ("sincronismo") — persisted per browser.
  const [autoRefresh, setAutoRefresh] = useState(localStorage.getItem('forgerouter_autorefresh') !== 'off');
  const autoRefreshRef = useRef(autoRefresh);
  const [contextCompaction, setContextCompaction] = useState(true);
  const [savingCompaction, setSavingCompaction] = useState(false);
  const [contextTruncation, setContextTruncation] = useState(false);
  const [contextTruncationMaxTokens, setContextTruncationMaxTokens] = useState(32000);
  const [contextTruncationTriggerPercent, setContextTruncationTriggerPercent] = useState(80);
  const [savingTruncation, setSavingTruncation] = useState(false);

  function markDemandEditing(demand: string) {
    if (!editingDemandsRef.current.includes(demand)) {
      editingDemandsRef.current = [...editingDemandsRef.current, demand];
      setEditingDemands(editingDemandsRef.current);
    }
  }

  function clearDemandEditing(demand: string) {
    editingDemandsRef.current = editingDemandsRef.current.filter((item) => item !== demand);
    setEditingDemands(editingDemandsRef.current);
  }

  function toggleAutoRefresh() {
    const next = !autoRefreshRef.current;
    autoRefreshRef.current = next;
    localStorage.setItem('forgerouter_autorefresh', next ? 'on' : 'off');
    setAutoRefresh(next);
    if (next) void loadAll();
  }

  async function toggleContextCompaction() {
    const next = !contextCompaction;
    setSavingCompaction(true);
    try {
      await fetchJson('/admin/settings/context-compaction', { method: 'POST', body: JSON.stringify({ enabled: next }) });
      setContextCompaction(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update context compaction');
    } finally {
      setSavingCompaction(false);
    }
  }

  async function toggleContextTruncation() {
    const next = !contextTruncation;
    setSavingTruncation(true);
    try {
      await fetchJson('/admin/settings/context-truncation', { method: 'POST', body: JSON.stringify({ enabled: next }) });
      setContextTruncation(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update context truncation');
    } finally {
      setSavingTruncation(false);
    }
  }

  async function editContextTruncationTriggerPercent() {
    const input = await showPrompt('Trigger truncation once the prompt fills this % of the selected model\'s real context window (10-100, matches the 80% hermes-agent\'s own compression already uses)', String(contextTruncationTriggerPercent));
    if (input === null) return;
    const value = Number(input.trim());
    if (!Number.isFinite(value) || value < 10 || value > 100) {
      setError('trigger_percent must be a number between 10 and 100');
      return;
    }
    setSavingTruncation(true);
    try {
      await fetchJson('/admin/settings/context-truncation', { method: 'POST', body: JSON.stringify({ enabled: contextTruncation, trigger_percent: value }) });
      setContextTruncationTriggerPercent(value);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update the trigger percent');
    } finally {
      setSavingTruncation(false);
    }
  }

  async function editContextTruncationFallbackLimit() {
    const input = await showPrompt('Fallback token budget — only used when the selected model\'s real context window isn\'t in the pricing catalog', String(contextTruncationMaxTokens));
    if (input === null) return;
    const value = Number(input.trim());
    if (!Number.isFinite(value) || value < 1000) {
      setError('max_tokens must be a number >= 1000');
      return;
    }
    setSavingTruncation(true);
    try {
      await fetchJson('/admin/settings/context-truncation', { method: 'POST', body: JSON.stringify({ enabled: contextTruncation, max_tokens: value }) });
      setContextTruncationMaxTokens(value);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update the fallback limit');
    } finally {
      setSavingTruncation(false);
    }
  }
  const [demandSearch, setDemandSearch] = useState<Record<string, string>>({});
  const [savingDemand, setSavingDemand] = useState<string | null>(null);
  const [playModelSearch, setPlayModelSearch] = useState('');
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [plusMenuOpen, setPlusMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(localStorage.getItem('proxyrouter_sidebar') === 'collapsed');
  const editorRef = useRef<HTMLElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const pickerRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const plusRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  function toggleSidebar() {
    setSidebarCollapsed((prev) => {
      localStorage.setItem('proxyrouter_sidebar', prev ? 'open' : 'collapsed');
      return !prev;
    });
  }

  // Status/error banners are contextual to the screen where the action ran —
  // navigating away dismisses them instead of leaking onto every other page.
  function navigateTo(target: Page) {
    if (target === page) return;
    setScanStatus(null);
    setError(null);
    setPage(target);
  }

  function openEditor(provider: RegistryProvider, originalName: string | null) {
    setPage('routing');
    setEditing({ ...deepClone(provider), api_key: '' });
    setEditingOriginalName(originalName);
  }

  useEffect(() => {
    if (editing) editorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [editing !== null, editingOriginalName]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [playMsgs, playSending]);

  useEffect(() => {
    if (!authToken) { setAuthChecked(true); return; }
    fetch('/auth/me', { headers: { Authorization: `Bearer ${authToken}` } })
      .then(async (res) => {
        if (!res.ok) throw new Error('invalid session');
        setAuthUser(await res.json());
      })
      .catch(() => {
        localStorage.removeItem('forgerouter_session');
        setAuthToken('');
      })
      .finally(() => setAuthChecked(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [themePref, setThemePref] = useState<ThemePref>(() => {
    const stored = localStorage.getItem('forgerouter_theme');
    return stored === 'light' || stored === 'system' ? stored : 'dark';
  });
  useEffect(() => {
    localStorage.setItem('forgerouter_theme', themePref);
    applyTheme(themePref);
    if (themePref !== 'system') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyTheme('system');
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [themePref]);

  // Permission gate for nav/pages: admins see everything; non-admins see
  // what their profile grants (no profile = nothing). Mirrors ForgeHub's
  // PermissionGate/usePermission pairing.
  const canView = (id: Page) => Boolean(authUser?.is_admin || authUser?.permissions?.[id]?.can_view);

  // Never leave the user parked on a page their profile hides (e.g. default
  // 'agents' landing for a monitoring-only profile).
  useEffect(() => {
    if (!authUser || authUser.must_change_password || canView(page)) return;
    const firstVisible = PAGES.find((item) => canView(item.id));
    if (firstVisible) setPage(firstVisible.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authUser, page]);

  function handleLogin(sessionToken: string, user: AuthUser, password: string) {
    localStorage.setItem('forgerouter_session', sessionToken);
    setAuthToken(sessionToken);
    setAuthUser(user);
    setLoginPassword(password);
  }

  async function logout() {
    try {
      await fetch('/auth/logout', { method: 'POST', headers: { Authorization: `Bearer ${authToken}` } });
    } catch {
      // session cleanup is best-effort
    }
    localStorage.removeItem('forgerouter_session');
    setAuthToken('');
    setAuthUser(null);
  }

  // Seed the model-controls draft only when picking an agent — re-seeding on every
  // 5s data refresh would wipe an edit in progress (same bug as the chain editor).
  const agentsRef = useRef<AgentInfo[]>([]);
  useEffect(() => { agentsRef.current = agents; }, [agents]);
  useEffect(() => {
    const agent = agentsRef.current.find((item) => item.name === selectedAgent);
    setAgentModelsDraft(agent?.models ?? []);
    setAgentModelSearch('');
    setAgentClientId('claude-code');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAgent]);

  useEffect(() => {
    if (!modelPickerOpen && !plusMenuOpen) return;
    const onDocClick = (event: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) setModelPickerOpen(false);
      if (plusRef.current && !plusRef.current.contains(event.target as Node)) setPlusMenuOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [modelPickerOpen, plusMenuOpen]);

  // All admin calls are authorized by the dashboard login session.
  const headers = useMemo(() => authToken ? { Authorization: `Bearer ${authToken}` } : undefined, [authToken]);

  async function fetchJson(path: string, init: RequestInit = {}) {
    const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}), ...(headers ?? {}) } });
    const data = await res.json();
    if (res.status === 401) {
      throw new Error('Session expired — sign out and sign in again.');
    }
    if (!res.ok) throw new Error(data?.error?.message ?? `Failed to load ${path}`);
    return data;
  }

  async function runScan() {
    if (scanning) return;
    // Full sync: re-discover, catalog and health-scan the free models of every provider.
    setScanning(true);
    try {
      setScanStatus('full sync running… (may take a few minutes)');
      const data = await fetchJson('/admin/providers/resync', { method: 'POST' });
      const report = data.providers as { provider: string; total?: number; healthy?: number; error?: string; skipped?: string }[];
      const scanned = report.filter((p) => !p.error && !p.skipped);
      const healthyProviders = scanned.filter((p) => (p.healthy ?? 0) > 0).length;
      const healthyModels = scanned.reduce((sum, p) => sum + (p.healthy ?? 0), 0);
      const totalModels = scanned.reduce((sum, p) => sum + (p.total ?? 0), 0);
      const parts = report
        .map((p) => p.error ? `${p.provider}: ${p.error}` : p.skipped ? `${p.provider}: ${p.skipped}` : `${p.provider}: ${p.healthy}/${p.total}`);
      setScanStatus(`Run scan: ${healthyProviders}/${report.length} providers · ${healthyModels}/${totalModels} models healthy — ${parts.join(' · ')}`);
      await loadAll();
    } catch (err) {
      setScanStatus('failed');
      setError(err instanceof Error ? err.message : 'Unknown scan error');
    } finally {
      setScanning(false);
    }
  }

  async function refreshHealth() {
    if (refreshingHealth) return;
    // Light validation: re-check status and latency of the registered (enabled) models.
    setRefreshingHealth(true);
    try {
      setScanStatus('validating status & latency…');
      const data = await fetchJson('/admin/providers/rescan', { method: 'POST' });
      // Provider-level rollup: a provider is healthy when at least one model responds.
      const results = (data.results ?? []) as { model_id: string; status: string }[];
      const byProvider: Record<string, { healthy: number; total: number }> = {};
      for (const result of results) {
        const provider = providerByModel[result.model_id] ?? result.model_id.split('/')[0];
        const entry = (byProvider[provider] ??= { healthy: 0, total: 0 });
        entry.total += 1;
        if (result.status === 'healthy') entry.healthy += 1;
      }
      const providerNames = Object.keys(byProvider);
      const healthyProviders = providerNames.filter((name) => byProvider[name].healthy > 0).length;
      const parts = providerNames.map((name) => `${name}: ${byProvider[name].healthy}/${byProvider[name].total}`);
      setScanStatus(`Refresh: ${healthyProviders}/${providerNames.length} providers · ${data.summary?.healthy ?? 0}/${data.summary?.total ?? 0} models healthy — ${parts.join(' · ')}`);
      await loadAll();
    } catch (err) {
      setScanStatus('failed');
      setError(err instanceof Error ? err.message : 'Unknown refresh error');
    } finally {
      setRefreshingHealth(false);
    }
  }

  async function associateAllHealthy() {
    if (associatingHealthy) return;
    setAssociatingHealthy(true);
    try {
      setScanStatus('associating all healthy & online models to all agents…');
      const data = await fetchJson('/admin/agents/associate-all-healthy', { method: 'POST' });
      setScanStatus(data.message || 'Associated all healthy models to all agents');
      await loadAll();
    } catch (err) {
      setScanStatus('failed');
      setError(err instanceof Error ? err.message : 'Failed to associate healthy models');
    } finally {
      setAssociatingHealthy(false);
    }
  }

  async function archiveOldMessages() {
    // Rolls up route_events older than the previous month into ai_router.usage_monthly,
    // then deletes them — the monthly table below keeps showing the full year.
    setArchiving(true);
    try {
      const result = await fetchJson('/admin/usage/archive', { method: 'POST' });
      setScanStatus(`Archived ${result.archived_months} agent-months · removed ${result.deleted_rows} old messages (older than ${formatDate(result.cutoff)})`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown archive error');
    } finally {
      setArchiving(false);
    }
  }

  async function syncPricing() {
    // Refreshes config/model_pricing.json from LiteLLM's public catalog, reads
    // live pricing straight from every registered provider's /models response
    // (OpenRouter/Kilo-style `pricing` object — the most authoritative source,
    // since it's the literal endpoint being routed through), then backfills
    // reference_cost on existing route_events that predate a model being
    // priced — so Overview/Messages reflect the new coverage right away.
    setPricingSyncing(true);
    try {
      const result = await fetchJson('/admin/pricing/sync', { method: 'POST' });
      setScanStatus(`Pricing sync: ${result.catalog_entries} catalog entries · ${result.live_entries} live provider prices · backfilled ${result.backfill_priced}/${result.backfill_checked} historical messages`);
      await loadAll();
    } catch (err) {
      setScanStatus('failed');
      setError(err instanceof Error ? err.message : 'Unknown pricing sync error');
    } finally {
      setPricingSyncing(false);
    }
  }

  async function loadAll() {
    try {
      const agentParam = agentFilter !== 'all' ? `&agent=${encodeURIComponent(agentFilter)}` : '';
      const [healthData, routeData, readyData, registryData, usageData, agentsData] = await Promise.all([
        fetchJson('/admin/providers/health'),
        fetchJson(`/admin/routes/recent?limit=100${agentParam}`),
        fetchJson('/admin/providers/readiness'),
        fetchJson('/admin/providers/registry'),
        fetchJson(`/admin/usage?days=30${agentParam}`),
        fetchJson('/admin/agents?days=30'),
      ]);
      try {
        const demand = await fetchJson('/admin/demand-routes');
        setDemandData(demand);
        setDemandDrafts((prev) => {
          // Never overwrite a chain that is being edited.
          const next = { ...(demand.routes ?? {}) } as Record<string, string[]>;
          for (const name of editingDemandsRef.current) if (prev[name] !== undefined) next[name] = prev[name];
          return next;
        });
      } catch {
        // demand routes are optional UI data; the rest of the dashboard still loads
      }
      try {
        const catalog = await fetchJson('/admin/subscriptions/catalog');
        setSubscriptionPlans(catalog.catalog ?? []);
      } catch {
        // subscription catalog is optional UI data
      }
      try {
        const tasks = await fetchJson('/admin/task-map');
        if (tasks.tasks?.length) setTaskMap(tasks.tasks);
      } catch {
        // task map falls back to the built-in defaults
      }
      try {
        const compaction = await fetchJson('/admin/settings/context-compaction');
        setContextCompaction(compaction.enabled ?? true);
      } catch {
        // context compaction setting defaults to enabled
      }
      try {
        const truncation = await fetchJson('/admin/settings/context-truncation');
        setContextTruncation(truncation.enabled ?? false);
        setContextTruncationMaxTokens(truncation.max_tokens ?? 32000);
        setContextTruncationTriggerPercent(truncation.trigger_percent ?? 80);
      } catch {
        // context truncation setting defaults to disabled
      }
      try {
        const yearly = await fetchJson('/admin/usage/yearly');
        setYearlyUsage(yearly);
      } catch {
        // yearly usage is optional UI data
      }
      try {
        const yearlyDemand = await fetchJson('/admin/usage/yearly-by-demand');
        setYearlyDemandUsage(yearlyDemand);
      } catch {
        // yearly demand usage is optional UI data
      }
      try {
        const pricing = await fetchJson('/admin/pricing/models');
        setPricingModels(pricing.models ?? []);
        setPricingMeta({ priced_count: pricing.priced_count ?? 0, total_count: pricing.total_count ?? 0, last_synced: pricing.last_synced ?? null });
      } catch {
        // pricing catalog coverage is optional UI data
      }
      try {
        const watchdogHealth = await fetchJson('/health');
        setWatchdog(watchdogHealth.providers ?? null);
      } catch {
        // watchdog status is optional UI data — its absence shouldn't hide the rest of the dashboard
      }
      setProviders(healthData.providers ?? []);
      setRoutes(routeData.routes ?? []);
      setReadiness(readyData.providers ?? []);
      setRegistry(registryData.providers ?? []);
      setUsage(usageData ?? null);
      setAgents(agentsData.agents ?? []);
      setDataLoaded(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }

  async function saveProvider() {
    if (!editing) return;
    const cleaned = {
      ...editing,
      name: editing.name.trim(),
      base_url: editing.base_url.trim(),
      api_key_env: editing.api_key_env.trim(),
      api_key: (editing.api_key ?? '').trim(),
      models: editing.models
        .filter((model) => model.id.trim())
        .map((model) => {
          const status = model.health ?? healthByModel[model.id.trim()];
          return {
            id: model.id.trim(),
            provider_model: model.provider_model.trim() || model.id.trim(),
            capabilities: model.capabilities.length ? model.capabilities : ['text'],
            enabled: model.enabled,
            health: model.health_detail ?? (status && status !== 'unknown' ? { status } : null),
          };
        }),
    };
    if (!cleaned.name || !cleaned.base_url) { setError('Provider name and base URL are required.'); return; }
    if (!cleaned.models.length) { setError('Add at least one model (model id is required).'); return; }
    const unscanned = cleaned.models.filter((model) => !model.health);
    if (unscanned.length && cleaned.access_type !== 'local') {
      setError(`Click "Detect models" before saving — ${unscanned.length} model(s) have no catalog/health status yet (${unscanned.slice(0, 3).map((model) => model.id).join(', ')}${unscanned.length > 3 ? '…' : ''}).`);
      return;
    }
    try {
      setSaving(true);
      if (editingOriginalName && editingOriginalName !== cleaned.name) {
        await fetchJson(`/admin/providers/${encodeURIComponent(editingOriginalName)}`, { method: 'DELETE' });
      }
      await fetchJson(`/admin/providers/${encodeURIComponent(cleaned.name)}`, { method: 'PUT', body: JSON.stringify(cleaned) });
      // The provider is saved under the selected agent (AGENTE_API_KEY): its enabled
      // models join the agent's list, healthy or not — routing only ever uses the
      // healthy ones. The server-side sync triggered by the save handles the rest:
      // models switched off here leave every agent's list, and per-agent opt-outs
      // (Model controls) are preserved.
      const agent = agents.find((item) => item.name === agentFilter);
      if (agent) {
        const optedOut = new Set(agent.models_off ?? []);
        const enabledIds = cleaned.models
          .filter((model) => model.enabled && !optedOut.has(model.id))
          .map((model) => model.id);
        const merged = Array.from(new Set([...agent.models, ...enabledIds]));
        await fetchJson(`/admin/agents/${encodeURIComponent(agent.name)}/models`, { method: 'PUT', body: JSON.stringify({ models: merged }) });
      }
      setEditing(null);
      setEditingOriginalName(null);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save provider');
    } finally {
      setSaving(false);
    }
  }

  async function validateProvider(name: string) {
    // Validate configuration: credential check + real chat completion per enabled model.
    try {
      setValidating(name);
      setError(null);
      setScanStatus(`validating ${name}…`);
      const data = await fetchJson(`/admin/providers/${encodeURIComponent(name)}/validate`, { method: 'POST' });
      if (data.message) {
        setScanStatus(`${name}: ${data.message}`);
      } else {
        setScanStatus(`${name}: ${data.summary?.healthy ?? 0}/${data.summary?.total ?? 0} models healthy`);
      }
      await loadAll();
    } catch (err) {
      setScanStatus(`${name}: validation failed`);
      setError(err instanceof Error ? err.message : 'Validation failed');
    } finally {
      setValidating(null);
    }
  }

  // On/off switch for the whole provider: disabled providers leave routing entirely
  // (their models stop being candidates) but keep their configuration and API key.
  async function toggleProviderEnabled(provider: RegistryProvider) {
    try {
      await fetchJson(`/admin/providers/${encodeURIComponent(provider.name)}`, {
        method: 'PUT',
        body: JSON.stringify({ ...provider, api_key: '', enabled: !provider.enabled }),
      });
      setScanStatus(`${provider.name}: ${provider.enabled ? 'disabled — its models left routing' : 'enabled — its models are back in routing'}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle the provider');
    }
  }

  async function removeProvider(name: string) {
    if (!window.confirm(`Delete provider "${name}" and all of its models?`)) return;
    try {
      await fetchJson(`/admin/providers/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete provider');
    }
  }

  async function discoverModels() {
    if (!editing) return;
    if (!editing.base_url.trim() && !editingOriginalName) { setError('Fill in the base URL before detecting models.'); return; }
    const subscriptionPlan = selectedSubscriptionPlan(subscriptionPlans, editing.name);
    const authHint = subscriptionPlan?.auth_method === 'oauth'
      ? ` ${subscriptionPlan.display_name} uses subscription authentication. ${subscriptionPlan.token_hint}`
      : '';
    try {
      setDiscovering(true);
      setError(null);
      const data = await fetchJson('/admin/providers/discover-models', {
        method: 'POST',
        body: JSON.stringify({
          provider_name: editingOriginalName ?? '',
          base_url: editing.base_url.trim(),
          api_key: (editing.api_key ?? '').trim(),
          api_key_env: editing.api_key_env.trim(),
        }),
      });
      const prefix = editing.name.trim() ? `${editing.name.trim()}/` : '';
      const found = data.models as DiscoveredModel[];
      if (!found.length) {
        setScanStatus(`No models detected.${authHint}`);
        return;
      }
      const foundById = new Map(found.map((model) => [model.id, model]));
      // Refresh the catalog (capabilities, rank, health) of models we already have,
      // keeping the operator's id and on/off choice; then append the newly found ones.
      const kept: RegistryModel[] = editing.models
        .filter((model) => model.id.trim())
        .map((model) => {
          const bareId = model.id.startsWith(prefix) ? model.id.slice(prefix.length) : model.id;
          const det = foundById.get(model.provider_model) ?? foundById.get(bareId) ?? foundById.get(model.id);
          if (!det) return model;
          return {
            ...model,
            capabilities: det.capabilities?.length ? det.capabilities : model.capabilities,
            score: det.score ?? model.score,
            health: det.health?.status ?? model.health,
            health_detail: det.health ?? model.health_detail,
          };
        });
      const existing = new Set(kept.flatMap((model) => [model.id, model.provider_model]));
      const discovered: RegistryModel[] = found
        .filter((model) => !existing.has(prefix + model.id) && !existing.has(model.id))
        .map((model) => ({
          id: prefix + model.id,
          provider_model: model.id,
          capabilities: model.capabilities?.length ? model.capabilities : ['text'],
          enabled: model.health ? model.health.status === 'healthy' : true,
          score: model.score,
          health: model.health?.status,
          health_detail: model.health ?? undefined,
        }));
      setEditing({ ...editing, models: [...kept, ...discovered] });
      const parts = [];
      if (data.scanned) parts.push(`${data.healthy}/${data.total} healthy`);
      if (data.excluded_paid > 0) parts.push(`${data.excluded_paid} paid skipped`);
      setScanStatus(parts.length ? parts.join(' · ') : `${found.length} models detected`);
    } catch (err) {
      setError(`${err instanceof Error ? err.message : 'Model discovery failed'}${authHint}`);
    } finally {
      setDiscovering(false);
    }
  }

  async function createAgent() {
    const name = newAgentName.trim();
    if (!name || creatingAgent) return;
    try {
      setCreatingAgent(true);
      setError(null);
      await fetchJson('/admin/agents', { method: 'POST', body: JSON.stringify({ name, api_key: newAgentKey, description: newAgentDescription.trim(), kind: newAgentKind, avatar_data_url: newAgentAvatar }) });
      setScanStatus(`agent "${name}" created — the key is stored in the agent's registration; paste it into the agent's connection settings`);
      setNewAgentName('');
      setNewAgentDescription('');
      setNewAgentKind('agent');
      setNewAgentAvatar(null);
      setNewAgentKey(generateAgentKey());
      setConnectOpen(false);
      setSelectedAgent(null);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create agent');
    } finally {
      setCreatingAgent(false);
    }
  }

  function openEditAgent(name: string) {
    setConnectOpen(false);
    setSelectedAgent(name);
    const agent = agentsRef.current.find((item) => item.name === name);
    setAgentModelsDraft(agent?.models ?? []);
    setAgentModelSearch('');
  }

  async function rotateAgentKey(name: string) {
    if (!(await showConfirm('Rotate API key', `Generate a new API key for "${name}"? The current key stops working immediately, but the agent keeps all of its provider/model controls.`, 'Rotate key'))) return;
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/rotate-key`, { method: 'POST' });
      const deploy = data.deploy as { applied?: boolean; status?: string; detail?: string } | undefined;
      if (deploy && deploy.applied && (deploy.status === 'applied' || deploy.status === 'written')) {
        // Config file + service already updated by the backend — nothing left for the operator to paste.
        setScanStatus(`${name}: key rotated — ${deploy.detail}`);
      } else if (deploy && deploy.status !== 'no_config') {
        // A deploy-config exists but applying it failed — the DB key IS new; the file is NOT.
        // This must read as an error, not a quiet status line, or it's exactly the silent-stale-key bug again.
        setError(`${name}: key rotated in the database, but the config file was NOT updated — ${deploy.detail}`);
      } else {
        setScanStatus((await copyText(data.api_key)) ? `${name}: new API key copied — update the agent's connection settings` : `${name}: new key is ${data.api_key}`);
      }
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rotate the agent key');
    }
  }

  async function saveAgentDeployConfig(name: string, config: { config_path: string; config_format: string; config_key: string; restart_service: string }) {
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/deploy-config`, { method: 'PUT', body: JSON.stringify(config) });
      setScanStatus(`${name}: deploy-config saved`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save deploy-config');
    }
  }

  async function editAgentDeployConfig(name: string, agent: AgentInfo) {
    const configPath = await showPrompt(
      `Config file path for "${name}" (as seen INSIDE the forgerouter container — e.g. /root/.hermes/profiles/${name.toLowerCase()}/config.yaml or /root/.claude/.env). Leave blank to clear — rotate-key then stays DB-only for this agent.`,
      agent.config_path || '',
    );
    if (configPath === null) return;
    if (!configPath.trim()) {
      await saveAgentDeployConfig(name, { config_path: '', config_format: '', config_key: '', restart_service: '' });
      return;
    }
    const formatInput = await showPrompt(`Config format for "${name}" — type "yaml" (nested key in a YAML file) or "env" (KEY=value line)`, agent.config_format || 'env');
    if (formatInput === null) return;
    const configFormat = formatInput.trim().toLowerCase() === 'yaml' ? 'yaml' : 'env';
    const configKey = await showPrompt(
      `Key name for "${name}" — documentation only (the write itself replaces the old key value verbatim): e.g. providers.forgerouter.api_key (yaml) or FORGEROUTER_API_KEY (env)`,
      agent.config_key || (configFormat === 'yaml' ? 'providers.forgerouter.api_key' : 'FORGEROUTER_API_KEY'),
    );
    if (configKey === null) return;
    const restartService = await showPrompt(
      `systemd service to restart after writing the new key for "${name}" (leave blank if nothing needs restarting — e.g. CLI-invoked agents like Aramis/Porthos/Dartan pick it up on their next run)`,
      agent.restart_service || `hermes-gateway-${name.toLowerCase()}.service`,
    );
    if (restartService === null) return;
    await saveAgentDeployConfig(name, {
      config_path: configPath.trim(),
      config_format: configFormat,
      config_key: configKey.trim(),
      restart_service: restartService.trim(),
    });
  }

  async function duplicateAgent(name: string) {
    const newName = await showPrompt(`Duplicate agent "${name}" as… (copies its provider/model controls; gets a new API key)`);
    if (!newName?.trim()) return;
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/duplicate`, { method: 'POST', body: JSON.stringify({ name: newName.trim() }) });
      setScanStatus((await copyText(data.api_key)) ? `${data.agent}: created from ${name} — API key copied` : `${data.agent} created from ${name}`);
      setSelectedAgent(data.agent);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to duplicate agent');
    }
  }

  async function saveAgentModels(name: string, returnToList = true) {
    try {
      setSavingAgentModels(true);
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/models`, { method: 'PUT', body: JSON.stringify({ models: agentModelsDraft }) });
      setScanStatus(`${name}: model controls saved${agentModelsDraft.length ? ` (${agentModelsDraft.length} models)` : ' (no models — the agent cannot route until providers are associated)'}`);
      if (returnToList) setSelectedAgent(null);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save agent models');
    } finally {
      setSavingAgentModels(false);
    }
  }

  async function renameAgent(name: string) {
    const newName = await showPrompt(`Rename agent "${name}" to…`, name);
    if (!newName?.trim() || newName.trim() === name) return;
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/name`, { method: 'PUT', body: JSON.stringify({ name: newName.trim() }) });
      setScanStatus(`${name} renamed to ${data.agent}`);
      setSelectedAgent(data.agent);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename agent');
    }
  }

  async function toggleAgentKind(name: string, current: 'agent' | 'service' | undefined) {
    const next = current === 'service' ? 'agent' : 'service';
    setTogglingKind(name);
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/kind`, { method: 'PUT', body: JSON.stringify({ kind: next }) });
      setScanStatus(`${name}: marked as ${next}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the agent kind');
    } finally {
      setTogglingKind((current) => (current === name ? null : current));
    }
  }

  async function editAgentDescription(name: string, current: string) {
    const description = await showPrompt(`Description for "${name}" — e.g. used by internal automation tasks`, current);
    if (description === null) return;
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/description`, { method: 'PUT', body: JSON.stringify({ description: description.trim() }) });
      setScanStatus(`${name}: description saved`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the agent description');
    }
  }

  async function saveAgentAvatar(name: string, avatarDataUrl: string | null) {
    try {
      setError(null);
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/avatar`, {
        method: 'PUT',
        body: JSON.stringify({ avatar_data_url: avatarDataUrl ?? '' }),
      });
      setScanStatus(`${name}: profile image ${avatarDataUrl ? 'saved' : 'removed'}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the agent profile image');
    }
  }

  async function editAgentBudget(name: string, currentLimit: number | null, currentAction: string) {
    const limitInput = await showPrompt(`Monthly reference-cost budget for "${name}" (USD, e.g. 10.00 — leave blank for no limit)`, currentLimit != null ? String(currentLimit) : '');
    if (limitInput === null) return;
    const trimmed = limitInput.trim();
    const limitUsd = trimmed ? Number(trimmed) : null;
    if (trimmed && (Number.isNaN(limitUsd as number) || (limitUsd as number) < 0)) {
      setError('Budget limit must be a non-negative number');
      return;
    }
    let action = currentAction;
    if (limitUsd !== null) {
      const actionInput = await showPrompt(`Action once "${name}" reaches its budget — type "alert" (dashboard-only, default) or "block" (also rejects new requests with 429 until next month)`, currentAction);
      if (actionInput === null) return;
      action = actionInput.trim().toLowerCase() === 'block' ? 'block' : 'alert';
    }
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}/budget`, { method: 'PUT', body: JSON.stringify({ limit_usd: limitUsd, action }) });
      setScanStatus(limitUsd != null ? `${name}: budget set to $${limitUsd.toFixed(2)}/mo (${action})` : `${name}: budget limit removed`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the agent budget');
    }
  }

  async function copyAgentKey(name: string) {
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/key`);
      setScanStatus((await copyText(data.api_key)) ? `${name} API key copied` : 'copy failed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch the agent API key');
    }
  }

  function agentClientBaseUrl(client: AgentClientDef): string {
    return client.baseUrlKind === 'root' ? window.location.origin : `${window.location.origin}/v1`;
  }

  async function copyAgentClientBlock(name: string, client: AgentClientDef) {
    const baseUrl = agentClientBaseUrl(client);
    if (!client.embedsSecret) {
      setScanStatus((await copyText(client.block(baseUrl, ''))) ? `${client.label} config copied` : 'copy failed');
      return;
    }
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/key`);
      setScanStatus((await copyText(client.block(baseUrl, data.api_key))) ? `${client.label} config copied (with the real key)` : 'copy failed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch the agent API key');
    }
  }

  async function copyAgentClientSecret(name: string, client: AgentClientDef) {
    if (!client.secret) return;
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/key`);
      setScanStatus((await copyText(client.secret.line(data.api_key))) ? `${client.label} ${client.secret.label} copied` : 'copy failed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch the agent API key');
    }
  }

  async function removeAgent(name: string) {
    if (!(await showConfirm('Delete agent', `Delete agent "${name}"? Its API key stops working immediately.`, 'Delete', true))) return;
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (selectedAgent === name) setSelectedAgent(null);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete agent');
    }
  }

  async function saveTaskModel(task: string, model: string) {
    const previous = taskMap;
    setTaskMap((rows) => rows.map((row) => row.task === task ? { ...row, model } : row));
    try {
      await fetchJson(`/admin/task-map/${encodeURIComponent(task)}`, { method: 'PUT', body: JSON.stringify({ model }) });
      setScanStatus(`${task} → ${model}`);
    } catch (err) {
      setTaskMap(previous);
      setError(err instanceof Error ? err.message : 'Failed to save task map');
    }
  }

  async function saveDemand(demand: string, models: string[]) {
    try {
      setSavingDemand(demand);
      await fetchJson(`/admin/demand-routes/${demand}`, { method: 'PUT', body: JSON.stringify({ models }) });
      setScanStatus(`${demand}: ${models.length ? `chain saved (${models.length} models)` : 'back to automatic rank-based routing'}`);
      clearDemandEditing(demand);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save demand chain');
    } finally {
      setSavingDemand(null);
    }
  }

  // Toggle the filtered agent's association with one model (Manage Models row) or a
  // provider's whole catalog (Manage providers row). set_agent_models keeps removed
  // associations as per-agent opt-outs, so they survive Refresh / Run scan syncs.
  async function setAgentModelAssociation(modelIds: string[], on: boolean) {
    const agent = agents.find((item) => item.name === agentFilter);
    if (!agent || !modelIds.length) return;
    const next = on
      ? Array.from(new Set([...agent.models, ...modelIds]))
      : agent.models.filter((id) => !modelIds.includes(id));
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(agent.name)}/models`, { method: 'PUT', body: JSON.stringify({ models: next }) });
      setScanStatus(`${agent.name}: ${modelIds.length === 1 ? modelIds[0] : `${modelIds.length} models`} ${on ? 'associated' : 'removed (kept as opt-out)'}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update the agent association');
    }
  }

  async function copyProviderKey(name: string) {
    try {
      const data = await fetchJson(`/admin/providers/${encodeURIComponent(name)}/key`);
      if (!data.api_key) { setScanStatus(`${name}: no API key configured`); return; }
      setScanStatus((await copyText(data.api_key)) ? `${name} API key copied` : 'copy failed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch the API key');
    }
  }

  function addAttachmentFiles(files: Iterable<File>) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        setScanStatus(`${file.name}: only image attachments are supported for now`);
        continue;
      }
      const reader = new FileReader();
      reader.onload = () => setAttachments((prev) => [...prev, { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, url: String(reader.result), name: file.name }]);
      reader.readAsDataURL(file);
    }
  }

  function onPickImage(event: React.ChangeEvent<HTMLInputElement>) {
    if (event.target.files?.length) addAttachmentFiles(Array.from(event.target.files));
    event.target.value = '';
  }

  // Auto-growing textarea: expands smoothly with the content up to ~8 lines.
  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [playInput, page]);

  async function agentKeyFor(name: string): Promise<string | null> {
    if (agentKeys[name]) return agentKeys[name];
    try {
      const data = await fetchJson(`/admin/agents/${encodeURIComponent(name)}/key`);
      setAgentKeys((keys) => ({ ...keys, [name]: data.api_key }));
      return data.api_key;
    } catch {
      return null;
    }
  }

  async function sendPlayground() {
    const content = playInput.trim();
    if ((!content && !attachments.length) || playSending) return;
    if (agentFilter === 'all') {
      setError('Pick an agent in the composer — Playground messages are always sent as a specific agent.');
      return;
    }
    const agentKey = await agentKeyFor(agentFilter);
    if (!agentKey) {
      setError(`Could not load the API key for agent "${agentFilter}".`);
      return;
    }
    const images = attachments.map((attachment) => attachment.url);
    const history: PlayMessage[] = [...playMsgs, { role: 'user', content, images: images.length ? images : undefined }];
    setPlayMsgs(history);
    setPlayInput('');
    setAttachments([]);
    setPlaySending(true);
    const started = Date.now();
    try {
      const apiMessages = history.map((message) => message.images?.length
        ? { role: message.role, content: [...(message.content ? [{ type: 'text', text: message.content }] : []), ...message.images.map((url) => ({ type: 'image_url', image_url: { url } }))] }
        : { role: message.role, content: message.content });
      // Chat as the selected agent: its key attributes usage and applies its model controls.
      const chatHeaders: Record<string, string> = { 'Content-Type': 'application/json', Authorization: `Bearer ${agentKey}` };
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: chatHeaders,
        body: JSON.stringify({ model: playModel, messages: apiMessages, max_tokens: 1024 }),
      });
      const data = await res.json();
      const elapsed = Date.now() - started;
      if (!res.ok) throw new Error(data?.error?.message ?? `http_${res.status}`);
      const reply = data.choices?.[0]?.message?.content || '(empty response)';
      const usedModel = res.headers.get('x-proxyrouter-model') ?? playModel;
      const meta = `${usedModel} · ${formatTokens(data.usage?.total_tokens ?? 0)} tokens · ${formatLatency(elapsed)}`;
      setPlayMsgs([...history, { role: 'assistant', content: reply, meta }]);
      void loadAll();
    } catch (err) {
      setPlayMsgs([...history, { role: 'assistant', content: err instanceof Error ? err.message : 'request failed', meta: 'error' }]);
    } finally {
      setPlaySending(false);
    }
  }

  function updateModel(index: number, patch: Partial<RegistryModel>) {
    if (!editing) return;
    setEditing({ ...editing, models: editing.models.map((model, i) => i === index ? { ...model, ...patch } : model) });
  }

  useEffect(() => {
    if (!authUser || authUser.must_change_password) return;
    void loadAll();

    const interval = setInterval(() => {
      // Paused by the sidebar switch, or while a chain is being reordered —
      // a reload would overwrite the operator's unsaved edits.
      if (!autoRefreshRef.current || editingDemandsRef.current.length) return;
      void loadAll();
    }, 5000); // auto-refresh every 5 seconds for real-time monitoring of route events and usage

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authUser, agentFilter]);

  function selectAgentFilter(value: string) {
    setAgentFilter(value);
    localStorage.setItem('forgerouter_agent', value);
  }

  // The other screens depend on agents: without one registered, stay on the Agents page.
  useEffect(() => {
    if (dataLoaded && !agents.length && page !== 'agents') setPage('agents');
  }, [dataLoaded, agents.length, page]);

  // Drop a stale selection (e.g. the agent was deleted).
  useEffect(() => {
    if (dataLoaded && agentFilter !== 'all' && !agents.some((agent) => agent.name === agentFilter)) selectAgentFilter('all');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataLoaded, agents]);

  // The Playground always chats as a specific agent — "all agents" is not a sender.
  // Routing also requires one: provider management acts under the selected agent's API key.
  useEffect(() => {
    if ((page === 'playground' || page === 'routing') && agentFilter === 'all' && agents.length) selectAgentFilter(agents[0].name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, agentFilter, agents]);

  const healthy = providers.filter((p) => p.status === 'healthy').length;
  const healthByModel = useMemo(() => Object.fromEntries(providers.map((p) => [p.model_id, p.status])), [providers]);
  // Three states, not two: "saudável" (enabled and responding), "desligado"
  // (turned off on purpose — manual_off or auto-disabled — never even routed
  // to, not an outage) and "off / não responde" (enabled but currently failing
  // health checks — the state that actually signals a real problem, e.g. no
  // internet). Folding the last two together (as a plain healthy/unhealthy
  // split would) makes an intentionally-off model look like an outage.
  const modelStatusCounts = useMemo(() => {
    let healthyCount = 0;
    let disabledCount = 0;
    let downCount = 0;
    for (const provider of registry) {
      for (const model of provider.models) {
        if (!model.enabled) { disabledCount += 1; continue; }
        if (healthByModel[model.id] === 'healthy') healthyCount += 1;
        else downCount += 1;
      }
    }
    return { healthy: healthyCount, disabled: disabledCount, down: downCount, total: healthyCount + disabledCount + downCount };
  }, [registry, healthByModel]);

  const keptModels = editing?.models.filter((model) => model.id.trim()) ?? [];
  const allModelsScanned = keptModels.length > 0 && keptModels.every((model) => {
    const status = model.health ?? healthByModel[model.id.trim()];
    return Boolean(status && status !== 'unknown');
  });
  // Local providers may save manually-added models before any scan — the
  // "Validate configuration" button checks and persists their health afterwards.
  const canSave = !saving && !discovering && (allModelsScanned || (editing?.access_type === 'local' && keptModels.length > 0));
  const capsByModel = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const provider of registry) for (const model of provider.models) map[model.id] = model.capabilities;
    return map;
  }, [registry]);
  const providerByModel = useMemo(() => {
    const map: Record<string, string> = {};
    for (const provider of registry) for (const model of provider.models) map[model.id] = provider.name;
    return map;
  }, [registry]);
  const readyByName = useMemo(() => Object.fromEntries(readiness.map((item) => [item.provider, item])), [readiness]);
  const scoreByModel = useMemo(() => {
    const map: Record<string, number> = {};
    for (const provider of registry) for (const model of provider.models) map[model.id] = model.score ?? 0;
    return map;
  }, [registry]);
  const contextByModel = useMemo(() => {
    const map: Record<string, number | null> = {};
    for (const provider of registry) for (const model of provider.models) map[model.id] = model.context_length ?? null;
    return map;
  }, [registry]);
  // All healthy AND enabled catalog model ids, over every provider — the universe
  // every per-badge selector below filters down from. Both checks matter: the
  // health endpoint keeps a model's last-known status even after it's been
  // manually disabled (manual_off), so a model turned off catalog-wide (e.g. the
  // openrouter/auto meta-routers) would otherwise still count as a toggle
  // candidate forever — permanently pinning every group it belongs to at
  // "not fully on" and turning every click on those groups into an "enable"
  // instead of ever reaching "disable".
  function healthyModelIds(): string[] {
    const ids: string[] = [];
    for (const provider of registry) for (const model of provider.models) if (model.enabled && healthByModel[model.id] === 'healthy') ids.push(model.id);
    return ids;
  }
  // Every healthy catalog model id that belongs to a task group (simple/standard/
  // complex by score band, reasoning/vision/audio/code by capability) — the same
  // classification the first agent-card badge row counts, just over the whole
  // catalog instead of only the agent's already-associated models.
  function modelsInGroup(group: string): string[] {
    return healthyModelIds().filter((id) => {
      const caps = capsByModel[id] ?? [];
      const score = scoreByModel[id] ?? 0;
      return (
        (group === 'vision' && caps.includes('vision')) ||
        (group === 'audio' && caps.includes('audio')) ||
        (group === 'code' && caps.includes('code')) ||
        (group === 'reasoning' && caps.includes('reasoning')) ||
        (group === 'complex' && score >= 50) ||
        (group === 'standard' && score >= 30 && score < 50) ||
        (group === 'simple' && score < 30)
      );
    });
  }
  function modelsInCostClass(costClass: CostClass): string[] {
    return modelIdsForCostClass(registry, costClass);
  }
  function allModelsInCostClass(costClass: CostClass): string[] {
    return allModelIdsForCostClass(registry, costClass);
  }
  async function toggleAgentModelSet(agent: AgentInfo, ids: string[], label: string, allIdsInClass?: string[]) {
    if ((!ids.length && !allIdsInClass?.length) || togglingGroup) return;
    const current = new Set(agent.models);
    const anyOn = ids.some((id) => current.has(id)) || (allIdsInClass?.some((id) => current.has(id)) ?? false);
    const next = nextAgentModelsForToggle(agent.models, ids, allIdsInClass);
    const key = `${agent.name}:${label}`;
    setTogglingGroup(key);
    try {
      await fetchJson(`/admin/agents/${encodeURIComponent(agent.name)}/models`, { method: 'PUT', body: JSON.stringify({ models: next }) });
      setScanStatus(`${agent.name}: ${label} ${anyOn ? 'disabled' : 'enabled'} (${anyOn ? (allIdsInClass?.filter((id) => current.has(id)).length ?? ids.length) : ids.length} models)`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to toggle ${label} for ${agent.name}`);
    } finally {
      setTogglingGroup((current) => (current === key ? null : current));
    }
  }
  // Models the selected agent may use; null only when "all agents" is selected.
  // An empty set = the agent has no providers associated yet (routes to nothing).
  const agentAllowed = useMemo(() => {
    if (agentFilter === 'all') return null;
    const agent = agents.find((item) => item.name === agentFilter);
    return agent ? new Set(agent.models) : null;
  }, [agentFilter, agents]);
  const agentByName = useMemo(() => new Map(agents.map((agent) => [agent.name, agent])), [agents]);
  // Manage providers always shows the full registry — hiding rows under an agent
  // filter made freshly saved providers look like the save had failed. The Models
  // column still reports how many models the filtered agent may use.
  const visibleRegistry = registry;
  // Provider-level health: a provider is working when at least one of its models is healthy.
  const healthByProvider = useMemo(() => {
    const map: Record<string, { healthy: number; total: number }> = {};
    for (const p of providers) {
      const entry = (map[p.provider] ??= { healthy: 0, total: 0 });
      entry.total += 1;
      if (p.status === 'healthy') entry.healthy += 1;
    }
    return map;
  }, [providers]);
  const providerTotals = useMemo(() => {
    const names = Object.keys(healthByProvider);
    return { healthy: names.filter((name) => healthByProvider[name].healthy > 0).length, total: names.length };
  }, [healthByProvider]);
  // Healthy models grouped by demand class — mirrors the per-agent breakdown on the Agents page.
  const taskGroups = useMemo(() => {
    const groups = { simple: 0, standard: 0, complex: 0, reasoning: 0, vision: 0, audio: 0, code: 0 };
    for (const p of providers) {
      if (p.status !== 'healthy') continue;
      const score = scoreByModel[p.model_id] ?? 0;
      const caps = capsByModel[p.model_id] ?? [];
      if (caps.includes('reasoning')) groups.reasoning += 1;
      if (caps.includes('vision')) groups.vision += 1;
      if (caps.includes('audio')) groups.audio += 1;
      if (caps.includes('code')) groups.code += 1;
      if (score >= 50) groups.complex += 1;
      else if (score >= 30) groups.standard += 1;
      else groups.simple += 1;
    }
    return groups;
  }, [providers, scoreByModel, capsByModel]);
  // Messages/tokens actually classified into each demand class, last `usage.days`
  // days — real ground truth from route_events.demand (set by app/demand.py at
  // request time), not inferred from which models happen to serve that capability.
  // A message counts toward exactly one group, its resolved demand.
  const taskGroupUsage = useMemo(() => {
    const groups = {
      simple: { messages: 0, tokens: 0 },
      standard: { messages: 0, tokens: 0 },
      complex: { messages: 0, tokens: 0 },
      reasoning: { messages: 0, tokens: 0 },
      vision: { messages: 0, tokens: 0 },
      audio: { messages: 0, tokens: 0 },
      code: { messages: 0, tokens: 0 },
    };
    for (const item of usage?.by_demand ?? []) {
      if (item.demand in groups) groups[item.demand as keyof typeof groups] = { messages: item.messages, tokens: item.tokens };
    }
    return groups;
  }, [usage]);
  // Manage Models lists the full catalog — every model is associated to every agent
  // by default; the Agent column shows (and toggles) the filtered agent's opt-out.
  const visibleProviders = providers
    .filter((p) => statusFilter === 'all' || (statusFilter === 'healthy' ? p.status === 'healthy' : p.status !== 'healthy'))
    .filter((p) => capabilityFilter === 'all' || (capsByModel[p.model_id] ?? []).includes(capabilityFilter))
    .filter((p) => !modelSearch.trim() || `${p.provider} ${p.model_id}`.toLowerCase().includes(modelSearch.trim().toLowerCase()))
    // Routing order: tier first (lower goes first), AI rank breaks ties (higher first).
    .sort((a, b) => a.tier - b.tier || (scoreByModel[b.model_id] ?? 0) - (scoreByModel[a.model_id] ?? 0) || a.model_id.localeCompare(b.model_id))
    .slice(0, 50);
  const visibleRoutes = routes
    .filter((route) => msgStatus === 'all' || (msgStatus === 'success' ? route.status === 'success' : route.status !== 'success'))
    .filter((route) => !msgSearch.trim() || `${route.model_id ?? ''} ${route.agent ?? ''} ${route.request_id}`.toLowerCase().includes(msgSearch.trim().toLowerCase()))
    .slice(0, 50);
  const visiblePricingModels = pricingModels.filter(
    (item) => !pricingSearch.trim() || item.public_id.toLowerCase().includes(pricingSearch.trim().toLowerCase())
  );
  const playModelOptions = providers
    .filter((p) => p.status === 'healthy')
    .filter((p) => !agentAllowed || agentAllowed.has(p.model_id))
    .sort((a, b) => (scoreByModel[b.model_id] ?? 0) - (scoreByModel[a.model_id] ?? 0));

  const agentSelect = (
    <AgentSelect
      agents={agents}
      value={agentFilter}
      onValueChange={selectAgentFilter}
      includeAll
      label="Filter by agent — applies to every screen"
    />
  );

  // Routing variant without "all agents": provider management acts under a
  // specific agent's API key (AGENTE_API_KEY), so one must always be selected.
  const agentSelectRequired = (
    <AgentSelect
      agents={agents}
      value={agentFilter === 'all' ? agents[0]?.name ?? '' : agentFilter}
      onValueChange={selectAgentFilter}
      label="Provider management acts under this agent's API key"
    />
  );

  const usagePanel = usage && (
    <section className="panel">
      <div className="usageTabs">
        {(['messages', 'tokens', 'cost', 'reference_cost'] as UsageMetric[]).map((metric) => (
          <button key={metric} className={`usageTab${usageMetric === metric ? ' active' : ''}`} onClick={() => setUsageMetric(metric)} title={metric === 'reference_cost' ? 'Notional cost at public commercial rates for an equivalent model — never billed, since only free-tier models are routed to' : undefined}>
            <span>{metric === 'messages' ? 'Messages' : metric === 'tokens' ? 'Token usage' : metric === 'cost' ? 'Cost' : 'Reference cost'}</span>
            <strong>{metric === 'messages' ? usage.totals.messages : metric === 'tokens' ? formatTokens(usage.totals.tokens) : metric === 'cost' ? formatCost(usage.totals.cost) : formatCost(usage.totals.reference_cost)}</strong>
          </button>
        ))}
        <span className="usageRange">last {usage.days} days</span>
      </div>
      <UsageChart
        seriesList={
          agentFilter === 'all' && usage.by_agent && usage.by_agent.length > 0
            ? usage.by_agent.map((a) => ({ name: a.agent, avatarUrl: agents.find((agent) => agent.name === a.agent)?.avatar_data_url, daily: a.daily }))
            : [{ name: agentFilter === 'all' ? 'Total' : agentFilter, avatarUrl: agents.find((agent) => agent.name === agentFilter)?.avatar_data_url, daily: usage.daily }]
        }
        metric={usageMetric}
        days={usage.days}
      />
    </section>
  );

  if (!authChecked) {
    return <div className="authShell"><Logo size={42} /></div>;
  }
  if (!authUser) {
    return <LoginScreen onLogin={handleLogin} />;
  }
  if (authUser.must_change_password) {
    return (
      <ChangeCredentialsScreen
        token={authToken}
        username={authUser.username}
        initialPassword={loginPassword}
        onDone={(name) => {
          // Same bug class as the login screen: the change-password response
          // only carries {username, must_change_password} — refetch the full
          // profile (is_admin, permissions, avatar, email) instead of
          // fabricating a stripped-down user object from just the username.
          fetchJson('/auth/me')
            .then((user) => setAuthUser(user as AuthUser))
            .catch(() => setAuthUser({ username: name, must_change_password: false }));
        }}
      />
    );
  }

  return (
    <div className="appShell">
      <aside className={`sidebar${sidebarCollapsed ? ' collapsed' : ''}`}>
        <div className="sbHeader">
          {sidebarCollapsed ? (
            // Icon-rail mode: the logo itself is the expand trigger — hover
            // swaps the mark for the expand icon, same affordance as the
            // dedicated collapse button in expanded mode below.
            <button type="button" className="logoExpandBtn" onClick={toggleSidebar} title="Expand sidebar" aria-label="Expand sidebar">
              <Logo size={30} />
              <PanelLeftOpen size={16} className="logoExpandIcon" />
              <span className="sidebarTooltip">Expand sidebar</span>
            </button>
          ) : (
            <>
              <span className="brandRow">
                <Logo size={28} />
                <span>
                  <p className="eyebrow">LLM Gateway</p>
                  <strong>ForgeRouter</strong>
                </span>
              </span>
              <button className="iconButton collapseBtn" title="Collapse sidebar" onClick={toggleSidebar}>
                <PanelLeftClose size={15} />
              </button>
            </>
          )}
        </div>
        <nav className="sbNav">
          {['Monitoring', 'Manage', 'Administration'].map((section) => {
            const visibleItems = PAGES.filter((item) => item.section === section && canView(item.id));
            if (!visibleItems.length) return null;
            return (
              <div key={section} className="navGroup">
                {!sidebarCollapsed && <p className="navSection">{section}</p>}
                {visibleItems.map((item) => {
                  const locked = section !== 'Administration' && item.id !== 'agents' && dataLoaded && !agents.length;
                  return (
                    <button
                      key={item.id}
                      className={`navItem${page === item.id ? ' active' : ''}`}
                      disabled={locked}
                      title={locked ? 'Register an agent first' : item.label}
                      onClick={() => navigateTo(item.id)}
                    >{item.icon}{!sidebarCollapsed && item.label}</button>
                  );
                })}
              </div>
            );
          })}
        </nav>
        <div className="sidebarFooter">
          <UserSettingsMenu
            authUser={authUser}
            fetchJson={fetchJson}
            onUserUpdated={(user) => setAuthUser(user)}
            themePref={themePref}
            onThemeChange={setThemePref}
            collapsed={sidebarCollapsed}
            autoRefresh={autoRefresh}
            onToggleAutoRefresh={toggleAutoRefresh}
            onLogout={() => void logout()}
          />
        </div>
      </aside>

      <main className="content">
        {watchdog?.degraded && (
          <div className="alert watchdogAlert" title="Verificado automaticamente a cada minuto pelo processo do ForgeRouter; tenta re-escanear a cada poucos minutos, mas nunca mais rápido que isso para não martelar os provedores.">
            <AlertTriangle size={16} />
            <span>
              <b>ForgeRouter degradado:</b> {watchdog.healthy_enabled ?? 0}/{watchdog.total_enabled ?? 0} modelos habilitados saudáveis (mínimo esperado: {watchdog.min_healthy}) — possível falta de internet ou provedores fora do ar.
              {' '}Última verificação: {formatDate(watchdog.last_check_at)}
              {watchdog.last_scan_at && <> · último rescan: {formatDate(watchdog.last_scan_at)}</>}
              {' '}· próxima tentativa de rescan em {watchdogNextRetryLabel(watchdog)}.
            </span>
          </div>
        )}
        {page === 'agents' && connectOpen && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Connect a new agent</h1>
                <p className="subtitle">Register a new AI agent with its own API key and model controls.</p>
              </div>
              <div className="actions">
                <button className="button secondary" onClick={() => setConnectOpen(false)}>
                  <ArrowLeft size={15} /> Back to Agents
                </button>
              </div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}

            <section className="panel editor">
              <div className="panelHeader"><h2>Agent details</h2><span>saved to the agents registry (PostgreSQL)</span></div>
              <div className="form">
                <div className="formGrid">
                  <div className="agentAvatarField">
                    <span>Profile image (optional)</span>
                    <AvatarPicker username={newAgentName || '?'} value={newAgentAvatar} onChange={setNewAgentAvatar} onError={setError} />
                  </div>
                  <label>Agent name<input value={newAgentName} placeholder="e.g. athos" autoFocus onChange={(e) => { const value = e.target.value; setNewAgentName(value); setNewAgentKey(generateAgentKey(value)); }} onKeyDown={(e) => { if (e.key === 'Enter') void createAgent(); }} /></label>
                  <label>Kind
                    <select value={newAgentKind} onChange={(e) => setNewAgentKind(e.target.value as 'agent' | 'service')}>
                      <option value="agent">Agent — real conversational agent (chat/bot/knowledge base)</option>
                      <option value="service">Service — internal caller only (e.g. Hindsight)</option>
                    </select>
                  </label>
                  <label>Description — what this agent is for (optional)<input value={newAgentDescription} placeholder="e.g. used by internal automation tasks" onChange={(e) => setNewAgentDescription(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void createAgent(); }} /></label>
                  <label>API key — auto-generated, stored in the agent's registration; paste it into the agent's connection settings
                    <span className="keyGen">
                      <input className="mono" readOnly value={newAgentKey} />
                      <button className="iconButton" title="Generate a new key" onClick={() => setNewAgentKey(generateAgentKey(newAgentName))}><RefreshCw size={14} /></button>
                      <button className="iconButton" title="Copy key" onClick={() => void copyText(newAgentKey).then((ok) => setScanStatus(ok ? 'agent key copied' : 'copy failed'))}><Copy size={14} /></button>
                    </span>
                  </label>
                </div>
                <div className="formActions">
                  <span className="muted">Each agent gets its own AGENT_API_KEY — it authenticates the agent's /v1 calls and admin actions.</span>
                  <span className="spacer" />
                  <button className="button secondary" onClick={() => setConnectOpen(false)}>Cancel</button>
                  <button className="button" disabled={creatingAgent || !newAgentName.trim()} onClick={() => void createAgent()}>{creatingAgent ? 'Creating…' : 'Create agent'}</button>
                </div>
              </div>
            </section>
          </>
        )}

        {page === 'agents' && !connectOpen && selectedAgent && (() => {
          const agent = agents.find((item) => item.name === selectedAgent);
          if (!agent) return null;
          return (
            <>
              <header className="pageHeader">
                <div>
                  <h1><AgentIdentity name={agent.name} avatarUrl={agent.avatar_data_url} size="lg" /></h1>
                  <p className="subtitle">Configure credentials, deploy settings, monthly budget and model controls.</p>
                </div>
                <div className="actions">
                  <button className="button secondary" onClick={() => setSelectedAgent(null)}>
                    <ArrowLeft size={15} /> Back to Agents
                  </button>
                  <button className="button" disabled={savingAgentModels} onClick={() => void saveAgentModels(agent.name, true)}>
                    {savingAgentModels ? 'Saving…' : 'Save & Return'}
                  </button>
                </div>
              </header>
              {scanStatus && <div className="scanStatus">{scanStatus}</div>}
              {error && <div className="alert">{error}</div>}

              <Panel
                title={<><AgentIdentity name={agent.name} avatarUrl={agent.avatar_data_url} size="md" /><button className="iconButton" title="Rename agent" onClick={() => void renameAgent(agent.name)}><Pencil size={13} /></button></>}
                meta={`created ${agent.created_at ? `${agent.created_at.slice(8, 10)}/${agent.created_at.slice(5, 7)}/${agent.created_at.slice(0, 4)}` : '-'}`}
              >
                <div className="setupRow agentProfileRow">
                  <span>Profile image</span>
                  <AvatarPicker
                    username={agent.name}
                    value={agent.avatar_data_url ?? null}
                    onChange={(next) => void saveAgentAvatar(agent.name, next)}
                    onError={setError}
                  />
                  <span />
                </div>
                <div className="setupRow"><span>API base URL</span><span className="mono">{window.location.origin}/v1</span><button className="iconButton" title="Copy base URL" onClick={() => void copyText(`${window.location.origin}/v1`).then((ok) => setScanStatus(ok ? 'base URL copied' : 'copy failed'))}><Copy size={13} /></button></div>
                <div className="setupRow"><span>API key</span><span className="mono">{agent.api_key_masked || '-'}</span><button className="iconButton" title="Copy API key (requires admin token)" onClick={() => void copyAgentKey(agent.name)}><Copy size={13} /></button></div>
                <div className="setupRow"><span>Model name</span><span className="mono">auto</span><button className="iconButton" title="Copy model name" onClick={() => void copyText('auto').then((ok) => setScanStatus(ok ? 'model name copied' : 'copy failed'))}><Copy size={13} /></button></div>
                <div className="setupRow"><span>Description</span><span>{agent.description || <span className="muted">— what is this agent for?</span>}</span><button className="iconButton" title="Edit description" onClick={() => void editAgentDescription(agent.name, agent.description ?? '')}><Pencil size={13} /></button></div>
                <div className="setupRow"><span>Usage (30d)</span><span>{agent.messages} messages · {formatTokens(agent.tokens)} tokens · {formatCost(agent.cost)} billed · {formatCost(agent.reference_cost)} ref.</span><span /></div>
                <div className="setupRow">
                  <span>Monthly budget</span>
                  <span>
                    {agent.budget_limit_usd != null ? (
                      <>
                        {formatCost(agent.month_spend)} / {formatCost(agent.budget_limit_usd)} this month
                        {agent.month_spend >= agent.budget_limit_usd ? (
                          <b className={`status ${agent.budget_action === 'block' ? 'unhealthy' : 'unknown'}`} title="Reference-cost spend has reached the configured limit">
                            {agent.budget_action === 'block' ? 'blocked' : 'over budget'}
                          </b>
                        ) : (
                          <b className="status healthy">{agent.budget_action}</b>
                        )}
                      </>
                    ) : (
                      <span className="muted">no limit set</span>
                    )}
                  </span>
                  <button className="iconButton" title="Set a monthly reference-cost budget for this agent" onClick={() => void editAgentBudget(agent.name, agent.budget_limit_usd, agent.budget_action)}><Pencil size={13} /></button>
                </div>
                <div className="setupRow">
                  <span>Deploy-config</span>
                  <span>
                    {agent.config_path ? (
                      <>
                        <span className="mono">{agent.config_path}</span> ({agent.config_format || 'env'}){agent.restart_service ? <> · restarts <span className="mono">{agent.restart_service}</span></> : null}
                      </>
                    ) : (
                      <span className="muted">not set — Rotate only updates the database key</span>
                    )}
                  </span>
                  <button className="iconButton" title="Set where this agent's own runtime config lives, so Rotate can write the new key there and restart it" onClick={() => void editAgentDeployConfig(agent.name, agent)}><Pencil size={13} /></button>
                </div>
                <div className="setupRow">
                  <span>Actions</span>
                  <span className="rowActions">
                    <button className="button secondary" title="New API key, same provider/model controls" onClick={() => void rotateAgentKey(agent.name)}><RefreshCw size={14} /> Rotate key</button>
                    <button className="button secondary" title="Clone this agent's provider/model controls into a new agent" onClick={() => void duplicateAgent(agent.name)}><CopyPlus size={14} /> Duplicate</button>
                  </span>
                  <button className="iconButton danger" title="Delete agent" onClick={() => void removeAgent(agent.name)}><Trash2 size={13} /></button>
                </div>
                <div className="agentClients">
                  <div className="agentClientsHeader">
                    <h3>Connect a coding tool <span className="muted">— ready-to-paste config for the base URL and key above, in each tool's own file format</span></h3>
                    <div className="agentClientTabs">
                      {AGENT_CLIENTS.map((client) => (
                        <button
                          key={client.id}
                          type="button"
                          className={`button secondary${agentClientId === client.id ? ' active' : ''}`}
                          onClick={() => setAgentClientId(client.id)}
                        >{client.label}</button>
                      ))}
                    </div>
                  </div>
                  {(() => {
                    const client = AGENT_CLIENTS.find((item) => item.id === agentClientId) ?? AGENT_CLIENTS[0];
                    const baseUrl = agentClientBaseUrl(client);
                    const displayKey = '<KEY AGENT>';
                    return (
                      <div className="agentClientBody">
                        <p className="muted">{client.hint}</p>
                        <pre className="codeBlock mono">{client.block(baseUrl, displayKey)}</pre>
                        <div className="rowActions">
                          <span className="muted mono">{client.filename}</span>
                          <button className="button secondary" onClick={() => void copyAgentClientBlock(agent.name, client)}>
                            <Copy size={13} /> Copy {client.embedsSecret ? 'config with real key' : 'config'}
                          </button>
                          {client.secret && (
                            <button className="button secondary" onClick={() => void copyAgentClientSecret(agent.name, client)}>
                              <Copy size={13} /> Copy {client.secret.label}
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })()}
                </div>
                <div className="agentModels">
                  <div className="modelsHeader">
                    <h3>Model controls <span className="muted">— the agent only routes through its associated models; register providers on the Routing page with this agent selected, or search to add models manually</span></h3>
                    <div className="filters">
                      <input className="searchBox" type="search" placeholder="Search model to add…" value={agentModelSearch} onChange={(e) => setAgentModelSearch(e.target.value)} />
                      <button
                        className="button secondary"
                        title="Add healthy models the agent doesn't already have (active or disabled)"
                        onClick={() => setAgentModelsDraft((draft) => {
                          const known = new Set([...draft, ...(agent.models_off ?? [])]);
                          const newHealthyIds = registry.flatMap((provider) => provider.models.map((model) => model.id))
                            .filter((id) => healthByModel[id] === 'healthy' && !known.has(id));
                          return [...draft, ...newHealthyIds];
                        })}
                      >Add all healthy</button>
                      <button className="button" disabled={savingAgentModels} onClick={() => void saveAgentModels(agent.name, true)}>{savingAgentModels ? 'Saving…' : `Save & Return (${agentModelsDraft.length} models)`}</button>
                    </div>
                  </div>
                  <div className="agentModelChips">
                    {registry.flatMap((provider) => provider.models.map((model) => model.id))
                      .filter((id) => agentModelSearch.trim()
                        ? id.toLowerCase().includes(agentModelSearch.trim().toLowerCase())
                        : agentModelsDraft.includes(id) || (agent.models_off ?? []).includes(id))
                      .map((id) => (
                        <button
                          key={id}
                          type="button"
                          className={`cap toggle${agentModelsDraft.includes(id) ? ' active' : ''}`}
                          onClick={() => setAgentModelsDraft((draft) => draft.includes(id) ? draft.filter((item) => item !== id) : [...draft, id])}
                        >{id}</button>
                      ))}
                    {!agentModelsDraft.length && !agentModelSearch.trim() && <p className="muted">No models yet — this agent has no providers. Register one on the Routing page with this agent selected, or search above to add models manually.</p>}
                  </div>
                </div>
              </Panel>
            </>
          );
        })()}

        {page === 'agents' && !connectOpen && !selectedAgent && (
          <>
            <header className="pageHeader">
              <div>
                <h1>My Agents</h1>
                <p className="subtitle">View and manage all your connected AI agents. Each agent authenticates with its own API key.</p>
              </div>
              <div className="actions">
                <button className="button" onClick={() => { setSelectedAgent(null); setConnectOpen(true); setNewAgentKey(generateAgentKey(newAgentName)); }}><Plus size={15} /> Connect Agent</button>
                <button className="button secondary" disabled={scanning || refreshingHealth} title="Discover new models, re-catalog and health-scan every provider — unhealthy models are unchecked (on/off) and all agents are updated" onClick={() => void runScan()}>
                  {scanning ? <Loader2 size={14} className="spin" /> : null} {scanning ? 'Scanning…' : 'Run scan'}
                </button>
                <button className="button" disabled={scanning || refreshingHealth} title="Re-check health and latency of the registered models — unhealthy models are unchecked (on/off) and all agents are updated" onClick={() => void refreshHealth()}>
                  {refreshingHealth ? <Loader2 size={14} className="spin" /> : null} {refreshingHealth ? 'Refreshing…' : 'Refresh'}
                </button>
                <button className="button secondary" disabled={associatingHealthy || scanning || refreshingHealth} title="Associate and enable all healthy and online models for all agents" onClick={() => void associateAllHealthy()}>
                  {associatingHealthy ? <Loader2 size={14} className="spin" /> : <CheckCheck size={14} />} {associatingHealthy ? 'Associating…' : 'Associate healthy to all agents'}
                </button>
              </div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}

            {dataLoaded && !agents.length && (
              <section className="infoBox">
                <Info size={16} />
                <p><b>Register your first agent to unlock the app.</b> Overview, Messages, Routing and Playground depend on an agent — click <b>Connect Agent</b>, name it (e.g. <span className="mono">athos</span>) and paste its API key into the agent's connection settings (the key itself lives in the agent's registration here).</p>
              </section>
            )}

            <section className="agentGrid">
              {agents.map((agent) => {
                const agentProviders = new Set(agent.models.map((id) => providerByModel[id] ?? id.split('/')[0]));
                const agentHealthy = agent.models.filter((id) => healthByModel[id] === 'healthy').length;
                const groups: Record<string, number> = { simple: 0, standard: 0, complex: 0, reasoning: 0, vision: 0, audio: 0, code: 0 };
                const categories: Record<string, number> = Object.fromEntries(CAPABILITIES.map((cap) => [cap, 0]));
                for (const id of agent.models) {
                  if (healthByModel[id] !== 'healthy') continue;
                  const score = scoreByModel[id] ?? 0;
                  const caps = capsByModel[id] ?? [];
                  for (const cap of caps) if (cap in categories) categories[cap] += 1;
                  if (caps.includes('reasoning')) groups.reasoning += 1;
                  if (caps.includes('vision')) groups.vision += 1;
                  if (caps.includes('audio')) groups.audio += 1;
                  if (caps.includes('code')) groups.code += 1;
                  if (score >= 50) groups.complex += 1;
                  else if (score >= 30) groups.standard += 1;
                  else groups.simple += 1;
                }
                return (
                  <div key={agent.name} className="agentCard">
                    <span className="agentHead">
                      <span className="agentHeadLeft">
                        <AgentIdentity name={agent.name} avatarUrl={agent.avatar_data_url} size="md" />
                        <b
                          className={`status ${agent.kind === 'service' ? 'unhealthy' : 'unknown'}${togglingKind === agent.name ? ' toggling' : ''}`}
                          title={togglingKind === agent.name ? 'Saving…' : (agent.kind === 'service' ? 'Internal service consuming LLMs through this agent identity — not a conversational agent (no chat/bot/knowledge base). ' : 'Real conversational agent (chat/bot/knowledge base). ') + 'Click to toggle.'}
                          onClick={(e) => { e.stopPropagation(); if (togglingKind !== agent.name) void toggleAgentKind(agent.name, agent.kind); }}
                        >{agent.kind === 'service' ? 'service' : 'agent'}</b>
                        {!agent.enabled && <b className="status unknown">disabled</b>}
                      </span>
                    </span>
                    {agent.description && <span className="muted">{agent.description}</span>}
                    <span className="agentStats">
                      <span>Tokens <b>{formatTokens(agent.tokens)}</b></span>
                      <span>Messages <b>{agent.messages}</b></span>
                      <span>Providers <b>{agentProviders.size}</b></span>
                      <span>Models <b className={agent.models.length ? (agentHealthy ? 'agentHealthy' : 'agentUnhealthy') : ''}>{agentHealthy}/{agent.models.length} healthy</b></span>
                    </span>
                    <span className="muted groupsHint">Click a group below to enable/disable it for this agent</span>
                    <span className="caps agentGroups" title="Healthy models per task group — click a group to enable/disable all of it for this agent">
                      {Object.entries(groups).map(([group]) => {
                        const groupIds = modelsInGroup(group);
                        const count = groupIds.filter((id) => agent.models.includes(id)).length;
                        const pressed = count === 0 ? false : count === groupIds.length ? true : 'mixed';
                        return (
                          <button
                            key={group}
                            type="button"
                            className={`cap groupToggle${count === 0 ? ' groupEmpty' : ''}${togglingGroup === `${agent.name}:${group}` ? ' toggling' : ''}`}
                            title={`Click to ${count > 0 ? 'disable' : 'enable'} all "${group}" models for ${agent.name}`}
                            aria-pressed={pressed}
                            aria-busy={togglingGroup === `${agent.name}:${group}`}
                            disabled={!groupIds.length || Boolean(togglingGroup)}
                            onClick={(e) => { e.stopPropagation(); void toggleAgentModelSet(agent, groupIds, group); }}
                          >{group} {count}</button>
                        );
                      })}
                    </span>
                    <span className="caps agentGroups" title="Task map — healthy models the agent can serve each task with">
                      {taskMap.map((item) => {
                        const target = item.model.startsWith('forgerouter/') ? item.model.slice('forgerouter/'.length) : item.model;
                        const count = target === 'auto'
                          ? agentHealthy
                          : target in groups
                            ? groups[target]
                            : (agent.models.includes(item.model) && healthByModel[item.model] === 'healthy' ? 1 : 0);
                        return <i key={item.task} className={`cap capTask${count === 0 ? ' groupEmpty' : ''}`} title={`${item.task} → ${item.model}`}>{item.task} {count}</i>;
                      })}
                    </span>
                    <span className="caps agentGroups" title="Healthy models per catalog category (capability)">
                      {Object.entries(categories).map(([cap, count]) => (
                        <i key={cap} className={`cap agentChip${count === 0 ? ' groupEmpty' : ''}`}>{cap} {count}</i>
                      ))}
                    </span>
                    <span className="caps agentGroups" title="Models by cost/access classification — click to enable or disable the group">
                      {(['free', 'paid', 'local'] as CostClass[]).map((costClass) => {
                        const groupIds = modelsInCostClass(costClass);
                        const allClassIds = allModelsInCostClass(costClass);
                        const count = allClassIds.filter((id) => agent.models.includes(id)).length;
                        const pressed = count === 0 ? false : count === groupIds.length ? true : 'mixed';
                        return (
                          <button
                            key={costClass}
                            type="button"
                            className={`cap groupToggle cost${costClass.charAt(0).toUpperCase() + costClass.slice(1)}${count === 0 ? ' groupEmpty' : ''}${togglingGroup === `${agent.name}:${costClass}` ? ' toggling' : ''}`}
                            title={`Click to ${count > 0 ? 'disable' : 'enable'} all ${costClass} models for ${agent.name}`}
                            aria-pressed={pressed}
                            aria-busy={togglingGroup === `${agent.name}:${costClass}`}
                            disabled={(!groupIds.length && !count) || Boolean(togglingGroup)}
                            onClick={() => void toggleAgentModelSet(agent, groupIds, costClass, allClassIds)}
                          >{costClass} {count}</button>
                        );
                      })}
                    </span>
                    <AgentSparkline daily={agent.daily} />
                    <div className="agentCardFooter">
                      <button className="button secondary small" onClick={() => openEditAgent(agent.name)}>
                        <Pencil size={13} /> Edit Agent
                      </button>
                      <button className="button danger small" title={`Delete ${agent.name}`} onClick={() => void removeAgent(agent.name)}>
                        <Trash2 size={13} /> Delete
                      </button>
                    </div>
                  </div>
                );
              })}
              {!agents.length && <p className="muted">No agents yet — click "Connect Agent" to register the first one.</p>}
            </section>
          </>
        )}

        {page === 'overview' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Overview</h1>
                <p className="subtitle">Real-time summary of messages, tokens and provider health.</p>
              </div>
              <div className="filters">{agentSelect}</div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}
            <section className="cards">
              <Metric icon={<Bot />} label="Agents" value={`${agents.length}`} accent="accent-violet" />
              <div className="metric">
                <div className="metricIcon accent-green"><Activity /></div>
                <span>Healthy models</span>
                <strong>{healthy}/{providers.length}</strong>
                <span className="metricSub">Providers <b className={providerTotals.healthy ? 'agentHealthy' : 'agentUnhealthy'}>{providerTotals.healthy}/{providerTotals.total}</b></span>
              </div>
              <div className="metric" title="Saudável: habilitado e respondendo. Desligado: desativado de propósito (manual ou auto), nunca roteado — não é uma queda. Off: habilitado mas não está respondendo agora — é isso que indica um provedor fora do ar ou falta de internet.">
                <div className="metricIcon accent-blue"><HeartPulse /></div>
                <span>Model status</span>
                <strong>{modelStatusCounts.total}</strong>
                <span className="metricSub statusBreakdown">
                  <b className="status healthy">{modelStatusCounts.healthy} saudável</b>
                  <b className="status unknown">{modelStatusCounts.disabled} desligado</b>
                  <b className="status unhealthy">{modelStatusCounts.down} off</b>
                </span>
              </div>
              <Metric icon={<Route />} label={<>Messages (30d){agentFilter !== 'all' && <> — <AgentIdentity name={agentFilter} avatarUrl={agentByName.get(agentFilter)?.avatar_data_url} size="xs" /></>}</>} value={`${usage?.totals.messages ?? 0}`} accent="accent-blue" />
              <div className="metric">
                <div className="metricIcon accent-green"><DollarSign /></div>
                <span>Real cost (30d){agentFilter !== 'all' && <> — <AgentIdentity name={agentFilter} avatarUrl={agentByName.get(agentFilter)?.avatar_data_url} size="xs" /></>}</span>
                <strong>{formatCost(usage?.totals.cost ?? 0)}</strong>
                <div className="costRefBlock">
                  <span>Reference cost (30d)</span>
                  <strong>{formatCost(usage?.totals.reference_cost ?? 0)}</strong>
                </div>
              </div>
            </section>
            <Panel title={<><span className="panelIcon accent-pink"><Layers size={15} /></span>Model groups</>} meta={`${healthy} healthy models · last ${usage?.days ?? 30} days`}>
              <div className="row taskGroups head">
                <span className="groupHead text-green"><span className="panelIcon accent-green"><SignalLow size={13} /></span>Simple</span>
                <span className="groupHead text-blue"><span className="panelIcon accent-blue"><SignalMedium size={13} /></span>Standard</span>
                <span className="groupHead text-amber"><span className="panelIcon accent-amber"><SignalHigh size={13} /></span>Complex</span>
                <span className="groupHead text-violet"><span className="panelIcon accent-violet"><Brain size={13} /></span>Reasoning</span>
                <span className="groupHead text-pink"><span className="panelIcon accent-pink"><Eye size={13} /></span>Vision</span>
                <span className="groupHead text-teal"><span className="panelIcon accent-teal"><AudioLines size={13} /></span>Audio</span>
                <span className="groupHead text-orange"><span className="panelIcon accent-orange"><Code size={13} /></span>Code</span>
              </div>
              <div className="row taskGroups">
                {(['simple', 'standard', 'complex', 'reasoning', 'vision', 'audio', 'code'] as const).map((group) => (
                  <div className="groupCell" key={group}>
                    <strong>{taskGroups[group]}</strong>
                    <div className="groupStats">
                      <span className="statRow">Messages <b>{taskGroupUsage[group].messages}</b></span>
                      <span className="statRow">Tokens <b>{formatTokens(taskGroupUsage[group].tokens)}</b></span>
                      {group === 'code' && (
                        <span className="statRow" title="Code requests served off the general model pool because zero code-capable models were healthy at the time — a capacity gap, not a misclassification">
                          No provider <b>{usage?.totals.code_downgrades ?? 0}</b>
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
            <Panel
              title={<><span className="panelIcon accent-green"><SlidersHorizontal size={15} /></span>Context compaction</>}
              meta={`last ${usage?.days ?? 30} days`}
              extra={
                <label className="check">
                  <input type="checkbox" checked={contextCompaction} disabled={savingCompaction} onChange={() => void toggleContextCompaction()} />
                  {contextCompaction ? 'Enabled' : 'Disabled'}
                </label>
              }
            >
              <div className="row compaction head"><span>Tokens before</span><span>Tokens after</span><span>Tokens saved</span><span>% saved</span></div>
              <div className="row compaction">
                <span>{formatTokens(usage?.totals.tokens_raw ?? 0)}</span>
                <span>{formatTokens((usage?.totals.tokens_raw ?? 0) - (usage?.totals.tokens_saved ?? 0))}</span>
                <span>{formatTokens(usage?.totals.tokens_saved ?? 0)}</span>
                <span>{usage?.totals.pct_saved ?? 0}%</span>
              </div>
            </Panel>
            <Panel
              title={<><span className="panelIcon accent-amber"><Scissors size={15} /></span>Context truncation</>}
              meta="lossy — an LLM summarizes the oldest turns once a request fills the % below of the selected model's real context window"
              extra={
                <label className="check">
                  <input type="checkbox" checked={contextTruncation} disabled={savingTruncation} onChange={() => void toggleContextTruncation()} />
                  {contextTruncation ? 'Enabled' : 'Disabled'}
                </label>
              }
            >
              <div className="setupRow">
                <span>Trigger at</span>
                <span className="mono">{contextTruncationTriggerPercent}% of context window</span>
                <button className="iconButton" title="% of the selected model's real context window (from the pricing catalog) that must fill up before truncation kicks in — mirrors hermes-agent's own 80% compression threshold" onClick={() => void editContextTruncationTriggerPercent()}><Pencil size={13} /></button>
              </div>
              <div className="setupRow">
                <span>Fallback budget</span>
                <span className="mono">{contextTruncationMaxTokens.toLocaleString()} tokens</span>
                <button className="iconButton" title="Only used when the selected model isn't in the pricing catalog, so its real context window is unknown" onClick={() => void editContextTruncationFallbackLimit()}><Pencil size={13} /></button>
              </div>
              <p className="muted small panelNote">
                System prompt and the current turn are always kept. The oldest turns past the trigger are summarized by
                a cheap model (preserving names, decisions, numbers) rather than discarded outright; if the summary
                call fails for any reason, they're dropped instead — a missing summary is always safe, a request that
                never gets answered isn't. Off by default. See the Messages page for which requests were affected.
              </p>
            </Panel>
            {usagePanel}
            {yearlyUsage && yearlyUsage.by_agent.length > 0 && (() => {
              const currentMonth = new Date().getMonth() + 1;
              const months = Array.from({ length: currentMonth }, (_, i) => i + 1);
              const gridCols = `1.6fr ${months.map(() => '.8fr').join(' ')} .9fr`;
              const monthTotals: Record<number, number> = {};
              for (const month of months) monthTotals[month] = 0;
              for (const agentRow of yearlyUsage.by_agent) {
                for (const month of months) monthTotals[month] += agentRow.months[month]?.[usageMetric] ?? 0;
              }
              const grandTotal = yearlyUsage.by_agent.reduce((sum, a) => sum + a.totals[usageMetric], 0);
              return (
                <Panel
                  title={<><span className="panelIcon accent-blue"><LayoutDashboard size={15} /></span>Monthly usage by agent</>}
                  meta={`${yearlyUsage.year} · resets every January`}
                  extra={
                    <button className="button secondary" disabled={archiving} title="Roll up route_events older than the previous month into the monthly totals, then delete them — keeps the database from growing without bound" onClick={() => void archiveOldMessages()}>
                      {archiving ? 'Archiving…' : 'Archive old messages'}
                    </button>
                  }
                >
                  <div className="row monthly head" style={{ gridTemplateColumns: gridCols }}>
                    <span>Agent</span>
                    {months.map((m) => <span key={m}>{MONTH_LABELS[m - 1]}</span>)}
                    <span>Total</span>
                  </div>
                  <div className="tableScroll">
                    {yearlyUsage.by_agent.map((agentRow, index) => (
                      <div className="row monthly" style={{ gridTemplateColumns: gridCols }} key={agentRow.agent}>
                        <span className="agentLegendIdentity"><i className="legendDot" style={{ background: agentColor(index) }} /><AgentIdentity name={agentRow.agent} avatarUrl={agentByName.get(agentRow.agent)?.avatar_data_url} size="xs" /></span>
                        {months.map((m) => <span key={m}>{formatMetricValue(agentRow.months[m]?.[usageMetric] ?? 0, usageMetric)}</span>)}
                        <span><b>{formatMetricValue(agentRow.totals[usageMetric], usageMetric)}</b></span>
                      </div>
                    ))}
                  </div>
                  <div className="row monthly foot" style={{ gridTemplateColumns: gridCols }}>
                    <span>Total</span>
                    {months.map((m) => <span key={m}>{formatMetricValue(monthTotals[m], usageMetric)}</span>)}
                    <span><b>{formatMetricValue(grandTotal, usageMetric)}</b></span>
                  </div>
                </Panel>
              );
            })()}
            {yearlyDemandUsage && yearlyDemandUsage.by_demand.length > 0 && (() => {
              const currentMonth = new Date().getMonth() + 1;
              const months = Array.from({ length: currentMonth }, (_, i) => i + 1);
              const gridCols = `1.6fr ${months.map(() => '.8fr').join(' ')} .9fr`;
              const monthTotals: Record<number, number> = {};
              for (const month of months) monthTotals[month] = 0;
              for (const demandRow of yearlyDemandUsage.by_demand) {
                for (const month of months) monthTotals[month] += demandRow.months[month]?.[usageMetric] ?? 0;
              }
              const grandTotal = yearlyDemandUsage.by_demand.reduce((sum, d) => sum + d.totals[usageMetric], 0);
              return (
                <Panel
                  title={<><span className="panelIcon accent-orange"><Code size={15} /></span>Monthly usage by demand</>}
                  meta={`${yearlyDemandUsage.year} · resets every January`}
                  extra={<span className="muted" title="Requests routed by a concrete model id (no forgerouter/auto or forgerouter/<demand>) have no demand and are not counted here">demand-routed only</span>}
                >
                  <div className="row monthly head" style={{ gridTemplateColumns: gridCols }}>
                    <span>Demand</span>
                    {months.map((m) => <span key={m}>{MONTH_LABELS[m - 1]}</span>)}
                    <span>Total</span>
                  </div>
                  <div className="tableScroll">
                    {yearlyDemandUsage.by_demand.map((demandRow) => (
                      <div className="row monthly" style={{ gridTemplateColumns: gridCols }} key={demandRow.demand}>
                        <span><DemandTag demand={demandRow.demand} /></span>
                        {months.map((m) => <span key={m}>{formatMetricValue(demandRow.months[m]?.[usageMetric] ?? 0, usageMetric)}</span>)}
                        <span><b>{formatMetricValue(demandRow.totals[usageMetric], usageMetric)}</b></span>
                      </div>
                    ))}
                  </div>
                  <div className="row monthly foot" style={{ gridTemplateColumns: gridCols }}>
                    <span>Total</span>
                    {months.map((m) => <span key={m}>{formatMetricValue(monthTotals[m], usageMetric)}</span>)}
                    <span><b>{formatMetricValue(grandTotal, usageMetric)}</b></span>
                  </div>
                </Panel>
              );
            })()}
            {agentFilter === 'all' && usage?.by_agent?.length ? (
              usage.by_agent.map((agentUsage, index) => agentUsage.by_model.length > 0 && (
                <Panel
                  key={agentUsage.agent}
                  title={<><i className="legendDot" style={{ background: agentColor(index) }} />Cost by model — <AgentIdentity name={agentUsage.agent} avatarUrl={agentByName.get(agentUsage.agent)?.avatar_data_url} size="xs" /></>}
                  meta={`${agentUsage.by_model.length} models · ${agentUsage.totals.messages} msgs · ${formatTokens(agentUsage.totals.tokens)} tok · ${formatCost(agentUsage.totals.cost)} · last ${usage.days} days`}
                >
                  <div className="row byModel head"><span>Model</span><span>Messages</span><span>Tokens</span><span>% of total</span><span>Cost</span><span title="Notional cost at public commercial rates for an equivalent model — never billed">Ref. cost</span></div>
                  <div className="tableScroll">
                  {agentUsage.by_model.map((item) => (
                    <div className="row byModel" key={item.model_id}>
                      <span className="mono">{item.model_id}</span>
                      <span>{item.messages}</span>
                      <span>{formatTokens(item.tokens)}</span>
                      <span className="pctCell"><i className="pctBar"><i style={{ width: `${Math.min(100, item.pct_total)}%` }} /></i>{item.pct_total}%</span>
                      <span>{formatCost(item.cost)}</span>
                      <span>{formatCost(item.reference_cost)}</span>
                    </div>
                  ))}
                  </div>
                </Panel>
              ))
            ) : (usage && usage.by_model.length > 0 && (
              <Panel title={<>Cost by model{agentFilter !== 'all' && <> — <AgentIdentity name={agentFilter} avatarUrl={agentByName.get(agentFilter)?.avatar_data_url} size="xs" /></>}</>} meta={`${usage.by_model.length} models · last ${usage.days} days`}>
                <div className="row byModel head"><span>Model</span><span>Messages</span><span>Tokens</span><span>% of total</span><span>Cost</span><span title="Notional cost at public commercial rates for an equivalent model — never billed">Ref. cost</span></div>
                <div className="tableScroll">
                {usage.by_model.map((item) => (
                  <div className="row byModel" key={item.model_id}>
                    <span className="mono">{item.model_id}</span>
                    <span>{item.messages}</span>
                    <span>{formatTokens(item.tokens)}</span>
                    <span className="pctCell"><i className="pctBar"><i style={{ width: `${Math.min(100, item.pct_total)}%` }} /></i>{item.pct_total}%</span>
                    <span>{formatCost(item.cost)}</span>
                    <span>{formatCost(item.reference_cost)}</span>
                  </div>
                ))}
                </div>
              </Panel>
            ))}
          </>
        )}

        {page === 'messages' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Messages</h1>
                <p className="subtitle">Full log of every routed call. Filter by model or status.</p>
              </div>
              <div className="filters">
                {agentSelect}
                <input className="searchBox" type="search" placeholder="Search model / agent / request…" value={msgSearch} onChange={(e) => setMsgSearch(e.target.value)} />
                {(['all', 'success', 'failed'] as const).map((f) => (
                  <button key={f} className={`chip${msgStatus === f ? ' active' : ''}`} onClick={() => setMsgStatus(f)}>{f}</button>
                ))}
              </div>
            </header>
            {error && <div className="alert">{error}</div>}
            <Panel title="Messages" meta={`${visibleRoutes.length}/${routes.length} shown`}>
              <div className="row msg head"><span>Date</span><span>Status</span><span>Model</span><span title="Demand class the request was classified into (forgerouter/auto and forgerouter/<demand> requests only)">Demand</span><span>Agent</span><span>Tokens</span><span>Cost</span><span title="Notional cost at public commercial rates for an equivalent model — never billed">Ref. cost</span><span>% total</span><span /></div>
              <div className="tableScroll">
              {visibleRoutes.map((route) => (
                <React.Fragment key={route.route_id}>
                  <div className="row msg clickable" onClick={() => setExpandedRoute(expandedRoute === route.route_id ? null : route.route_id)}>
                    <span className="mono small">{formatDate(route.created_at)}</span>
                    <span><b className={`status ${route.status}`}>{route.status}</b></span>
                    <span className="mono">{route.model_id ?? '-'}</span>
                    <span><DemandTag demand={route.demand} /></span>
                    <span className="caps">{route.agent ? <span className="cap agentChip"><AgentIdentity name={route.agent} avatarUrl={agentByName.get(route.agent)?.avatar_data_url} size="xs" /></span> : <span className="muted">-</span>}</span>
                    <span>{formatTokens(route.total_tokens)}</span>
                    <span>{route.total_tokens ? formatCost(route.cost) : '-'}</span>
                    <span>{route.reference_cost != null ? formatCost(route.reference_cost) : '-'}</span>
                    <span>{usage?.totals.tokens && route.total_tokens ? `${Math.max(0.1, Math.round(1000 * route.total_tokens / usage.totals.tokens) / 10)}%` : '-'}</span>
                    <span>{expandedRoute === route.route_id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
                  </div>
                  {expandedRoute === route.route_id && (
                    <div className="msgDetail">
                      <p><b>Message</b></p>
                      <p>ID <span className="mono">{route.request_id}</span> · Route #{route.route_id} · Capability <span className="mono">{route.required_capability}</span> · Agent {route.agent ? <AgentIdentity name={route.agent} avatarUrl={agentByName.get(route.agent)?.avatar_data_url} size="xs" /> : <span className="mono">-</span>}</p>
                      <p>Model <span className="mono">{route.model_id ?? '-'}</span> · Demand <span className="mono">{route.demand ?? '-'}</span> · Tokens {route.total_tokens ?? 0} · Cost {formatCost(route.cost)}{route.reference_cost != null ? <> · Ref. cost {formatCost(route.reference_cost)}</> : null}{route.error_type ? <> · Error <span className="mono">{route.error_type}</span></> : null}</p>
                      {route.prompt_preview && (
                        <p title="First ~100 characters of the last user message — ForgeRouter does not persist full conversation content">
                          Prompt <span className="mono">{route.prompt_preview}{route.prompt_preview.length >= 100 ? '…' : ''}</span>
                        </p>
                      )}
                      {((route.context_action && route.context_action !== 'none') || !!route.context_candidates_skipped) && (
                        <p title="What app.context_policy decided for this request's prompt size vs. the selected model's real context window">
                          {route.context_action && route.context_action !== 'none' && (
                            <b className={`status ${route.context_action === 'rejected' ? 'unhealthy' : 'unknown'}`}>context {route.context_action}</b>
                          )}
                          {route.context_budget != null && <> · budget {formatContextLength(route.context_budget)}</>}
                          {!!route.messages_dropped && <> · {route.messages_dropped} message{route.messages_dropped === 1 ? '' : 's'} dropped</>}
                          {!!route.context_candidates_skipped && <> · {route.context_candidates_skipped} incompatible candidate{route.context_candidates_skipped === 1 ? '' : 's'} skipped</>}
                        </p>
                      )}
                    </div>
                  )}
                </React.Fragment>
              ))}
              {!visibleRoutes.length && <div className="row"><span className="muted">No messages match this filter.</span></div>}
              </div>
            </Panel>
          </>
        )}

        {page === 'routing' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Routing</h1>
                <p className="subtitle">Providers, models, health and routing priority.</p>
              </div>
              <div className="actions">
                <button className="button secondary" onClick={() => openEditor(EMPTY_PROVIDER, null)}><Plus size={15} /> Add provider</button>
                <button className="button secondary" disabled={scanning || refreshingHealth} title="Discover new models, re-catalog and health-scan every provider — unhealthy models are unchecked (on/off) and all agents are updated" onClick={() => void runScan()}>
                  {scanning ? <Loader2 size={14} className="spin" /> : null} {scanning ? 'Scanning…' : 'Run scan'}
                </button>
                <button className="button" disabled={scanning || refreshingHealth} title="Re-check health and latency of the registered models — unhealthy models are unchecked (on/off) and all agents are updated" onClick={() => void refreshHealth()}>
                  {refreshingHealth ? <Loader2 size={14} className="spin" /> : null} {refreshingHealth ? 'Refreshing…' : 'Refresh'}
                </button>
              </div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}

            <section className="infoBox">
              <Info size={16} />
              <p>
                <b>AI rank</b>: intelligence score from 0 to 100 — <b>higher is better</b>. It sorts the model lists but is informational only.{' '}
                <b>Tier</b>: routing priority — <b>lower goes first</b>. "auto" tries healthy models in ascending tier order (tier 1 first; local Ollama, tier 4, is the last resort), and the score does not change the fallback order.
              </p>
            </section>

            {editing && (
              <section className="panel editor" ref={editorRef}>
                <div className="panelHeader"><h2>{editingOriginalName ? `Edit provider: ${editingOriginalName}` : 'Add provider'}</h2><span>saved to PostgreSQL registry</span></div>
                <div className="form">
                  <div className="accessRow">
                    <span className="accessLabel">Access</span>
                    {(['subscription', 'api_key', 'local'] as AccessType[]).map((type) => (
                      <button key={type} type="button" className={`chip${(editing.access_type ?? 'api_key') === type ? ' active' : ''}`}
                        onClick={() => setEditing({ ...editing, access_type: type, ...(type === 'local' ? { api_key: '', api_key_env: '' } : {}), ...(type === 'subscription' ? { cost_type: 'paid' as const } : {}) })}>
                        {type === 'subscription' ? 'Subscription' : type === 'api_key' ? 'API Key' : 'Local'}
                      </button>
                    ))}
                    <span className="accessLabel">Cost</span>
                    {(['free', 'paid'] as const).map((cost) => (
                      <button key={cost} type="button" className={`chip${(editing.cost_type ?? 'free') === cost ? ' active' : ''}`}
                        disabled={editing.access_type === 'subscription'}
                        title={editing.access_type === 'subscription' ? 'Subscriptions are always paid plans' : undefined}
                        onClick={() => setEditing({ ...editing, cost_type: cost })}>
                        {cost === 'free' ? 'Free' : 'Paid'}
                      </button>
                    ))}
                    <span className="accessLabel">API format</span>
                    {(['openai', 'anthropic'] as const).map((format) => (
                      <button key={format} type="button" className={`chip${(editing.api_format ?? 'openai') === format ? ' active' : ''}`}
                        title={format === 'openai' ? 'OpenAI-compatible: POST {base URL}/chat/completions' : 'Anthropic Messages API: POST {base URL}/v1/messages'}
                        onClick={() => setEditing({ ...editing, api_format: format })}>
                        {format === 'openai' ? 'OpenAI' : 'Anthropic'}
                      </button>
                    ))}
                  </div>
                  {editing.access_type === 'api_key' && !editingOriginalName && (
                    <div className="formGrid">
                      <label>Free provider preset (autofills endpoint & defaults)
                        <select onChange={(e) => {
                          const preset = FREE_PROVIDER_PRESETS.find((p) => p.name === e.target.value);
                          if (preset) {
                            setEditing({
                              ...editing,
                              name: preset.name,
                              base_url: preset.base_url,
                              cost_type: 'free',
                              api_format: preset.api_format ?? 'openai',
                            });
                          }
                        }}>
                          <option value="">choose a known free provider…</option>
                          {FREE_PROVIDER_PRESETS.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
                        </select>
                      </label>
                    </div>
                  )}
                  {editing.access_type === 'subscription' && (
                    <div className="formGrid">
                      <label>Subscription plan (fills name, URL & headers)
                        <select value={selectedSubscriptionPlanName(subscriptionPlans, editing.name)} onChange={(e) => {
                          const plan = subscriptionPlans.find((p) => p.name === e.target.value);
                          // Full autofill: only the token (when the plan needs one) is left to paste.
                          if (plan) {
                            setEditing({ ...editing, name: plan.name, base_url: plan.base_url, cost_type: 'paid', auth_method: plan.auth_method, api_key: '', auth_config: { ...editing.auth_config, extra_headers: plan.extra_headers } });
                          }
                        }}>
                          <option value="">choose a known plan…</option>
                          {subscriptionPlans.map((plan) => <option key={plan.name} value={plan.name}>{plan.display_name} — {plan.plan_hint}</option>)}
                        </select>
                      </label>
                      {subscriptionPlanAuthUrl(subscriptionPlans, editing.name) && (
                        <a className="button secondary inlineLink" href={subscriptionPlanAuthUrl(subscriptionPlans, editing.name)} target="_blank" rel="noreferrer">
                          <Link2 size={14} /> Authenticate with {selectedSubscriptionPlan(subscriptionPlans, editing.name)?.display_name ?? editing.name}
                        </a>
                      )}
                    </div>
                  )}
                  <div className="formGrid">
                    <label>
                      <span className="labelHead">
                        Name
                        {editing.name && (() => {
                          const links = getProviderLinks(editing.name, editing.base_url);
                          return links.website ? (
                            <a href={links.website} target="_blank" rel="noreferrer" className="inlineIconLink small" title="Open provider website">
                              <ExternalLink size={11} /> website
                            </a>
                          ) : null;
                        })()}
                      </span>
                      <input value={editing.name} placeholder="e.g. groq" onChange={(e) => setEditing({ ...editing, name: e.target.value })} />
                    </label>
                    <label>Tier (1 = first choice)<input type="number" min={1} max={9} value={editing.tier} onChange={(e) => setEditing({ ...editing, tier: Number(e.target.value) || 1 })} /></label>
                    <label>Base URL<input value={editing.base_url} placeholder={editing.access_type === 'local' ? 'http://127.0.0.1:11434/v1' : 'https://api.example.com/v1'} onChange={(e) => setEditing({ ...editing, base_url: e.target.value })} /></label>
                    {editing.access_type !== 'local' && editing.auth_method !== 'oauth' && (
                      <label>
                        <span className="labelHead">
                          <span>{editing.access_type === 'subscription' ? 'Subscription token' : 'API key'} {editing.api_key_set ? `(saved: ${editing.api_key_masked} — leave empty to keep)` : '(empty = no key)'}</span>
                          {(() => {
                            const links = getProviderLinks(editing.name, editing.base_url);
                            return links.key_portal ? (
                              <a href={links.key_portal} target="_blank" rel="noreferrer" className="inlineIconLink small" title="Get API key on provider website">
                                <KeyRound size={11} /> get key
                              </a>
                            ) : null;
                          })()}
                        </span>
                        <input type="password" autoComplete="new-password" value={editing.api_key ?? ''} placeholder={editing.api_key_set ? `saved: ${editing.api_key_masked}` : editing.access_type === 'subscription' ? 'paste the token from your plan' : 'paste the provider API key'} onChange={(e) => setEditing({ ...editing, api_key: e.target.value })} />
                      </label>
                    )}
                    <label className="check"><input type="checkbox" checked={editing.enabled} onChange={(e) => setEditing({ ...editing, enabled: e.target.checked })} /> Provider enabled</label>
                  </div>
                  {editing.access_type === 'subscription' && (
                    <p className="muted small">OAuth plans open the provider login and read a local token file when available; token plans require pasting the provider token above. {subscriptionPlans.length ? 'Token hints: ' + subscriptionPlans.map((p) => `${p.display_name}: ${p.token_hint}`).join(' · ') : ''}</p>
                  )}
                  <div className="modelsHeader">
                    <h3>Models <span className="muted">{editing.access_type === 'local' ? '— "Detect models" queries the local endpoint; if it fails, add models manually and use "Validate configuration"' : '— run "Detect models" to catalog and scan; required before saving'}</span></h3>
                    <div className="actions">
                      <button className="button secondary" disabled={discovering} onClick={() => void discoverModels()}>{discovering ? 'Detecting & scanning… (may take a minute)' : 'Detect models'}</button>
                      <button className="button secondary" onClick={() => setEditing({ ...editing, models: [...editing.models, { id: '', provider_model: '', capabilities: ['text'], enabled: true }] })}>Add model manually</button>
                      {editingOriginalName && (
                        <button className="button secondary" disabled={validating !== null} onClick={() => void validateProvider(editingOriginalName)}>{validating === editingOriginalName ? 'Validating…' : 'Validate configuration'}</button>
                      )}
                    </div>
                  </div>
                  {editing.models.map((model, index) => (
                    <div className="modelRow" key={index}>
                      <input value={model.id} placeholder="public id, e.g. groq/llama-3.1-8b-instant" onChange={(e) => updateModel(index, { id: e.target.value })} />
                      <input value={model.provider_model} placeholder="provider model, e.g. llama-3.1-8b-instant" onChange={(e) => updateModel(index, { provider_model: e.target.value })} />
                      <span className="caps">
                        {CAPABILITIES.map((cap) => (
                          <button
                            key={cap}
                            type="button"
                            className={`cap toggle${model.capabilities.includes(cap) ? ' active' : ''}`}
                            onClick={() => updateModel(index, { capabilities: model.capabilities.includes(cap) ? model.capabilities.filter((c) => c !== cap) : [...model.capabilities, cap] })}
                          >{cap}</button>
                        ))}
                      </span>
                      <b className="rank" title="AI intelligence rank">{model.score ?? scoreByModel[model.id] ?? '-'}</b>
                      <b className={`status ${model.health ?? healthByModel[model.id] ?? 'unknown'}`}>{model.health ?? healthByModel[model.id] ?? 'not scanned'}</b>
                      <label className="check"><input type="checkbox" checked={model.enabled} onChange={(e) => updateModel(index, { enabled: e.target.checked })} /> on</label>
                      <button className="iconButton" title="Remove model" onClick={() => setEditing({ ...editing, models: editing.models.filter((_, i) => i !== index) })}><Trash2 size={15} /></button>
                    </div>
                  ))}
                  {!editing.models.length && <p className="muted">No models yet — click "Detect models" to discover, catalog and scan this provider's free models.</p>}
                  <div className="formActions">
                    <span className="spacer" />
                    <button className="button secondary" onClick={() => { setEditing(null); setEditingOriginalName(null); }}>Cancel</button>
                    <button className="button" disabled={!canSave} title={canSave ? undefined : 'Run "Detect models" first — all models need a health status'} onClick={() => void saveProvider()}>{saving ? 'Saving…' : discovering ? 'Detecting…' : 'Save provider'}</button>
                  </div>
                </div>
              </section>
            )}

            <Panel title="Manage providers" meta={`${visibleRegistry.length}/${registry.length} shown · ${readiness.filter((item) => !item.api_key_required || item.api_key_configured).length}/${readiness.length} ready · ${providerTotals.healthy}/${providerTotals.total} providers · ${healthy}/${providers.length} models healthy`}>
              <div className="row manage head"><span>Provider</span><span>Tier</span><span>Base URL</span><span>API key</span><span>Status</span><span>Models</span><span>Actions</span></div>
              <div className="tableScroll">
              {visibleRegistry.map((provider) => {
                const ready = readyByName[provider.name];
                const health = healthByProvider[provider.name];
                const links = getProviderLinks(provider.name, provider.base_url);
                return (
                  <div className="row manage" key={provider.name}>
                    <span>
                      {provider.name}
                      {links.website && (
                        <a href={links.website} target="_blank" rel="noreferrer" className="inlineIconLink" title={`Open ${provider.name} website`}>
                          <ExternalLink size={12} />
                        </a>
                      )}
                      {!provider.enabled && <b className="status unknown">disabled</b>}
                      <span className="caps">
                        <i className="cap">{(provider.access_type ?? 'api_key') === 'api_key' ? 'API key' : provider.access_type === 'subscription' && provider.auth_method === 'oauth' ? 'OAuth' : provider.access_type}</i>
                        {provider.api_format === 'anthropic' && <i className="cap" title="Anthropic Messages API (/v1/messages)">anthropic</i>}
                        <i className={`cap ${provider.cost_type === 'paid' ? 'costPaid' : 'costFree'}`}>{provider.cost_type === 'paid' ? 'paid' : 'free'}</i>
                      </span>
                    </span>
                    <span>{provider.tier}</span>
                    <span className="mono small">{provider.base_url}</span>
                    <span className="mono keyCell">
                      {provider.access_type === 'local' ? 'not needed' : ready?.api_key_masked ? `${ready.api_key_masked} (${ready.api_key_source})` : ready?.api_key_env || 'none'}
                      {(ready?.api_key_masked || ready?.api_key_env) && (
                        <button className="iconButton" title="Copy API key" onClick={() => void copyProviderKey(provider.name)}><Copy size={13} /></button>
                      )}
                      {links.key_portal && (
                        <a href={links.key_portal} target="_blank" rel="noreferrer" className="iconButton" title="Get / Manage API key on provider website">
                          <KeyRound size={13} />
                        </a>
                      )}
                    </span>
                    <span title={health ? `${health.healthy} of ${health.total} models healthy` : 'no health data yet — run a scan'}>
                      {health
                        ? <b className={`status ${health.healthy > 0 ? 'healthy' : 'unhealthy'}`}>{health.healthy > 0 ? 'healthy' : 'unhealthy'}</b>
                        : <b className="status unknown">unknown</b>}
                    </span>
                    <span title={health ? `${health.healthy} healthy / ${health.total} total models` : 'total models'}>
                      {health ? <><span className="mono">{health.healthy}</span><span className="muted">/{health.total}</span></> : <span className="mono">{provider.models.length}</span>}
                    </span>
                    <span className="rowActions">
                      <button className="iconButton" title={provider.enabled ? 'Disable provider — its models leave routing (config and key are kept)' : 'Enable provider — its models return to routing'} onClick={() => void toggleProviderEnabled(provider)}>{provider.enabled ? <Power size={15} /> : <PowerOff size={15} />}</button>
                      <button className="iconButton" title="Validate configuration — credential check + real call to each enabled model" disabled={validating !== null} onClick={() => void validateProvider(provider.name)}>{validating === provider.name ? <Loader2 size={15} className="spin" /> : <CheckCircle2 size={15} />}</button>
                      <button className="iconButton" title="Edit" onClick={() => openEditor(provider, provider.name)}><Pencil size={15} /></button>
                      <button className="iconButton danger" title="Delete" onClick={() => void removeProvider(provider.name)}><Trash2 size={15} /></button>
                    </span>
                  </div>
                );
              })}
              {!visibleRegistry.length && <div className="row"><span className="muted">No providers registered yet — click "Add provider".</span></div>}
              </div>
            </Panel>

            <Panel
              title="Manage Models"
              meta={`${visibleProviders.length}/${providers.length} shown · ${providerTotals.healthy}/${providerTotals.total} providers · ${healthy}/${providers.length} models healthy`}
              extra={
                <div className="filters">
                  <input className="searchBox" type="search" placeholder="Search model…" value={modelSearch} onChange={(e) => setModelSearch(e.target.value)} />
                  {(['all', 'healthy', 'unhealthy'] as StatusFilter[]).map((f) => (
                    <button key={f} className={`chip${statusFilter === f ? ' active' : ''}`} onClick={() => setStatusFilter(f)}>{f}</button>
                  ))}
                  <select className="chip select" value={capabilityFilter} onChange={(e) => setCapabilityFilter(e.target.value)}>
                    <option value="all">all capabilities</option>
                    {CAPABILITIES.map((cap) => <option key={cap} value={cap}>{cap}</option>)}
                  </select>
                </div>
              }
            >
              <div className="row head"><span>Provider</span><span>Model</span><span>Tier</span><span>AI rank</span><span>Context</span><span>Capabilities</span><span>Status</span><span>Latency</span><span>Error</span></div>
              <div className="tableScroll">
              {visibleProviders.map((provider) => {
                return <div className="row" key={provider.model_id}><span>{provider.provider}</span><span className="mono">{provider.model_id}</span><span>{provider.tier}</span><span><b className="rank">{scoreByModel[provider.model_id] ?? '-'}</b></span><span className="mono" title={contextByModel[provider.model_id] ? `${contextByModel[provider.model_id]?.toLocaleString()} tokens` : 'Unknown — not in the pricing catalog'}>{formatContextLength(contextByModel[provider.model_id])}</span><span className="caps">{(capsByModel[provider.model_id] ?? []).map((cap) => <i key={cap} className="cap">{cap}</i>)}</span><span><b className={`status ${provider.status}`}>{provider.status}</b></span><span>{formatLatency(provider.latency_ms)}</span><span>{provider.error_message ?? '-'}</span></div>;
              })}
              {!visibleProviders.length && <div className="row"><span className="muted">No providers match this filter.</span></div>}
              </div>
            </Panel>
          </>

        )}

        {page === 'tasks' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Tasks</h1>
                <p className="subtitle">Distribute models by demand class. <span className="mono">forgerouter/auto</span> analyzes each request and routes it to the matching chain; unhealthy models are skipped automatically and re-enter when they recover.</p>
              </div>
              <div className="actions">
                <button className="button" disabled={scanning || refreshingHealth} title="Re-check health and latency of the registered models — unhealthy models are unchecked (on/off) and all agents are updated" onClick={() => void refreshHealth()}>
                  {refreshingHealth ? <Loader2 size={14} className="spin" /> : null} {refreshingHealth ? 'Refreshing…' : 'Refresh'}
                </button>
              </div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}

            <section className="infoBox">
              <Info size={16} />
              <p><b>Token economy:</b> short utility jobs go to small models, keeping the free-tier quotas of the big models for complex work. Models knocked out by a rate limit (429) re-enter routing automatically after a 10-minute cooldown.</p>
            </section>

            <section className="demandGrid">
              {(demandData?.demands ?? []).map((demand) => {
                const draft = demandDrafts[demand] ?? [];
                const isCustom = draft.length > 0;
                const chain = isCustom ? draft : (demandData?.defaults[demand] ?? []);
                // The table is locked until Edit is pressed: while locked the order
                // controls are disabled and the auto-refresh keeps running; Edit
                // unlocks maintenance (and pauses the refresh), Save/Cancel re-lock.
                const isEditing = editingDemands.includes(demand);
                const search = demandSearch[demand] ?? '';
                const matches = search.trim()
                  ? providers
                      .filter((p) => p.status === 'healthy')
                      .filter((p) => !chain.includes(p.model_id) && p.model_id.toLowerCase().includes(search.trim().toLowerCase()))
                      .sort((a, b) => (scoreByModel[b.model_id] ?? 0) - (scoreByModel[a.model_id] ?? 0))
                      .slice(0, 6)
                  : [];
                return (
                  <section className="panel demandCard" key={demand}>
                    <div className="panelHeader">
                      <h2 className="demandTitle">
                        {DEMAND_GROUP_STYLE[demand] && <span className={`panelIcon ${DEMAND_GROUP_STYLE[demand].accent}`}>{DEMAND_GROUP_STYLE[demand].icon}</span>}
                        {demand}
                      </h2>
                      <div className="panelMeta">
                        <button className="vmChip" title="Copy virtual model id" onClick={() => void copyText(`forgerouter/${demand}`).then((ok) => setScanStatus(ok ? `forgerouter/${demand} copied` : 'copy failed'))}>
                          <span className="mono">forgerouter/{demand}</span><Copy size={12} />
                        </button>
                        {isEditing && <b className="status unknown" title="Auto-refresh is paused while you reorder — Save or Cancel to resume">editing</b>}
                      </div>
                    </div>
                    <p className="demandDesc">{demandData?.info[demand]}</p>
                    <div className="chainList">
                      {chain.map((modelId, index) => (
                        <div className={`chainItem${healthByModel[modelId] !== 'healthy' ? ' chainSkipped' : ''}`} key={modelId} title={healthByModel[modelId] !== 'healthy' ? 'Unhealthy right now — routing skips it automatically until it recovers' : undefined}>
                          <b className="chainPos">{index + 1}</b>
                          <span className="mono">{modelId}</span>
                          <b className={`status ${healthByModel[modelId] ?? 'unknown'}`}>{healthByModel[modelId] ?? 'unknown'}</b>
                          <b className="rank">{scoreByModel[modelId] ?? '-'}</b>
                          <span className="rowActions">
                            <button className="iconButton" title={isEditing ? 'Move up' : 'Press Edit to reorder'} disabled={!isEditing || index === 0} onClick={() => { const next = [...chain]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; setDemandDrafts((d) => ({ ...d, [demand]: next })); }}><ChevronUp size={13} /></button>
                            <button className="iconButton" title={isEditing ? 'Move down' : 'Press Edit to reorder'} disabled={!isEditing || index === chain.length - 1} onClick={() => { const next = [...chain]; [next[index], next[index + 1]] = [next[index + 1], next[index]]; setDemandDrafts((d) => ({ ...d, [demand]: next })); }}><ChevronDown size={13} /></button>
                            <button className="iconButton danger" title={isEditing ? 'Remove from the chain' : 'Press Edit to modify'} disabled={!isEditing} onClick={() => setDemandDrafts((d) => ({ ...d, [demand]: chain.filter((id) => id !== modelId) }))}><X size={13} /></button>
                          </span>
                        </div>
                      ))}
                      {!chain.length && <p className="muted demandDesc">No healthy models available for this class yet.</p>}
                    </div>
                    <div className="chainAdd">
                      <input
                        className="searchBox wide"
                        type="search"
                        disabled={!isEditing}
                        placeholder={isEditing ? 'Add model to the chain… (search)' : 'Press Edit to modify the chain'}
                        value={search}
                        onChange={(e) => setDemandSearch((s) => ({ ...s, [demand]: e.target.value }))}
                      />
                      {matches.length > 0 && (
                        <div className="chainMatches">
                          {matches.map((p) => (
                            <button key={p.model_id} className="modelOption" onClick={() => {
                              markDemandEditing(demand);
                              setDemandDrafts((d) => ({ ...d, [demand]: [...(d[demand]?.length ? d[demand] : chain), p.model_id] }));
                              setDemandSearch((s) => ({ ...s, [demand]: '' }));
                            }}>
                              <span className="mono">{p.model_id}</span>
                              <b className="rank">{scoreByModel[p.model_id] ?? '-'}</b>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="formActions chainActions">
                      {/* Icon-only actions — the label shows on hover (title). Reset is a
                          draft action: nothing is persisted until Save. */}
                      <button className="iconButton" disabled={!isEditing || !isCustom} title="Reset — back to the automatic rank-based chain (draft); press Save to persist" onClick={() => setDemandDrafts((d) => ({ ...d, [demand]: [] }))}><RefreshCw size={15} /></button>
                      {isCustom && <b className="status healthy">custom</b>}
                      <span className="spacer" />
                      <button className="iconButton" disabled={isEditing} title={isEditing ? 'Already editing — Save or Cancel to finish' : 'Edit — unlock this chain for maintenance (pauses the auto-refresh while you work)'} onClick={() => { markDemandEditing(demand); if (!draft.length && chain.length) setDemandDrafts((d) => ({ ...d, [demand]: chain })); }}><Pencil size={15} /></button>
                      <button className="iconButton danger" disabled={!isEditing} title={isEditing ? 'Cancel — discard the unsaved changes, lock the table and resume auto-refresh' : 'Cancel — press Edit first'} onClick={() => { setDemandDrafts((d) => ({ ...d, [demand]: demandData?.routes?.[demand] ?? [] })); clearDemandEditing(demand); }}><X size={15} /></button>
                      <button className="iconButton" disabled={savingDemand === demand || !isEditing} title={isEditing ? (isCustom ? 'Save — persist this chain' : 'Save — persist: back to automatic rank-based routing') : 'Save — press Edit first'} onClick={() => void saveDemand(demand, draft)}>{savingDemand === demand ? <Loader2 size={15} className="spin" /> : <Save size={15} />}</button>
                    </div>
                  </section>
                );
              })}
            </section>

            <Panel
              title="Auxiliary tasks"
              meta="pick a group or model per task"
            >
              {(() => {
                return (
                  <div className="auxToolbar">
                    <div className="auxRow">
                      <Link2 size={15} />
                      <span className="auxLabel">Anthropic base URL</span>
                      <span className="mono auxValue">{window.location.origin}</span>
                      <button className="iconButton" title="Copy the Anthropic-compatible root endpoint for Claude Code" onClick={() => void copyText(window.location.origin).then((ok) => setScanStatus(ok ? 'Anthropic base URL copied' : 'copy failed'))}><Copy size={14} /></button>
                    </div>
                    <div className="auxRow">
                      <Link2 size={15} />
                      <span className="auxLabel">OpenAI base URL</span>
                      <span className="mono auxValue">{`${window.location.origin}/v1`}</span>
                      <button className="iconButton" title="Copy the OpenAI-compatible API base URL for any agent, CLI, or custom client" onClick={() => void copyText(`${window.location.origin}/v1`).then((ok) => setScanStatus(ok ? 'OpenAI base URL copied' : 'copy failed'))}><Copy size={14} /></button>
                    </div>
                    <p className="auxDesc">Paste the base URL and the ids below into your agent/client's own model settings — any agent's own API key (Agents page) authenticates the call, ForgeRouter only checks that the key exists.</p>
                  </div>
                );
              })()}
              <div className="row hermesMap head"><span>Task</span><span>What it does</span><span>Group / model</span><span /></div>
              {taskMap.map((item) => (
                <div className="row hermesMap" key={item.task}>
                  <span>{item.task}</span>
                  <span className="muted">{item.hint}</span>
                  <span>
                    <select className="chip select taskModel" value={item.model} onChange={(e) => void saveTaskModel(item.task, e.target.value)}>
                      <optgroup label="Task groups">
                        {(demandData?.virtual_models ?? DEFAULT_TASK_MAP.map((t) => t.model)).filter((id, i, arr) => arr.indexOf(id) === i).map((id) => <option key={id} value={id}>{id}</option>)}
                      </optgroup>
                      <optgroup label="Healthy models">
                        {playModelOptions.map((p) => <option key={p.model_id} value={p.model_id}>{p.model_id}</option>)}
                      </optgroup>
                      {/* keep the stored value selectable even when it is not in the lists above */}
                      {![...(demandData?.virtual_models ?? []), ...playModelOptions.map((p) => p.model_id)].includes(item.model) && <option value={item.model}>{item.model}</option>}
                    </select>
                  </span>
                  <span><button className="iconButton" title="Copy model id" onClick={() => void copyText(item.model).then((ok) => setScanStatus(ok ? `${item.model} copied` : 'copy failed'))}><Copy size={13} /></button></span>
                </div>
              ))}
            </Panel>
          </>
        )}

        {page === 'playground' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>Playground</h1>
                <p className="subtitle">Chat through the router. "auto" picks the best healthy model by tier.{agentFilter !== 'all' && <> Chatting as <AgentIdentity name={agentFilter} avatarUrl={agentByName.get(agentFilter)?.avatar_data_url} size="xs" /> — usage counts for this agent and its model controls apply.</>}</p>
              </div>
            </header>
            {error && <div className="alert">{error}</div>}
            <section className="panel chatPanel">
              <div className="chatThread">
                {!playMsgs.length && <p className="muted chatEmpty">Send a message to test the router. Each call appears in Messages and counts in Overview.</p>}
                {playMsgs.map((message, index) => (
                  <div key={index} className={`chatMsg ${message.role}${message.meta === 'error' ? ' error' : ''}`}>
                    <div className="chatBubble">
                      {message.images?.map((url, i) => <img key={i} className="chatImg" src={url} alt={`attachment ${i + 1}`} />)}
                      {message.content}
                    </div>
                    {message.meta && message.meta !== 'error' && <span className="chatMeta">{message.meta}</span>}
                  </div>
                ))}
                {playSending && <div className="chatMsg assistant"><div className="chatBubble typing">…</div></div>}
                <div ref={chatEndRef} />
              </div>
              <div
                className={`composer${dragOver ? ' dragOver' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); addAttachmentFiles(Array.from(e.dataTransfer.files)); }}
              >
                <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={onPickImage} />
                {attachments.length > 0 && (
                  <div className="composerChips">
                    {attachments.map((attachment) => (
                      <span className="imgChip" key={attachment.id} title={attachment.name}>
                        <img src={attachment.url} alt={attachment.name} />
                        <button className="iconButton" aria-label={`Remove ${attachment.name}`} onClick={() => setAttachments((prev) => prev.filter((item) => item.id !== attachment.id))}><X size={12} /></button>
                      </span>
                    ))}
                  </div>
                )}
                <textarea
                  ref={composerRef}
                  rows={1}
                  autoFocus
                  className="composerInput resize-none"
                  aria-label="Message"
                  placeholder="Type a message… (Enter sends, Shift+Enter for a new line)"
                  value={playInput}
                  onChange={(e) => setPlayInput(e.target.value)}
                  onPaste={(e) => {
                    const files = Array.from(e.clipboardData.files);
                    if (files.length) { e.preventDefault(); addAttachmentFiles(files); }
                  }}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void sendPlayground(); } }}
                />
                <div className="composerBar">
                  <div className="plusWrap" ref={plusRef}>
                    <button className="iconButton plusBtn" aria-label="Add photos and files" aria-expanded={plusMenuOpen} onClick={() => setPlusMenuOpen(!plusMenuOpen)}><Plus size={16} /></button>
                    {plusMenuOpen && (
                      <div className="plusMenu" role="menu">
                        <button role="menuitem" onClick={() => { setPlusMenuOpen(false); fileRef.current?.click(); }}><ImagePlus size={15} /> Add photos &amp; files</button>
                      </div>
                    )}
                  </div>
                  <span className="composerRight">
                    {playInput.trim() && <span className="tokenEst" title="Rough token estimate">~{Math.ceil(playInput.length / 4)} tok</span>}
                    <AgentSelect
                      agents={agents}
                      value={agentFilter === 'all' ? '' : agentFilter}
                      onValueChange={selectAgentFilter}
                      label="Chat as agent (required)"
                      placeholder="Select agent…"
                    />
                    <div className="modelPicker" ref={pickerRef}>
                      <button className="modelPickerBtn" aria-label="Select model" aria-expanded={modelPickerOpen} onClick={() => { setModelPickerOpen(!modelPickerOpen); setPlayModelSearch(''); }}>
                        <span className="mono">{playModel === 'auto' ? 'auto (router decides)' : playModel}</span>
                        {playModel !== 'auto' && <CapIcons caps={capsByModel[playModel] ?? []} />}
                        <ChevronDown size={14} />
                      </button>
                      {modelPickerOpen && (
                        <div className="modelPickerMenu">
                          <input
                            className="searchBox wide"
                            type="search"
                            autoFocus
                            placeholder="Search healthy models…"
                            value={playModelSearch}
                            onChange={(e) => setPlayModelSearch(e.target.value)}
                          />
                          <div className="modelPickerList" role="listbox">
                            <button role="option" aria-selected={playModel === 'auto'} className={`modelOption${playModel === 'auto' ? ' active' : ''}`} onClick={() => { setPlayModel('auto'); setModelPickerOpen(false); }}>
                              <span className="mono">auto (router decides)</span>
                            </button>
                            {(demandData?.demands ?? [])
                              .map((demand) => `forgerouter/${demand}`)
                              .filter((id) => !playModelSearch.trim() || id.includes(playModelSearch.trim().toLowerCase()))
                              .map((id) => (
                                <button key={id} role="option" aria-selected={playModel === id} className={`modelOption${playModel === id ? ' active' : ''}`} onClick={() => { setPlayModel(id); setModelPickerOpen(false); }}>
                                  <span className="mono">{id}</span>
                                  <i className="cap">demand</i>
                                </button>
                              ))}
                            {playModelOptions
                              .filter((p) => !playModelSearch.trim() || p.model_id.toLowerCase().includes(playModelSearch.trim().toLowerCase()))
                              .map((p) => (
                                <button key={p.model_id} role="option" aria-selected={playModel === p.model_id} className={`modelOption${playModel === p.model_id ? ' active' : ''}`} onClick={() => { setPlayModel(p.model_id); setModelPickerOpen(false); }}>
                                  <span className="mono">{p.model_id}</span>
                                  <CapIcons caps={capsByModel[p.model_id] ?? []} />
                                  <b className="rank">{scoreByModel[p.model_id] ?? '-'}</b>
                                </button>
                              ))}
                            {!playModelOptions.length && <p className="muted modelEmpty">No healthy models available.</p>}
                          </div>
                        </div>
                      )}
                    </div>
                    <button
                      className="sendBtn"
                      aria-label={playSending ? 'Sending…' : 'Send message'}
                      disabled={playSending || agentFilter === 'all' || (!playInput.trim() && !attachments.length)}
                      onClick={() => void sendPlayground()}
                    >{playSending ? <Loader2 size={17} className="spin" /> : <Send size={16} />}</button>
                  </span>
                </div>
                {dragOver && <div className="dropOverlay">Drop images here</div>}
              </div>
            </section>
          </>
        )}
        {page === 'pricing' && (
          <>
            <header className="pageHeader">
              <div>
                <h1>LLM Pricing</h1>
                <p className="subtitle">Reference/notional cost catalog — what each routed model would cost at public commercial rates for an equivalent model. Never billed: ForgeRouter only routes to free-tier models. Used to estimate spend in Overview/Messages when a provider reports no cost.</p>
              </div>
              <div className="actions">
                <input className="searchBox" type="search" placeholder="Search model…" value={pricingSearch} onChange={(e) => setPricingSearch(e.target.value)} />
                <button className="button" disabled={pricingSyncing} title="Refresh the pricing catalog from LiteLLM's public price list and from every provider's live /models pricing, then backfill reference cost on past messages that predate a model being priced" onClick={() => void syncPricing()}>
                  {pricingSyncing ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} Sync
                </button>
              </div>
            </header>
            {scanStatus && <div className="scanStatus">{scanStatus}</div>}
            {error && <div className="alert">{error}</div>}

            <Panel title="Model pricing catalog" meta={`${visiblePricingModels.length}/${pricingModels.length} shown · ${pricingMeta.priced_count}/${pricingMeta.total_count} priced · last synced ${pricingMeta.last_synced ? formatDate(pricingMeta.last_synced) : 'never'}`}>
              <div className="row pricing head"><span>Model</span><span>Status</span><span>Input $/1M</span><span>Output $/1M</span><span>Source</span></div>
              <div className="tableScroll">
              {visiblePricingModels.map((item) => (
                <div className="row pricing" key={item.public_id}>
                  <span className="mono">{item.public_id}</span>
                  <span>{item.priced ? <b className="status healthy">priced</b> : <b className="status unknown">unpriced</b>}</span>
                  <span>{item.input_cost_per_token != null ? `$${(item.input_cost_per_token * 1_000_000).toFixed(3)}` : '-'}</span>
                  <span>{item.output_cost_per_token != null ? `$${(item.output_cost_per_token * 1_000_000).toFixed(3)}` : '-'}</span>
                  <span className="small muted">{item.source ?? '-'}</span>
                </div>
              ))}
              {!visiblePricingModels.length && <p className="muted modelEmpty">No models match this search.</p>}
              </div>
            </Panel>
          </>
        )}
        {page === 'users' && authUser.is_admin && <UsersAdminPage fetchJson={fetchJson} currentUsername={authUser.username} />}
        {page === 'profiles' && authUser.is_admin && <ProfilesAdminPage fetchJson={fetchJson} />}
      </main>
      {promptModal && (
        <div className="modalOverlay" onClick={() => { promptModal.resolve(null); setPromptModal(null); }}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <h2>{promptModal.title}</h2>
            <label>
              <input
                autoFocus
                value={promptModal.value}
                onChange={(e) => setPromptModal({ ...promptModal, value: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { promptModal.resolve(promptModal.value); setPromptModal(null); }
                  if (e.key === 'Escape') { promptModal.resolve(null); setPromptModal(null); }
                }}
              />
            </label>
            <div className="modalActions">
              <button className="button secondary" onClick={() => { promptModal.resolve(null); setPromptModal(null); }}>Cancel</button>
              <button className="button" onClick={() => { promptModal.resolve(promptModal.value); setPromptModal(null); }}><Save size={15} /> OK</button>
            </div>
          </div>
        </div>
      )}
      {confirmModal && (
        <div className="modalOverlay" onClick={() => { confirmModal.resolve(false); setConfirmModal(null); }}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <h2><AlertTriangle size={17} /> {confirmModal.title}</h2>
            <p className="muted">{confirmModal.message}</p>
            <div className="modalActions">
              <button className="button secondary" autoFocus onClick={() => { confirmModal.resolve(false); setConfirmModal(null); }}>Cancel</button>
              <button className={confirmModal.danger ? 'button danger' : 'button'} onClick={() => { confirmModal.resolve(true); setConfirmModal(null); }}>
                {confirmModal.danger ? <Trash2 size={15} /> : <RefreshCw size={15} />} {confirmModal.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const AGENT_COLORS = ['#2dd4bf', '#a78bfa', '#f59e0b', '#38bdf8', '#f472b6', '#a3e635', '#fb7185', '#facc15'];

function agentColor(index: number): string {
  return AGENT_COLORS[index % AGENT_COLORS.length];
}

function UsageChart({ seriesList, metric, days }: { seriesList: { name: string; avatarUrl?: string | null; daily: UsageDay[] }[]; metric: UsageMetric; days: number }) {
  const dayKeys: string[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    dayKeys.push(date.toISOString().slice(0, 10));
  }
  const lines = seriesList.map((series, index) => {
    const byDay = Object.fromEntries(series.daily.map((item) => [item.day, item[metric]]));
    return { name: series.name, avatarUrl: series.avatarUrl, color: agentColor(index), values: dayKeys.map((key) => byDay[key] ?? 0) };
  });
  const width = 920, height = 180, padX = 42, padY = 16;
  const max = Math.max(1, ...lines.flatMap((line) => line.values));
  const stepX = (width - padX * 2) / Math.max(1, dayKeys.length - 1);
  const y = (value: number) => height - padY - ((height - padY * 2) * value) / max;
  const fmt = (value: number) => metric === 'cost' || metric === 'reference_cost' ? formatCost(value) : metric === 'tokens' ? formatTokens(value) : `${value}`;
  const single = lines.length === 1;
  return (
    <div className="chartWrap">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="chart" role="img" aria-label="Usage chart">
        {[0, 0.5, 1].map((fraction) => (
          <g key={fraction}>
            <line x1={padX} x2={width - padX} y1={y(max * fraction)} y2={y(max * fraction)} className="gridLine" />
            <text x={padX - 6} y={y(max * fraction) + 4} className="axisLabel" textAnchor="end">{fmt(Math.round(max * fraction * 100) / 100)}</text>
          </g>
        ))}
        {lines.map((line) => {
          const points = line.values.map((value, index) => `${padX + index * stepX},${y(value)}`).join(' ');
          const area = `${padX},${height - padY} ${points} ${padX + (dayKeys.length - 1) * stepX},${height - padY}`;
          return (
            <g key={line.name}>
              {single && <polygon points={area} className="chartArea" />}
              <polyline points={points} fill="none" stroke={line.color} strokeWidth={2} />
              {line.values.map((value, index) => value > 0 && (
                <circle key={dayKeys[index]} cx={padX + index * stepX} cy={y(value)} r={3} fill="#0c0c0f" stroke={line.color} strokeWidth={2}>
                  <title>{`${line.name} · ${formatDay(dayKeys[index])}: ${fmt(value)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        <text x={padX} y={height - 2} className="axisLabel">{formatDay(dayKeys[0])}</text>
        <text x={width - padX} y={height - 2} className="axisLabel" textAnchor="end">{formatDay(dayKeys[dayKeys.length - 1])}</text>
      </svg>
      {!single && (
        <div className="chartLegend">
          {lines.map((line) => <span key={line.name}><i className="legendDot" style={{ background: line.color }} /><AgentIdentity name={line.name} avatarUrl={line.avatarUrl} size="xs" /></span>)}
        </div>
      )}
    </div>
  );
}

function AgentSparkline({ daily }: { daily: AgentDaily[] }) {
  const width = 260, height = 44;
  if (!daily.length) return <svg viewBox={`0 0 ${width} ${height}`} className="agentSpark" preserveAspectRatio="none"><line x1={0} x2={width} y1={height - 2} y2={height - 2} className="gridLine" /></svg>;
  const max = Math.max(1, ...daily.map((point) => point.tokens));
  const stepX = width / Math.max(1, daily.length - 1);
  const points = daily.map((point, index) => `${index * stepX},${height - 3 - (height - 8) * (point.tokens / max)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="agentSpark" preserveAspectRatio="none">
      <polyline points={daily.length === 1 ? `0,${height - 3} ${points}` : points} className="chartLine" fill="none" />
    </svg>
  );
}

function Metric({ icon, label, value, accent }: { icon: React.ReactNode; label: React.ReactNode; value: string; accent?: string }) { return <div className="metric"><div className={`metricIcon${accent ? ` ${accent}` : ''}`}>{icon}</div><span>{label}</span><strong>{value}</strong></div>; }
function Panel({ title, meta, extra, children }: { title: React.ReactNode; meta: string; extra?: React.ReactNode; children: React.ReactNode }) { return <section className="panel"><div className="panelHeader"><h2>{title}</h2><div className="panelMeta">{extra}<span>{meta}</span></div></div><div className="table">{children}</div></section>; }

createRoot(document.getElementById('root')!).render(<App />);
