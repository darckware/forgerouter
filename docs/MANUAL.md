# Manual do ForgeRouter

> Atualizado em 2026-07-11. Manual do operador — como usar, configurar e
> diagnosticar. As invariantes formais da aplicação estão em
> [`REGRAS_DA_APLICACAO.md`](REGRAS_DA_APLICACAO.md); a decisão de banco em
> [`DATABASE_DECISION.md`](DATABASE_DECISION.md).

## 1. O que é

ForgeRouter é um roteador de LLMs auto-hospedado: expõe **uma** API
OpenAI-compatible (e um endpoint Anthropic Messages) na porta **2100** e
distribui cada requisição entre dezenas de providers (Groq, OpenRouter,
Mistral, NVIDIA, Cloudflare, GitHub Models, planos de assinatura como Claude
Code/Codex, Ollama local…), com seleção por saúde, classes de demanda e
fallback automático. O objetivo central é **economia de tokens**: trabalho
pequeno vai para modelo pequeno, preservando as cotas free dos modelos grandes.

## 2. Subir e operar

Tudo roda em Docker — o Python do host não tem as dependências.

```bash
# Serviço local (host networking, necessário para alcançar o Ollama do host)
docker compose -f docker-compose.local.yml up -d --build

# Testes
docker compose run --rm forgerouter pytest -q

# Saúde
curl http://127.0.0.1:2100/health
```

**⚠️ Antes de qualquer `up`/recreate**: confirme que o `.env` real está no
diretório. Ele é gitignorado e pode existir apenas dentro do container em
execução — recupere com
`docker inspect forgerouter --format '{{range .Config.Env}}{{println .}}{{end}}'`
antes de recriar, ou o deploy sobe sem `DATABASE_URL`/keys.

O frontend (Vite + React, `frontend/`) é servido de `frontend/dist`, copiado
para a imagem no build: mudou o dashboard → `npm run build` em `frontend/` →
rebuild da imagem.

## 3. Consumindo a API

### Autenticação

Cada agente (Athos, Opencode…) tem uma key própria `hermes_*` cadastrada na
página **Agents**. Com pelo menos um agente ativo, `/v1/*` exige
`Authorization: Bearer <key do agente>` — a key atribui o consumo ao agente e
aplica os controles de modelo dele.

### Endpoints

| Endpoint | Uso |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible; suporta `stream: true` e tools |
| `GET /v1/models` | Modelos reais + virtuais (`forgerouter/*`) |
| `POST /v1/messages` | Anthropic Messages API (para clientes como Claude Code) |

```bash
curl http://127.0.0.1:2100/v1/chat/completions \
  -H "Authorization: Bearer hermes_..." -H "Content-Type: application/json" \
  -d '{"model": "forgerouter/auto", "messages": [{"role": "user", "content": "oi"}]}'
```

A resposta traz os headers `x-proxyrouter-request-id` e `x-proxyrouter-model`
(qual modelo serviu de fato).

### Modelos virtuais (classes de demanda)

| Modelo | Serve para |
| --- | --- |
| `forgerouter/auto` | Classifica cada pedido e roteia sozinho (recomendado) |
| `forgerouter/simple` | Trabalho utilitário curto: títulos, extração, sim/não |
| `forgerouter/standard` | Chat do dia a dia, resumos, tool calls |
| `forgerouter/complex` | Contexto longo, trabalho multi-etapa |
| `forgerouter/reasoning` | Análise profunda, passo a passo |
| `forgerouter/vision` | Pedidos com imagem (só modelos com a capability) |
| `forgerouter/audio` | Trabalho de áudio (só modelos com a capability) |
| `forgerouter/code` | Geração/edição de código (só modelos com a capability) |

**Como o `auto` classifica**, nesta ordem:

1. Tem imagem → `vision`
2. `auto` **nunca** escolhe `code` (2026-11-15): a classe só é usada quando pedida
   como `forgerouter/code`, e atende só modelos gerais com `code` + `tool_call`.
3. Dica de raciocínio por palavra inteira ("passo a passo", "prove",
   "chain of thought") → `reasoning`
4. Senão, por tamanho: simple (< ~1,5k chars) · standard (< ~8k) · complex —
   onde o **histórico conta só 1/4** do peso (um "obrigado" em conversa longa
   não vira complex); com tools presentes, o piso é standard.

Pedir um **modelo concreto** é preferência, não filtro: ele vai primeiro e o
resto fica de fallback — um 429 nunca para o chamador.

## 4. Como a distribuição decide (pipeline por requisição)

1. **Autenticação/permissões** — key do agente; só os modelos associados a ele.
2. **Capacidade** — tools → `tool_call` · imagem → `vision` · senão `text`.
3. **Saúde** — só modelos healthy entram (ver §6).
4. **Ordenação**:
   - **Sticky** — o último modelo que respondeu com sucesso para aquele
     agente+demanda vai primeiro por `STICKY_TTL_SECONDS` (600) — preserva o
     prompt cache do provider;
   - **Chain da demanda** — customizada (tela Tasks) ou automática por rank;
     dentro da chain a ordem usa o **score dinâmico** (rank estático × taxa de
     sucesso 7d − penalidade de latência) — modelo falhando afunda sem mudar
     de banda;
   - **Demais healthy** por tier ↑ (fallback total).
5. **Reservas** — pool healthy < `AUTO_INCLUDE_MIN_HEALTHY` (3): modelos
   degradados só por falha de runtime reentram no fim da fila.
6. **Circuit breaker** — provider com `BREAKER_THRESHOLD` (4) falhas seguidas
   vai para o fim de tudo por `BREAKER_COOLDOWN_SECONDS` (120); depois entra
   meio-aberto (uma sonda; nova falha reabre). Deprioriza, nunca exclui.
7. **Execução** — tenta em ordem; cada erro grava `route_event`, marca o modelo
   unhealthy (runtime) e segue; todos falharam → `502 all_providers_failed`.

Tudo do breaker/sticky/score dinâmico é estado **em memória** — zera no
restart e se repovoa com o tráfego.

## 5. Tela Tasks (chains por demanda)

Cada classe tem um card com sua chain. **Edit** (✏️) destrava a manutenção
(reordenar, remover, buscar modelo); **Save** persiste; **Reset** (↻) volta
para a chain automática (rascunho — só persiste com Save). Chain vazia =
automática por banda de rank: simple < 30 · standard 30–49 · complex ≥ 50 ·
reasoning/vision/audio/code = só modelos com a capability no catálogo.

O painel **Auxiliary tasks** mapeia as tarefas do Hermes (Main model, Vision,
Compression, TTS…) para grupos `forgerouter/*`, e fornece Base URL + token do
agente aux-tasks para colar na configuração do Hermes.

## 6. Saúde, cooldowns e recuperação

- O **scanner** manda uma chat completion real para cada modelo habilitado e
  classifica a resposta — inclusive "falhas silenciosas" (HTTP 200 com corpo
  vazio ou texto de quota/billing/auth).
- **Falha de runtime** (429, timeout, 5xx durante uma requisição real): o
  modelo sai do routing por **10 minutos** — ou pelo **`Retry-After`** que o
  provider mandar (cap de 6h). Depois volta sozinho.
- **Falha dura** (401/404, veredito do scanner): só volta com rescan —
  botão **Refresh** (Tasks/Routing) ou `POST /admin/providers/rescan`.
- **On/off manual × automático** (`models.manual_off`): desmarcar um modelo no
  editor do provider é permanente — Refresh/Resync nunca o religam. Um modelo
  desligado automaticamente por veredito de saúde continua sendo escaneado
  pelo Refresh e religa sozinho quando recupera.
- Health é histórico append-only em `ai_router.provider_health`.

## 7. Providers e modelos

- **Tipos** (`access_type`): `api_key` (Groq, OpenRouter…), `subscription`
  (Claude Code, Codex, Antigravity, Z.AI — token OAuth resolvido dos diretórios
  de CLI montados no container), `local` (Ollama).
- **Protocolo** (`api_format`): `openai` (padrão) ou `anthropic` (Messages API).
- **Cadastro** pela página Routing: Add provider → discover-models (importa o
  catálogo `/models` do provider, infere capabilities e escaneia) → Save →
  Validate. Só modelos free são importados por padrão.
- **Tier** = prioridade (1 roteia primeiro; Ollama local é 4, último recurso).
  **Rank (score)** = inteligência 0–100 da tabela curada em `app/ranking.py`;
  modelos fora da tabela caem para 20 — ajuste a tabela quando adicionar
  famílias novas.

## 8. Dashboard

| Página | O que faz |
| --- | --- |
| Agents | Keys, controles de modelo por agente, uso |
| Overview | Métricas 30d, grupos de modelos, compactação, custo por modelo |
| Messages | Log bruto de todas as rotas (com a classe `demand` gravada) |
| Routing | Providers/modelos, saúde, validate, cadastro |
| Tasks | Chains por demanda + tarefas auxiliares do Hermes |
| Playground | Chat de teste pelo router (conta como o agente escolhido) |
| Users / Access Profiles | Login do dashboard e permissões por módulo |

Login inicial `admin`/`admin` com troca de senha obrigatória. Endpoints admin
de **leitura** são públicos (nunca expõem secrets); os de **escrita** exigem
key de agente ou sessão do dashboard.

## 9. Variáveis de ambiente (rotamento)

| Variável | Default | Efeito |
| --- | --- | --- |
| `DATABASE_URL` | — | PostgreSQL Foundation (banco `forgerouter`) |
| `AUTO_INCLUDE_MIN_HEALTHY` | 3 | Piso do pool antes de readmitir degradados |
| `BREAKER_THRESHOLD` | 4 | Falhas seguidas para abrir o breaker do provider |
| `BREAKER_COOLDOWN_SECONDS` | 120 | Duração do breaker aberto |
| `STICKY_TTL_SECONDS` | 600 | Janela do sticky routing (0 desliga) |

## 10. Diagnóstico rápido

| Sintoma | Causa provável | Ação |
| --- | --- | --- |
| Modelo healthy no scan mas falha em requisição real | Diferença de payload (histórico: nulls rejeitados por providers estritos — corrigido) | Ver `error_type` em Messages; testar o modelo direto com o payload do router |
| Modelo some do routing e volta em 10 min | Cooldown de runtime (429/timeout) | Normal; `Retry-After` pode alongar |
| Modelo não volta nunca | Falha dura (auth/404/scanner) | Refresh/rescan; conferir key do provider |
| Modelo OFF não religa nem com Refresh | Desligado manualmente (`manual_off`) | Religar à mão no editor do provider |
| Tudo indo para o mesmo modelo | Sticky (10 min) — comportamento esperado | `STICKY_TTL_SECONDS=0` desliga |
| `502 all_providers_failed` | Todos os candidatos falharam | `last_error` no corpo aponta o último motivo |
| `401` no `/v1` | Key de agente ausente/inválida | Página Agents → copiar/rotacionar key |
| Dashboard sem dados | Banco fora | Routing cai para YAML; conferir `foundation_postgres` |

## 11. Consultas úteis (auditoria)

```sql
-- Distribuição real por classe de demanda (últimos 7 dias)
SELECT demand, status, count(*) FROM ai_router.route_events
WHERE created_at > now() - interval '7 days' GROUP BY 1, 2 ORDER BY 1, 2;

-- Últimas rotas de uma classe
SELECT r.created_at, r.status, r.error_type, m.public_id
FROM ai_router.route_events r
LEFT JOIN ai_router.models m ON m.model_id = r.selected_model_id
WHERE r.demand = 'code' ORDER BY r.route_id DESC LIMIT 20;
```
