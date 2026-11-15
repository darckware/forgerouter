# ForgeRouter — Software Specification (SPEC)

> **Status documental:** especificação viva do estado atual implementado. Substitui `docs/HERMES_AI_PROXY_ROUTER_SPEC_v1.md` (rascunho inicial, desatualizado — mantido só como referência histórica) como fonte de verdade sobre arquitetura/stack/banco. Complementa — não substitui — `docs/HERMES_AI_PROXY_ROUTER_PRD_v2.md` (produto) e `docs/DATABASE_DECISION.md` (decisão de banco).

## 1. Scope

ForgeRouter é um gateway self-hosted de LLMs: expõe uma única API (compatível com três protocolos de cliente diferentes) na frente de um pool de múltiplos provedores — modelos locais via Ollama, provedores por API key (Groq, OpenRouter, Mistral, NVIDIA, Cloudflare, Cohere, Gemini Studio, GitHub Models, entre outros) e planos de assinatura via OAuth (Claude Code, Codex, Antigravity, DeepSeek, Z.ai) — com seleção por saúde, fallback automático e roteamento por demanda (classificação automática do tipo de requisição). Inclui um dashboard administrativo (React/TypeScript) servido pela própria API, sem serviço separado.

## 2. Canonical Stack Constraints (Linguagens e Frameworks)

### 2.1 Backend

- **Linguagem:** Python ≥ 3.11 (`pyproject.toml`).
- **Framework web:** FastAPI 0.115 + Uvicorn (`uvicorn[standard]`) 0.34, `pydantic` 2.10 para validação/schemas.
- **Acesso a banco:** `psycopg[binary]` 3.2 — SQL puro contra PostgreSQL, **sem ORM** (nem SQLAlchemy nem outro) e **sem ferramenta de migração** — schema/seed são arquivos `.sql` numerados aplicados manualmente (ver §4).
- **Outras dependências relevantes:** `PyYAML` 6.0 (fallback de registry quando o banco está fora do ar), `tiktoken` 0.8 (contagem de tokens para compactação/truncamento de contexto, encoding `cl100k_base`), `wasmtime` 28.0, `httpx` 0.28 (cliente HTTP para os provedores).
- **Testes:** `pytest` 8.3, via `TestClient` do FastAPI; sem banco/provedores reais nos testes (ver §8).

### 2.2 Frontend (dashboard administrativo)

- **Linguagem:** TypeScript.
- **Stack:** React + Vite (build), `lucide-react` para ícones. Sem framework de UI componentizado de terceiros.
- **Particularidade arquitetural:** todo o dashboard vive em um único arquivo `frontend/src/main.tsx` (~3.650 linhas) + `frontend/src/style.css` — não há divisão em componentes por arquivo/pasta, diferente do padrão "um módulo por domínio" usado em outros projetos do mesmo operador (ex.: Duplica, ForgeHub). O build (`npm run build` em `frontend/`) gera `frontend/dist/`, que o `Dockerfile` copia para dentro da imagem — o dashboard é servido pela própria API, não por um servidor separado.

### 2.3 Infraestrutura

- **Contêineres:** tudo roda via Docker — a stack Python do host não tem as dependências necessárias (`CLAUDE.md`: "Everything runs in Docker"). Build sempre via `./scripts/build.sh` (não `docker compose build` puro — motivo documentado no próprio script/CLAUDE.md: precisa gravar o commit na imagem e regravar as tags `forgerouter:<VERSION>` + `forgerouter:latest`).
- **Rede:** o serviço em produção entra na rede `foundation_network`; execuções avulsas (`docker run` para scan de saúde, sync de preço) precisam entrar na mesma rede e replicar o `extra_hosts` (`host.docker.internal:host-gateway`) que o `docker compose` injeta automaticamente — `--network host` resolve esse hostname para um endereço que o `pg_hba.conf` do Postgres da Foundation rejeita.
- **Versão atual:** `0.1.0` (`VERSION`).

## 3. Architecture Overview

### 3.1 Fluxo de requisição (`app/main.py`)

Para `/v1/chat/completions` (o caminho compartilhado por todos os protocolos de entrada):

1. `load_registry_with_db_health()` carrega o registry de provedores do PostgreSQL (`ai_router.providers` + `ai_router.models`, fonte de verdade, gerenciada via dashboard/CRUD), com fallback para `config/providers.yaml` se o banco estiver inacessível ou vazio; sobrepõe o `healthy` de cada modelo com o status mais recente de `ai_router.provider_health`.
2. A capacidade é inferida da requisição (`tool_call` se `tools` estiver presente, senão `text`). Candidatos = modelos saudáveis + habilitados com essa capacidade, ordenados por `tier` ascendente (tier 1 = maior prioridade; Ollama local é tier 4, último recurso).
3. Candidatos são tentados em ordem — um `model` específico na requisição é preferência, não filtro exclusivo (ele só é ordenado primeiro; o resto do pool saudável continua disponível como fallback, para que um 429 de free-tier no modelo pedido nunca interrompa a chamada). Qualquer HTTP ≥ 400 ou exceção grava um `route_event`, marca o modelo como não-saudável (`mark_runtime_failure_unhealthy`) e passa para o próximo candidato. Só retorna `502 all_providers_failed` se todos falharem.

### 3.2 Três protocolos de entrada, um roteador só

- **`POST /v1/chat/completions`** + **`GET /v1/models`** — nativo, compatível com OpenAI, incluindo streaming (SSE).
- **`POST /v1/messages`** — Anthropic Messages API (para Claude Code). Traduzido para um `ChatCompletionRequest` e roteado pelo mesmo `chat_completions()`; streaming é incremental de verdade — consome o `body_iterator` chunk a chunk (`_anthropic_stream_from_openai_chunks`), não substitui por uma resposta completa reformatada.
- **`POST /v1/responses`** — OpenAI Responses API (para o Codex CLI, obrigatório desde que o Codex abandonou `wire_api = "chat"` na v0.138). Mesmo tratamento de streaming incremental (`_responses_stream_from_openai_chunks`).
- **`POST /v1/embeddings`** — compatível com OpenAI, pela mesma seleção de candidatos e fallback do chat.

Fragmentos de tool-call em streaming são repassados como string JSON parcial sem parsing intermediário (tanto OpenAI quanto Anthropic streamam assim nativamente). Uma falha no meio do stream não pode mais fazer failover para outro candidato (limitação aceita) — o stream simplesmente termina cedo, mas os eventos terminais de cada protocolo (`message_stop`, `response.completed`) são sempre emitidos.

### 3.3 Módulos principais (`app/`)

| Módulo | Responsabilidade |
|---|---|
| `main.py` | Rotas FastAPI, roteamento/fallback, tradutores de protocolo |
| `registry.py` | Parsing do YAML, overlay de saúde vinda do banco, prontidão de provedor (só reporta se a env var da API key está definida, nunca o valor) |
| `providers/openai_compatible.py` | Cliente padrão (`api_format: openai`) |
| `providers/anthropic_compatible.py` | Cliente genérico Anthropic Messages API (`api_format: anthropic`), reaproveita a tradução do adapter do Claude Code sem as particularidades de OAuth |
| `providers/plans.py` | Handlers de plano de assinatura (Claude Code, Codex, Antigravity, DeepSeek, Z.ai) — têm precedência sobre `api_format` |
| `pricing.py` | Estimativa de custo de referência (nunca custo real cobrado — ver §6.3) |
| `validation/scanner.py` + `validation/health.py` | Scanner de saúde: envia uma chat completion real por modelo e detecta falhas silenciosas (HTTP 200 com corpo vazio ou texto de erro de cota/billing/auth) |
| `storage.py` | Todo o acesso a PostgreSQL (psycopg, SQL cru, schema `ai_router`) |
| `demand.py` | Classificação de demanda e roteamento por classe (ver §6.1) |
| `routing_state.py` | Estado de roteamento em memória: circuit breaker, sticky routing, `dynamic_score` (ver §6.2) |
| `normalize.py` | Compactação (lossless) e truncamento (lossy, opt-in) de contexto (ver §6.4) |
| `health_watchdog.py` | Verificação de saúde em background |
| `deploy_config.py`, `ranking.py` | Configuração de deploy por agente; ranking/pontuação de modelos |

## 4. Banco de Dados

- **Instância:** PostgreSQL gerenciado pela Foundation — contêiner `foundation_postgres` (pgvector/pg16), porta 5432 do host.
- **Banco:** `forgerouter` (nasceu como `proxyrouter`, renomeado; usuário de aplicação `proxyrouter_user` manteve o nome original).
- **Schema:** `ai_router` — todos os objetos da aplicação.
- **Conexão:** `DATABASE_URL` no `.env` (nunca commitado).
- **Migrations:** SQL numerado em `db/*.sql` (`001` a `040` no momento desta spec), aplicado **manualmente**, sempre como superusuário `foundation` (`docker exec -i foundation_postgres psql -U foundation -d forgerouter < db/0NN_arquivo.sql`) — `proxyrouter_user` não é dono das tabelas. Não há ferramenta de migração (nem Alembic nem equivalente); cada arquivo numerado é uma alteração incremental aplicada em ordem.
- **Tabelas principais:** `providers` (`access_type` subscription/api_key/local, `cost_type` free/paid, `api_format` openai/anthropic, `auth_config` JSONB), `models` (join por `public_id`, ex. `groq/llama-3.1-8b-instant`; `capabilities TEXT[]`, `enabled`), `provider_health` (histórico append-only; `cooldown_seconds` guarda o `Retry-After` de 429s), `route_events` (uma linha por tentativa de provedor — `agent_id`, tokens, `cost` real, `reference_cost` estimado, `demand` resolvido, `prompt_preview` ~100 chars da última mensagem do usuário — única exceção deliberada a nunca persistir conteúdo de conversa, `messages_dropped` do truncamento de contexto), `demand_routes` (chain customizada por classe de demanda), `agents` + `agent_models` (keys, imagem de perfil e controle de modelos por agente), `users` + `sessions` (login do dashboard — PBKDF2, sessões de 7 dias), `usage_monthly` (rollup mensal por agente, sobrevive à poda de linhas brutas), `subscription_catalog`, `settings` (chave/valor genérico, ex. `context_compaction_enabled`, `pricing_last_synced`).
- **Fallback sem banco:** se o Postgres estiver inacessível ou vazio, o registry cai para `config/providers.yaml` — o roteamento nunca para por causa do banco (toda chamada de persistência no caminho de requisição está em try/except).

## 5. API (visão geral)

- `POST /v1/chat/completions`, `GET /v1/models`, `POST /v1/embeddings` — compatíveis com OpenAI.
- `POST /v1/messages` — Anthropic Messages API.
- `POST /v1/responses` — OpenAI Responses API.
- `GET /health` — status do serviço, versão e `git_sha`.
- `GET /admin/providers/health`, `/admin/providers/readiness`, `/admin/providers/registry`, `/admin/routes/recent`, `/admin/agents`, `/admin/usage` — leitura pública (sem token), nunca expõem valores de segredo (keys mascaradas); o dashboard depende de carregá-los sem autenticação.
- Endpoints que alteram estado (`POST /admin/providers/rescan|resync|discover-models`, `POST /admin/providers/{name}/validate`, `PUT`/`DELETE /admin/providers/{name}`, `POST /admin/agents`, `POST /admin/agents/{name}/rotate-key|duplicate`, `PUT /admin/agents/{name}/models`, `DELETE /admin/agents/{name}`, `GET /admin/providers/{name}/key`, `GET /admin/agents/{name}/key`) exigem Bearer token — key de agente registrado ou sessão de dashboard. Sem nenhum agente cadastrado (ou sem banco), o admin fica aberto para a configuração inicial.
- `/auth/login`, `/auth/me`, `/auth/change-password`, `/auth/logout` — login do dashboard (usuário `admin`/`admin` semeado no primeiro login, com troca de senha obrigatória).
- `POST /admin/pricing/sync` — resincroniza as três camadas de preço e faz backfill histórico.

## 6. Business Rules

### 6.1 Roteamento por demanda (`app/demand.py`, tela "Tasks" do dashboard)

Modelos virtuais `forgerouter/auto|simple|standard|complex|reasoning|vision|audio|code` são expostos em `/v1/models`. `auto` classifica cada requisição pelo conteúdo: partes de imagem → `vision`; partes `input_audio` → `audio` (`auto` nunca escolhe `code` desde 2026-11-15: `forgerouter/code` é só explícito e usa modelos gerais com `code` + `tool_call`); sinais de raciocínio → `reasoning`; senão, tamanho do prompt + presença de `tools` (histórico conta com desconto para não classificar respostas curtas em conversas longas como `complex`). Cada classe roteia por uma chain ordenada (`ai_router.demand_routes` ou uma ordem default derivada do `intelligence_score`), depois por qualquer outro candidato saudável. Falhas de runtime (ex. 429) expiram após um cooldown (padrão 10 min, ou o `Retry-After` do provedor). **Regra de auto-inclusão:** quando o pool saudável cai abaixo de `AUTO_INCLUDE_MIN_HEALTHY` (padrão 3), modelos degradados só por falha de runtime voltam como reserva de último recurso — falhas graves (auth, not found, veredito do scanner) nunca voltam sozinhas.

### 6.2 Estado de roteamento em memória (`app/routing_state.py`)

Advisory, reseta a cada restart, nunca exclui o último recurso: **circuit breaker** por provedor (abre após N falhas consecutivas por um cooldown configurável); **sticky routing** (o último modelo que teve sucesso para um agente+demanda fica em primeiro por um TTL, preservando cache de prompt do provedor); `dynamic_score` (taxa de sucesso recente + latência do último scan, cache de 60s) — o `default_chain` agrupa por `intelligence_score` estático mas ordena dentro do grupo pelo score dinâmico.

### 6.3 Custo de referência (`app/pricing.py`)

Como o roteamento é construído em torno de free-tier, o custo real cobrado é quase sempre zero — `reference_cost` estima o que a requisição teria custado a preço público comercial de um modelo equivalente, puramente como custo de oportunidade, nunca confundido com cobrança real. Busca em três níveis, do mais prioritário ao menos: preço ao vivo lido do próprio `/models` do provedor → override manual (`config/model_pricing_overrides.json`) → catálogo vendorizado da LiteLLM (`config/model_pricing.json`). Nunca "chuta" — sem correspondência, sem custo de referência.

### 6.4 Compactação e truncamento de contexto (`app/normalize.py`)

- **Compactação (lossless, sempre ativa por padrão):** normaliza espaços em branco incidentais nas mensagens antes de montar o payload — nenhum conteúdo semântico é removido. Tokens antes/depois são contados (`tiktoken`, `cl100k_base`) e persistidos por `route_event`.
- **Truncamento (lossy, opt-in, desativado por padrão):** quando o prompt estimado ultrapassa `trigger_percent` (padrão 80%) da janela de contexto real do candidato selecionado, remove as turnos mais antigos (nunca separando uma `tool_calls` da sua resposta `tool`) até caber no orçamento — mensagens de sistema e o turno final são sempre preservados. Antes de descartar, tenta condensar os turnos removidos com um modelo barato (mesma chain do `forgerouter/simple`); qualquer falha nessa chamada cai silenciosamente para o descarte simples.

### 6.5 Segurança e autenticação

- Nunca expor segredos: o endpoint de prontidão retorna só nomes de env var e um booleano, nunca os valores; keys de agente/provedor sempre mascaradas nos endpoints de leitura.
- Nunca rotear para um provedor não-saudável.
- Falhas no banco nunca devem quebrar o roteamento — toda chamada de persistência no caminho de requisição é protegida por try/except.

## 7. Testing Strategy

- `pytest` (38 arquivos em `tests/`), via `TestClient` do FastAPI.
- Sem banco nem provedores reais: os testes fazem monkeypatch dos nomes importados em `app.main` (ex. `app.main.load_registry_with_db_health`, `app.main.chat_completion`, `app.main.persist_route_event`).
- Estado em memória de roteamento (`routing_state`) é resetado por uma fixture `autouse` em `tests/conftest.py` entre os testes.
- Comando: `docker compose run --rm forgerouter pytest -q` (ou um arquivo/teste específico).

## 8. Metodologia

- Todo o ciclo de desenvolvimento roda em Docker — não há dependência do runtime Python do host; build sempre via `./scripts/build.sh` (grava commit/tags corretamente).
- Sem ORM e sem ferramenta de migração — SQL cru, migrations numeradas aplicadas manualmente como superusuário `foundation`; decisão registrada em `docs/DATABASE_DECISION.md`.
- Design defensivo consistente: qualquer chamada de persistência no caminho de uma requisição é protegida por try/except — uma falha de banco nunca deve derrubar o roteamento.
- Testes isolam completamente banco/provedores reais via monkeypatch — permite rodar a suíte sem infraestrutura externa.
- Documentação viva em `docs/`: PRD (`HERMES_AI_PROXY_ROUTER_PRD_v2.md`), decisão de banco (`DATABASE_DECISION.md`), regras de negócio (`REGRAS_DA_APLICACAO.md`), manual (`MANUAL.md`), referências de assinatura por provedor (`SUBSCRIPTION_*_REFERENCE.md`).

## 9. Directory Layout (resumo)

```txt
forgerouter/
├── app/                    # backend FastAPI (Python)
│   ├── main.py              # rotas, roteamento/fallback, tradutores de protocolo
│   ├── registry.py          # registry de provedores (DB + fallback YAML)
│   ├── providers/           # clientes por api_format + planos de assinatura
│   ├── validation/          # scanner de saúde
│   ├── demand.py            # roteamento por demanda
│   ├── routing_state.py     # circuit breaker, sticky routing, dynamic score
│   ├── normalize.py         # compactação/truncamento de contexto
│   ├── pricing.py           # custo de referência
│   └── storage.py           # acesso a Postgres (psycopg, SQL cru)
├── frontend/                # dashboard admin (React + TypeScript + Vite)
│   ├── src/main.tsx          # dashboard inteiro em um único arquivo
│   └── dist/                 # build, copiado pro Dockerfile
├── db/                      # migrations SQL numeradas (001–040+), aplicação manual
├── config/                  # providers.yaml, catálogos/overrides de preço
├── tests/                   # pytest, sem banco/provedores reais
├── scripts/                 # build.sh, health_scan_sync.py, sync_pricing.py, install_standalone.sh
├── docs/                    # PRD, SPEC (este arquivo), decisão de banco, regras de negócio, manual
├── Dockerfile, docker-compose*.yml
├── requirements.txt, pyproject.toml
└── VERSION
```

## 10. Deploy

Tudo empacotado em uma única imagem Docker (`Dockerfile`), com o frontend já buildado copiado para dentro (`frontend/dist` precisa ser gerado com `npm run build` em `frontend/` antes do build da imagem, quando o dashboard muda). `ARG GIT_SHA` fica visível em `GET /health`, junto da versão — permite identificar se um contêiner rodando está com o código do commit esperado.

### 10.1 Build — sempre via `./scripts/build.sh`

Nunca `docker compose build` puro: o `docker compose` nomeia a imagem gerada de um jeito que os scripts de cron e o `CLAUDE.md` não esperam. `scripts/build.sh` builda e re-tageia o resultado corretamente (`forgerouter:<VERSION>` **e** `forgerouter:latest`) — sem isso, a tag `latest` fica presa numa imagem antiga silenciosamente (incidente já registrado no `CLAUDE.md`).

### 10.2 Variantes de `docker-compose`

O repositório tem três arquivos `docker-compose*.yml` para cenários diferentes de rede/Postgres (deploy dentro do ecossistema do mantenedor, execução local, e instalação standalone genérica com Postgres bundlado). Os detalhes de rede, volumes e integrações específicas de cada variante — inclusive quaisquer mounts de host — não são reproduzidos aqui por serem informação de infraestrutura sensível; consulte os arquivos `docker-compose*.yml` diretamente com o mantenedor caso precise operacionalizar um deploy.

### 10.3 Instalação standalone (`./scripts/install_standalone.sh`)

Fluxo idempotente (`INSTALL.md`), seguro de rodar mais de uma vez: cria `.env` a partir de `.env.example` (senhas de banco geradas aleatoriamente), sobe o Postgres bundlado do `docker-compose.standalone.yml`, aplica o schema (`db/*.sql` em ordem numérica) e builda/sobe o serviço `forgerouter`. Ao final, `GET /health` confirma o serviço no ar e o dashboard fica disponível para o primeiro login (troca de senha obrigatória). A instalação manual (sem o script) segue os mesmos passos, documentados um a um em `INSTALL.md`.
