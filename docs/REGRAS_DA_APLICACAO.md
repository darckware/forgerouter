# ForgeRouter — Regras da Aplicação

> Documento de referência das regras de negócio e comportamento do ForgeRouter
> (Hermes AI Proxy Router). Atualizado em 2026-11-15. Complementa o PRD
> (`HERMES_AI_PROXY_ROUTER_PRD_v2.md`) e o `CLAUDE.md` com as regras vigentes no código.

---

## 1. Visão geral

- O ForgeRouter expõe **uma única API OpenAI-compatible** (`/v1/chat/completions`,
  `/v1/models`) na porta **2100** e roteia cada chamada entre múltiplos providers
  de LLM (locais e remotos) com seleção por saúde e fallback automático.
- O dashboard (React, servido em `/`) é a ferramenta de operação: agentes,
  providers, modelos, saúde, tarefas e uso.
- A fonte da verdade do registro é o **PostgreSQL** (banco `forgerouter`, schema
  `ai_router`); o `config/providers.yaml` é apenas fallback quando o banco está
  indisponível ou vazio.

## 2. Agentes e autenticação

### 2.1 API `/v1` (consumo)
- **Todo consumo precisa ser atribuído a um agente.** Uma vez que exista ao menos
  um agente cadastrado, `POST /v1/chat/completions` **exige** uma API key de
  agente válida (`Authorization: Bearer hermes_…`). Sem key ou com key inválida →
  **401 `invalid_agent_key`** — nada é roteado nem gravado.
- **Primeiro setup aberto**: sem nenhum agente cadastrado, o `/v1` aceita chamadas
  sem key (mesmo modelo de proteção do `/admin`).
- **Falha de banco nunca derruba o roteamento**: se o store de agentes estiver
  inacessível na hora de validar a key, a chamada passa (sem atribuição) em vez
  de falhar.
- `GET /v1/models` é público (somente leitura do catálogo, não consome nada).

### 2.2 Keys de agente
- A key de cada agente (`hermes_…`) vive **somente no cadastro do agente, no
  banco** (`ai_router.agents`) — nunca em arquivos `.env`. O agente consumidor
  cola a key nas configurações de conexão dele (ex.: provider custom do Hermes).
- **Rotate key** troca a key mantendo a identidade do agente e seus controles de
  modelos. **Duplicate** clona os controles (e a descrição) em um agente novo com
  key própria.
- Nas leituras públicas as keys aparecem **mascaradas**; revelar a key completa
  (`GET /admin/agents/{name}/key`) exige autenticação admin.

### 2.3 Cadastro do agente
- Campos: nome, key (gerada automaticamente), **descrição** (propósito do agente)
  e **imagem de perfil** opcional, recortada em 256 px e persistida como data URI.
  Descrição e imagem permanecem editáveis no painel "Set up agent".
- **Papel aux-tasks é exclusivo**: apenas **um** agente pode ser o agente das
  tarefas auxiliares (flag `aux_tasks`, garantida por índice único no banco).
  Atribuir o papel a outro agente remove do anterior automaticamente.

### 2.4 Dashboard (login)
- Login próprio (`/auth/*`): usuário padrão `admin`/`admin` criado no primeiro
  acesso com troca de senha obrigatória. Senhas com PBKDF2 em `ai_router.users`;
  sessões de 7 dias em `ai_router.sessions`.
- O login protege **apenas o dashboard**. Endpoints `/admin/*` de leitura são
  públicos (sem segredos no payload); os de escrita exigem Bearer de sessão do
  dashboard **ou** key de qualquer agente cadastrado. Sem agentes (ou sem banco),
  o admin fica aberto para o primeiro setup. **Não existe master key em variável
  de ambiente.**

## 3. Associação agente × provider × modelo

- **Regra padrão: todos os providers e modelos são associados a todos os
  agentes.** O sync (`sync_agent_model_associations`) roda em todo Run scan,
  Refresh e salvamento de provider, e garante isso automaticamente — novos
  providers e novos agentes se encontram sozinhos.
- **Opt-out por agente é a exceção persistente**: desligar um modelo para um
  agente (em Model controls na tela My Agents) mantém a linha no banco com `enabled = FALSE`. Por existir, o sync não
  religa — a escolha sobrevive a todos os scans.
- Modelos desabilitados no provider (on/off desmarcado) saem da lista de todos os
  agentes.
- A tela **Routing** foca exclusivamente na validação de infraestrutura, cadastro de provedores e verificação de saúde/latência dos modelos. O gerenciamento e associação de modelos a cada agente é realizado exclusivamente na tela **My Agents** (em Model controls e group toggles).
- Falha na consulta de agente **nunca** interrompe o roteamento.

## 4. Providers

- **Access types**: `subscription` (plano de assinatura), `api_key`, `local`.
- **Cost**: `free` ou `paid`. **Subscription trava o custo em `paid`**
  automaticamente (a opção fica bloqueada no formulário).
- **Tokens/keys de provider ficam sempre em `api_key`** no banco;
  particularidades não-secretas (ex.: headers extras) vão em `auth_config`
  (JSONB). Segredos nunca são logados nem expostos — leituras retornam só nome de
  env var/boolean/valor mascarado.
- **Botão liga/desliga** (tela Routing): desativar tira o provider inteiro do
  roteamento preservando configuração e key; provider desativado também é pulado
  pelo Run scan (não gasta cota).
- **Validate configuration**: checagem de credencial + chamada real a cada modelo
  habilitado, persistindo a saúde.
- **Planos de assinatura** são catalogados em `ai_router.subscription_catalog` e
  listados no formulário. Escolher um plano **preenche automaticamente nome, base
  URL, headers e custo** — só o token fica para colar (quando o plano exige).
- **Cada assinatura com particularidades tem seu próprio handler**
  (`app/providers/plans.py`): protocolo, origem do token e descoberta de modelos
  por plano. Planos OpenAI-compatible (Z.ai, Moonshot, MiniMax, Ollama Cloud) não
  precisam de handler.
  - **OpenAI Codex (ChatGPT Plus/Pro)**: token OAuth resolvido automaticamente do
    login do Codex CLI na máquina (`~/.codex/auth.json`, montado read-only no
    container) — nada é preenchido nem gravado; `chatgpt-account-id` extraído do
    próprio token; modelos lidos do `models_cache.json` do CLI; o adaptador
    traduz chat-completions ↔ Responses API (streaming-only), incluindo tools,
    imagens e usage.
  - **Google Antigravity (Gemini Code Assist)**: token OAuth resolvido
    automaticamente do login do `agy` CLI na máquina
    (`~/.gemini/antigravity-cli/antigravity-oauth-token`, montado read-only no
    container) — nada é preenchido nem gravado; o `cloudaicompanionProject` da
    conta é resolvido via `loadCodeAssist` e cacheado em memória; modelos vêm de
    uma lista estática (família Gemini); o adaptador traduz chat-completions ↔
    `generateContent`/`streamGenerateContent` (Cloud Code Assist), incluindo
    tools, imagens e usage.
  - **Claude Code (Claude Pro/Max)**: token OAuth resolvido automaticamente do
    login do `claude` CLI na máquina (`~/.claude/.credentials.json`, montado
    read-only — só o arquivo de credenciais, não o diretório inteiro, para não
    expor histórico de conversas) — nada é preenchido nem gravado; toda
    requisição leva o header `anthropic-beta: oauth-2025-04-20` e um bloco
    `system` obrigatório de identificação ("You are Claude Code, Anthropic's
    official CLI for Claude.") — sem isso a API retorna um `rate_limit_error`
    genérico independente da cota real; modelos vêm de uma lista estática
    (família Claude 4.x); o adaptador traduz chat-completions ↔ Messages API
    (`/v1/messages`), incluindo tools, imagens e usage. **Atenção**: esse token
    é o mesmo usado pelas sessões interativas do Claude Code na máquina —
    requisições roteadas por aqui compartilham a cota do plano Pro/Max.

## 5. Modelos, saúde e scans

- **Health scan real**: cada modelo recebe um chat completion de verdade e a
  resposta é classificada. **Falhas silenciosas** (HTTP 200 com conteúdo vazio ou
  texto de erro de cota/billing/auth no corpo) são marcadas unhealthy.
- **Run scan** (resync): redescobre os modelos de cada provider habilitado,
  recataloga capacidades/rank e refaz o health scan completo. É ele que encontra
  modelos novos e religa os recuperados.
- **Refresh** (rescan): revalida saúde/latência dos modelos registrados que não
  foram desligados manualmente (não descobre novos).
- **O on/off do modelo segue o veredito do scan, exceto desligamento manual**
  (`models.manual_off`): desmarcar um modelo no editor do provider é
  **permanente** — Refresh e Run scan nunca o religam; só marcá-lo de novo à
  mão. Já um modelo desmarcado **automaticamente** por veredito de saúde
  continua sendo escaneado e religa sozinho quando recupera (unhealthy →
  desmarcado; healthy → religado).
- Ambos os botões existem nas telas Routing, Agents e Tasks e sempre terminam
  sincronizando as associações de todos os agentes.
- **Cooldown de runtime**: modelo derrubado por falha em tempo de execução
  (ex.: 429, timeout) volta ao roteamento automaticamente após **10 minutos**.
- **Nunca rotear para modelo unhealthy** — exceto pela regra de reservas (seção 6,
  camada 6).
- Histórico de saúde é **append-only** (`ai_router.provider_health`).

## 6. Distribuição (camadas, por linha)

```
Requisição → POST /v1/chat/completions
│
├─ 1. AUTENTICAÇÃO ─── key de agente válida? sem key/inválida → 401
│                      banco fora do ar → passa sem atribuição
├─ 2. PERMISSÕES ───── só os modelos associados ("on") ao agente
├─ 3. CAPACIDADE ───── tools → tool_call · imagem → vision · senão text;
│                      classe code (só forgerouter/code) → modelos com code +
│                      tool_call, com rebaixamento para tool_call/text quando
│                      nenhum consegue atender o pedido (seção 7)
├─ 4. SAÚDE ────────── só modelos healthy (falha de runtime volta após 10 min,
│                      ou após o Retry-After enviado pelo provider num 429)
├─ 5. ORDENAÇÃO ─────  a) sticky: o último modelo com sucesso para o par
│                         agente+demanda vai primeiro por STICKY_TTL_SECONDS
│                         (600) — preserva o prompt cache do provider
│                      b) modelo específico pedido → primeiro (preferência,
│                         não exclusivo: o resto fica como fallback)
│                      c) auto / forgerouter/<classe> → chain da tela Tasks
│                         (customizada, ou automática por rank); dentro da
│                         chain a ordem usa o score dinâmico (rank estático ×
│                         taxa de sucesso 7d − penalidade de latência)
│                      d) demais modelos → Tier ↑ e, no mesmo tier, AI rank ↓
│                         (= exatamente a ordem da tabela Manage Models)
├─ 6. RESERVAS ─────── pool saudável < AUTO_INCLUDE_MIN_HEALTHY (padrão 3):
│                      modelos degradados SÓ por runtime reentram no fim da fila
│                      (falhas duras — auth, not found, scanner — nunca voltam aqui)
├─ 7. BREAKER ──────── provider com BREAKER_THRESHOLD (4) falhas seguidas vai
│                      para o fim de tudo por BREAKER_COOLDOWN_SECONDS (120);
│                      meio-aberto depois (uma sonda). Deprioriza, nunca exclui.
│                      Estado em memória — zera no restart.
├─ 8. CONTEXTO ─────── só candidatos cuja janela útil comporta o prompt; rotas
│                      virtuais exigem janela catalogada ≥ 64k (seção 10);
│                      nada cabe → truncamento (se ligado) ou 413 context_too_large
└─ 9. EXECUÇÃO ─────── tenta em ordem; erro (HTTP ≥ 400/exceção) → grava evento
                       (com a classe `demand` resolvida), marca unhealthy
                       (runtime) e tenta o próximo;
                       todos falharam → 502 all_providers_failed
```

- **Tier**: prioridade de roteamento — menor roteia primeiro (local Ollama é
  tier 4, último recurso). **AI rank**: score informativo 0–100 que ordena
  listas e desempata dentro do tier.
- Um 429 no modelo pedido **nunca** pode parar o chamador: o fallback continua.
- **Streaming** (`stream: true`) é suportado; o usage chega no chunk final e é
  contabilizado do mesmo jeito.

## 7. Classes de demanda e tarefas (tela Tasks)

- Modelos virtuais expostos em `/v1/models`: `forgerouter/auto|simple|standard|
  complex|reasoning|vision|audio|code`.
- `auto` classifica cada pedido, nesta ordem: imagem → vision; áudio → audio;
  dica de raciocínio por
  **palavra inteira** ("passo a passo", "prove" — "aprove"/"provedor" não
  disparam) → reasoning; senão tamanho do prompt e presença de tools decidem
  entre simple (< ~1,5k chars), standard (< ~8k) e complex — o histórico da
  conversa conta com peso 1/4, para follow-ups curtos em conversas longas não
  escalarem de classe.
- **`auto` nunca escolhe a classe code** (Marcelo, 2026-11-15). Quem usa o
  ForgeRouter são os agentes do Hermes e os serviços, que conversam e usam
  ferramentas e nunca editam código (código é com Claude Code, Codex e agy, que
  usam login próprio e não passam pelo ForgeRouter). A detecção de código por
  palavras mandava esses pedidos para o grupo de código: memória injetada,
  caminhos de arquivo, "não implemente" nas rotinas (incidentes de 2026-07-23 e
  2026-10-12) e, em 2026-11-15, 639 pedidos numa semana, 13,8% com falha. A classe
  code só existe quando pedida explicitamente: `forgerouter/code`.
- **O grupo `forgerouter/code` é de modelos gerais** com as capabilities `code` **e**
  `tool_call`. Modelo só de código (sem `tool_call`) nunca o atende.
- **Modelos só de código ficam desligados** (desligamento manual, seção 5):
  codestral e north-mini-code. Não servem a agentes, que precisam de texto e
  ferramentas para se comunicar; a configuração fica guardada e religa à mão.
- A classe resolvida é gravada em cada evento (`route_events.demand`) para
  auditoria; pedidos de modelo concreto ficam com demand nula.
- **Cadeias automáticas por banda de rank**: simple = rank < 30 · standard =
  30–49 · complex = ≥ 50 · reasoning/vision/audio = somente modelos com a
  capability no catálogo; code = modelos com `code` e `tool_call` (particularidade de catálogo: sem a capability, o
  modelo não entra na cadeia da classe). A banda usa o rank estático; a ordem
  dentro da cadeia usa o score dinâmico (seção 6).
- **Rebaixamento da classe code** (desde 2026-07-23, confirmado pelo Marcelo em
  2026-11-15): `code` é **preferência de qualidade**, não exigência — um modelo
  sem a capability ainda escreve código, só pior. Quando nenhum modelo do grupo
  consegue atender o pedido (saudável, permitido para o agente e com janela que
  comporta o prompt — seção 10), o pedido **não falha**: cai para o pool geral (`tool_call` se houver tools, senão `text`),
  ordenado pelo score dinâmico. O evento grava `demand = code` com
  `required_capability ≠ code`, o sinal contável de "sem opção de código". Já
  `vision` e `audio` são **exigência**: sem modelo com a capability, não há
  rebaixamento.
- **Cadeia customizada** por classe é salva em `ai_router.demand_routes` e tem
  precedência sobre a automática.
- **Economia de tokens** é o objetivo: trabalho curto vai para modelo pequeno,
  preservando as cotas free dos modelos grandes.
- **Edição das cadeias**: a tabela fica **bloqueada** até apertar **Edit** (✏️).
  Em edição: setas/remover/buscar liberados, Save (💾) habilitado, Edit
  bloqueado, Cancel (✕) descarta. **Reset** (↻) é ação de rascunho — restaura a
  cadeia automática no editor e só persiste com Save. Todos os botões ficam
  sempre visíveis (desabilitados quando não aplicáveis) e a atualização
  automática pausa enquanto o card está em edição.
- **Auxiliary tasks** (mapa de tarefas do Hermes, `ai_router.task_map`): cada
  tarefa auxiliar (Main model, Vision, Compression, TTS Audio Tags, etc.) aponta
  para um grupo `forgerouter/*` ou modelo específico. O painel fornece a **Base
  URL** e o **Token do agente aux-tasks** (papel exclusivo — seção 2.3) para
  colar nas configurações do Hermes.

## 8. Uso e contabilização

- Cada tentativa de provider gera uma linha em `ai_router.route_events`
  (com `agent_id`, modelo, tokens, custo).
- O **Overview** considera apenas consumo **atribuído a agentes** — eventos sem
  agente não aparecem (e, com a regra 2.1, não voltam a existir).
- A página **Messages** mostra o log bruto de todas as rotas.
- Persistência **nunca** quebra o caminho da requisição: toda gravação no banco
  está protegida por try/except.

## 9. Compactação de contexto (lossless)

- Antes de montar o payload para o provider, `app/normalize.py` aplica
  **normalização sem perda semântica** às mensagens: remove espaços/tabs no
  final de linha, colapsa 3+ linhas em branco para 2, e minifica JSON
  válido em mensagens `role: "tool"` (resultados de ferramentas — a
  formatação "pretty" é incidental, não parte do conteúdo). Nada de texto,
  código ou dado é removido — o modelo recebe o mesmo conteúdo, só sem bytes
  de formatação incidentais.
- **Liga/desliga**: `ai_router.settings` (chave `context_compaction_enabled`,
  default `true`) controla o processo globalmente. Endpoints
  `GET /admin/settings/context-compaction` (público) e
  `POST /admin/settings/context-compaction` (admin) — toggle no card
  **"Context compaction"** do Overview. Falha ao ler o setting nunca quebra o
  roteamento: degrada para `enabled = true`.
- **Indicadores ("prova" da redução)**: cada `route_events` grava
  `prompt_tokens_raw` (estimativa via `tiktoken cl100k_base` das mensagens
  *antes* da normalização) e `prompt_tokens_compacted` (mesma estimativa
  *depois*) — o mesmo tokenizer nos dois lados, então o delta isola o efeito
  da compactação (e fica ~0 quando o toggle está desligado, já que as
  mensagens não mudam). O card "Context compaction" mostra Tokens
  before/after/saved e % saved agregados (`/admin/usage` →
  `totals.tokens_raw/tokens_saved/pct_saved`).
- Estimador de tokens é **único e agnóstico de provider** (sempre
  `cl100k_base`, independente de a chamada ir para OpenAI, Anthropic, Groq
  etc.) — não há detecção de "tipo de API" por design. Se o encoding não
  carregar (sem rede no primeiro uso), a contagem é pulada (`None`) e a
  requisição segue normalmente — "falha de tokenizer nunca quebra roteamento",
  igual à regra de falha de banco.

## 10. Janela de contexto e roteamento seguro

- **Nunca chamar um candidato cuja janela útil não comporta o prompt.** Janela
  útil = janela real do modelo × `trigger_percent` (padrão 80%), comparada ao
  prompt normalizado (mesmo estimador da seção 9).
- **Rotas virtuais** (`auto` e `forgerouter/*`) exigem janela **catalogada** de
  pelo menos 64k tokens (`MINIMUM_VIRTUAL_CONTEXT`, o piso do Hermes Agent),
  mesmo que o prompt coubesse. **Janela desconhecida não é passe livre**: o
  modelo é excluído da rota virtual (desde 2026-09-18). Pedido de **modelo
  concreto** pode usar modelo abaixo de 64k ou sem catálogo, se o prompt couber.
- **De onde vem a janela**: fonte documentada (catálogo, preço do provedor,
  `config/model_pricing_overrides.json` com fonte) ou a **descoberta ativa**
  (`POST /admin/pricing/discover-context`, `scripts/discover_context_windows.py`),
  que testa o modelo de verdade e grava só um **limite inferior confirmado** em
  `config/model_context_discovered.json`. **Nunca deduzir a janela por sinais
  indiretos** (ex.: "o modelo irmão tem 1M"). A descoberta nunca roda dentro de
  uma requisição nem no health scan, porque cada modelo custa até 5 chamadas reais;
  um 429 nunca é lido como "prompt longo demais".
- **Quando nenhum candidato cabe**: com truncamento ligado, corta os turnos mais
  antigos (resumidos por um modelo barato quando possível) até caber no maior
  orçamento disponível; o truncamento é **com perda** e vem **desligado** por
  padrão (`context_truncation_enabled`). Desligado, ou se só a parte protegida
  (mensagens de sistema + último turno) já não couber, a resposta é
  **`413 context_too_large`**, sem chamar nenhum provedor.
- Cada pedido grava em `route_events` `context_window`, `context_budget`,
  `context_action` (none/truncated/summarized/rejected) e
  `context_candidates_skipped`.
- **Rebaixamento e reservas contam só candidatos que servem o pedido**
  (2026-11-15). O rebaixamento da classe code (seção 7) dispara quando nenhum
  modelo de código consegue atender o prompt, e as reservas (seção 6, camada 6)
  quando menos de `AUTO_INCLUDE_MIN_HEALTHY` conseguem — pelo mesmo critério
  deste filtro (`app/context_policy.py: serviceable_candidates`). Antes, as duas
  regras olhavam a lista crua: em 2026-11-15, das 00:26 às 00:33 UTC, os únicos
  modelos de código saudáveis eram sem janela catalogada, nenhuma das duas
  disparou e 42 pedidos (Hindsight, Hephaestus, Athos) receberam 413.

## 11. Dashboard — comportamentos

- **Sincronismo**: atualização automática a cada 5s, com **botão global
  "Sync: on/off"** na sidebar (persistido por navegador) e **pausa automática**
  enquanto qualquer cadeia está em edição.
- **Banners de status/erro são contextuais**: aparecem na tela onde a ação rodou
  e somem ao navegar.
- **Tabela Manage Models** (Routing): ordenada por **Tier ↑, AI rank ↓** — a
  mesma ordem do fallback de distribuição (seção 6). Coluna Tier antes de AI rank.
- **Manage providers** vem antes de **Manage Models** na tela Routing.
- Botões de ícone desabilitados ficam esmaecidos e não-clicáveis (estado visual
  obrigatório).
- O app exige um agente cadastrado para liberar as demais telas; Playground
  sempre conversa como um agente específico (a key dele atribui o uso e aplica
  os controles de modelo).

## 12. Banco de dados e migrations

- Banco **`forgerouter`** (Foundation-managed, container
  `hermes_foundation_pg_postgres`), usuário `proxyrouter_user`, schema
  `ai_router`.
- Migrations são os arquivos numerados em `db/*.sql`, aplicados manualmente como
  superusuário: `docker exec -i hermes_foundation_pg_postgres psql -U foundation
  -d forgerouter < db/NNN_arquivo.sql` (o `proxyrouter_user` não é dono das
  tabelas).
- Tabelas: `providers`, `models`, `provider_health` (append-only),
  `route_events` (inclui `prompt_tokens_raw`/`prompt_tokens_compacted` —
  indicadores de compactação de contexto), `agents` (keys + `description` + `avatar_data_url` +
  `aux_tasks` exclusivo), `agent_models` (associações/opt-outs),
  `agent_providers` (participação durável), `demand_routes`, `task_map`,
  `users`, `sessions`, `subscription_catalog`, `settings` (key/value genérico,
  ex.: `context_compaction_enabled`).
- Modelos se ligam por `public_id` (ex.: `local/qwen2.5:1.5b`).
- `.env` guarda apenas `DATABASE_URL` e afins — **nunca commitar**.

## 13. Segurança — resumo das invariantes

1. Segredos nunca são logados, expostos ou retornados sem mascaramento; revelar
   key completa exige autenticação admin.
2. `/v1` exige key de agente válida quando há agentes (consumo sempre atribuído).
3. Falha de banco nunca quebra roteamento (degrada para "sem atribuição"/YAML).
4. Nunca rotear para modelo unhealthy (exceto reservas de runtime, no fim da fila).
5. Endpoints admin de escrita exigem sessão do dashboard ou key de agente;
   leitura é pública e sem segredos.
6. Só um agente detém o papel aux-tasks (garantido pelo banco).
7. Nunca chamar um modelo cuja janela não comporta o prompt; rotas virtuais só
   usam janela catalogada ≥ 64k; janela nunca é deduzida, só documentada ou
   descoberta pelo teste ativo.
