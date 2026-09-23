# Installing ForgeRouter standalone

This is the generic install path for running ForgeRouter on its own — no
pre-existing Docker network, no externally-managed PostgreSQL instance, no
dependency on any particular multi-agent deployment. ForgeRouter itself has
no ties to any specific ecosystem or vendor — it's a generic LLM gateway
that works with any OpenAI-compatible client, Claude Code, or the Codex CLI.

If you're deploying it as part of a larger multi-service setup with its own
shared Docker network and externally-managed Postgres instead, use
`docker-compose.yml` (or `docker-compose.local.yml` for host networking) and
see `CLAUDE.md`, which documents the specific deployment this repo's own
maintainer runs it in.

## What's different from `docker-compose.yml`

- **`docker-compose.standalone.yml`** bundles its own PostgreSQL container —
  it doesn't require an externally-managed database or an existing Docker
  network to join.
- It **omits** the systemd/D-Bus host mounts and `pid: host` that
  `docker-compose.yml` uses for one feature: automatically restarting a
  *sibling agent's own service* on the host after rotating its API key
  (`POST /admin/agents/{name}/rotate-key` with a `restart_service` configured).
  That's inherently tied to running other agents' services on the same host —
  standalone installs don't have that, so key rotation here just updates the
  database; you restart whatever's consuming the key yourself.
- Everything else — routing, providers, the dashboard, agents, demand
  routing, embeddings, streaming — is identical. Nothing in the application
  code differs between the two; it's purely a difference in how the
  container(s) are wired up.

## Quick start (scripted)

```bash
git clone https://github.com/darckware/forgerouter.git
cd forgerouter
./scripts/install_standalone.sh
```

This creates `.env` from `.env.example` (if missing), generates random
database passwords in it, starts PostgreSQL, applies the schema, builds the
image, and starts ForgeRouter. It's safe to re-run — every step is
idempotent.

When it finishes:

```bash
curl http://127.0.0.1:2100/health
```

The dashboard is at `http://127.0.0.1:2100/` — default login `admin` / `admin`,
you'll be asked to change it on first sign-in.

## Manual step-by-step

If you'd rather not run a script blind, here's exactly what it does:

1. **Copy the environment file:**

   ```bash
   cp .env.example .env
   ```

   Set `POSTGRES_PASSWORD` (the bundled Postgres container's admin password)
   and `PROXYROUTER_PASSWORD` (the restricted app role ForgeRouter actually
   connects as — `db/*.sql` grants everything to a role literally named
   `proxyrouter_user`, the fixed name every deployment mode uses) to real
   random values. Add whichever `<PROVIDER>_API_KEY` variables you want
   pre-filled (all of it is also configurable later through the dashboard).

2. **Start PostgreSQL:**

   ```bash
   docker compose -f docker-compose.standalone.yml up -d postgres
   ```

   Wait until it's healthy:

   ```bash
   docker compose -f docker-compose.standalone.yml exec postgres pg_isready -U forgerouter_user -d forgerouter
   ```

3. **Create the app role** (`db/001_initial.sql` creates the schema
   `AUTHORIZATION proxyrouter_user`, and every migration after it grants that
   role access — it has to exist first):

   ```bash
   docker compose -f docker-compose.standalone.yml exec -T postgres \
     psql -U forgerouter_user -d forgerouter -c \
     "CREATE ROLE proxyrouter_user LOGIN PASSWORD 'your-proxyrouter-password';"
   ```

4. **Apply the schema** — every numbered file in `db/`, in order:

   ```bash
   for f in db/*.sql; do
     docker compose -f docker-compose.standalone.yml exec -T postgres \
       psql -U forgerouter_user -d forgerouter -v ON_ERROR_STOP=1 < "$f"
   done
   ```

5. **Build and start ForgeRouter:**

   ```bash
   ./scripts/build.sh
   docker compose -f docker-compose.standalone.yml up -d forgerouter
   ```

## Optional CLI subscription-plan logins

`docker-compose.standalone.yml` mounts (read-only) the local login files each
subscription-plan handler resolves its OAuth token from, if present:

| Plan | Host path |
|---|---|
| OpenAI Codex CLI | `~/.codex` |
| Google Antigravity CLI (`agy`) | `~/.gemini` |
| Claude Code CLI | `~/.claude/.credentials.json` |
| Z.ai | `~/.zai` |
| DeepSeek Web | `~/.deepseek` |

You only need to have logged into the ones you actually want to use — a
missing mount just leaves that provider showing as not-logged-in, it never
breaks anything else.

## Local Ollama fallback

If you run Ollama on the same host, it listens on `127.0.0.1:11434` — not
reachable from inside a container on its own bridge network. Either run
Ollama itself in a container on the same `forgerouter` network and point a
provider's `base_url` at it, or adapt `docker-compose.local.yml`'s
`network_mode: host` approach (dropping its ecosystem-specific mounts the same
way `docker-compose.standalone.yml` does).

## Backups

The bundled Postgres data lives in the `forgerouter_postgres_data` named
volume. Back it up like any other Postgres data directory, e.g.:

```bash
docker compose -f docker-compose.standalone.yml exec postgres \
  pg_dump -U forgerouter_user forgerouter > backup.sql
```
