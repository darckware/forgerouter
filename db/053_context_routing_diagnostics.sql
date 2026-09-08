-- Context-safe routing diagnostics: what app.context_policy decided for a
-- request, without persisting any conversation content. Apply manually as
-- the foundation superuser (see CLAUDE.md) before deploying code that writes
-- these columns.
ALTER TABLE ai_router.route_events
    ADD COLUMN IF NOT EXISTS context_window INTEGER,
    ADD COLUMN IF NOT EXISTS context_budget INTEGER,
    ADD COLUMN IF NOT EXISTS context_action TEXT,
    ADD COLUMN IF NOT EXISTS context_candidates_skipped INTEGER NOT NULL DEFAULT 0;
