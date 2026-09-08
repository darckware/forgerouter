-- Constrain route_events.context_action to the values app.main actually
-- writes (see ContextPayload.action / app.context_policy), matching this
-- codebase's convention for enum-like text columns (providers.api_format,
-- agents.kind, agents.budget_action). NULL stays allowed for rows written
-- before 053_context_routing_diagnostics.sql.
ALTER TABLE ai_router.route_events DROP CONSTRAINT IF EXISTS route_events_context_action_check;
ALTER TABLE ai_router.route_events ADD CONSTRAINT route_events_context_action_check
    CHECK (context_action IS NULL OR context_action IN ('none', 'truncated', 'summarized', 'rejected'));
