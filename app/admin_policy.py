"""Who may administer ForgeRouter.

Admin actions (every /admin/* endpoint and the MCP write tools) are allowed to a logged-in
dashboard session or to an agent whose name is in the admin allow-list. Until 2026-09-28 *any*
registered agent's API key was accepted as an admin credential -- including customer-facing
agents -- so every agent could register or delete providers and agents, and read provider keys.

The allow-list comes from FORGEROUTER_ADMIN_AGENTS (comma-separated agent names, matched
case-insensitively), defaulting to Athos (orchestrator) and Hephaestus (operations), as decided
by Marcelo. Read-only MCP tools keep accepting any agent key (see app/mcp_server.py).
"""
from __future__ import annotations

import os

ADMIN_AGENTS_ENV = "FORGEROUTER_ADMIN_AGENTS"
DEFAULT_ADMIN_AGENTS = ("Athos", "Hephaestus")


def admin_agent_names() -> set[str]:
    raw = os.environ.get(ADMIN_AGENTS_ENV)
    names = raw.split(",") if raw is not None else DEFAULT_ADMIN_AGENTS
    return {name.strip().casefold() for name in names if name.strip()}


def is_admin_agent(agent_name: str | None) -> bool:
    return bool(agent_name) and agent_name.casefold() in admin_agent_names()


def is_admin_token(token: str) -> bool:
    """A dashboard session, or the API key of an allow-listed, enabled agent."""
    from app.storage import find_agent_by_key, session_user

    if not token:
        return False
    try:
        if session_user(token):
            return True
    except Exception:
        pass
    try:
        return is_admin_agent(find_agent_by_key(token))
    except Exception:
        return False
