from contextlib import contextmanager
from datetime import datetime, timezone

import app.storage as storage


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = rows
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params):
        self.query, self.params = query, params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        pass


def patch_connection(monkeypatch, cursor):
    @contextmanager
    def connect():
        yield FakeConnection(cursor)

    monkeypatch.setattr(storage, "db_connect", connect)


def test_persist_route_event_binds_context_diagnostics(monkeypatch):
    cursor = FakeCursor()
    patch_connection(monkeypatch, cursor)
    storage.persist_route_event(
        "00000000-0000-0000-0000-000000000001", "p/model", "text", "success",
        context_window=200_000, context_budget=160_000,
        context_action="summarized", context_candidates_skipped=2,
    )
    assert cursor.params["context_window"] == 200_000
    assert cursor.params["context_budget"] == 160_000
    assert cursor.params["context_action"] == "summarized"
    assert cursor.params["context_candidates_skipped"] == 2


def test_persist_route_event_defaults_context_diagnostics(monkeypatch):
    cursor = FakeCursor()
    patch_connection(monkeypatch, cursor)
    storage.persist_route_event("00000000-0000-0000-0000-000000000001", "p/model", "text", "success")
    assert cursor.params["context_window"] is None
    assert cursor.params["context_budget"] is None
    assert cursor.params["context_action"] == "none"
    assert cursor.params["context_candidates_skipped"] == 0


def test_recent_routes_returns_context_diagnostics(monkeypatch):
    row = (1, "request", "p/model", "text", "success", None,
           datetime.now(timezone.utc), 10, 0, "agent", "simple", None,
           "preview", 3, 200_000, 160_000, "summarized", 2)
    cursor = FakeCursor([row])
    patch_connection(monkeypatch, cursor)
    event = storage.recent_route_events()[0]
    assert event["context_window"] == 200_000
    assert event["context_budget"] == 160_000
    assert event["context_action"] == "summarized"
    assert event["context_candidates_skipped"] == 2
