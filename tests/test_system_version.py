from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_system_version_reports_app_and_git_metadata(monkeypatch):
    monkeypatch.setattr("app.main._read_version", lambda: "0.1.0")
    monkeypatch.setenv("FORGEROUTER_GIT_SHA", "abc1234")
    monkeypatch.setenv("FORGEROUTER_BUILD_DATE", "2026-08-27T12:00:00Z")
    monkeypatch.setattr("app.storage.get_postgres_version", lambda: "PostgreSQL 16.4")
    monkeypatch.setattr("app.main._latest_bundled_migration", lambda: "047_nous_portal_api_key.sql")

    response = client.get("/admin/system/version")

    assert response.status_code == 200
    body = response.json()
    assert body["app_version"] == "0.1.0"
    assert body["git_sha"] == "abc1234"
    assert body["git_commit_url"] == "https://github.com/darckware/forgerouter/commit/abc1234"
    assert body["build_date"] == "2026-08-27T12:00:00Z"
    assert body["postgres_version"] == "PostgreSQL 16.4"
    assert body["latest_migration_bundled"] == "047_nous_portal_api_key.sql"
    assert body["github_repo_url"] == "https://github.com/darckware/forgerouter"


def test_system_version_survives_db_failure(monkeypatch):
    monkeypatch.setenv("FORGEROUTER_GIT_SHA", "unknown")

    def _boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("app.storage.get_postgres_version", _boom)

    response = client.get("/admin/system/version")

    assert response.status_code == 200
    body = response.json()
    assert body["postgres_version"] is None
    assert body["git_commit_url"] is None
