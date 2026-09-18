from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)
VALID_AVATAR = "data:image/jpeg;base64,/9j/2Q=="


def agent_row(**overrides):
    row = {
        "name": "athos",
        "api_key": "hermes_supersecretvalue",
        "enabled": True,
        "created_at": None,
        "avatar_data_url": VALID_AVATAR,
        "messages": 2,
        "tokens": 100,
        "cost": 0.0,
        "daily": [],
    }
    row.update(overrides)
    return row


def test_agents_list_masks_api_keys(monkeypatch):
    monkeypatch.setattr("app.main.list_agents_with_usage", lambda days=30: [agent_row()])

    response = client.get("/admin/agents")

    assert response.status_code == 200
    agent = response.json()["agents"][0]
    assert "api_key" not in agent
    assert agent["api_key_masked"]
    assert "avatar_data_url" not in agent
    assert "supersecret" not in response.text


def test_agents_list_includes_profile_image_for_dashboard_session(monkeypatch):
    monkeypatch.setattr("app.main.list_agents_with_usage", lambda days=30: [agent_row()])
    monkeypatch.setattr("app.main.session_user", lambda token: {"username": "admin"} if token == "session" else None)

    response = client.get("/admin/agents", headers={"Authorization": "Bearer session"})

    assert response.status_code == 200
    assert response.json()["agents"][0]["avatar_data_url"] == VALID_AVATAR


def test_agent_create_requires_admin_token(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "sekret" else None)

    response = client.post("/admin/agents", json={"name": "athos"})

    assert response.status_code == 401


def test_agent_create_generates_key(monkeypatch):
    created = {}
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr("app.main.create_agent", lambda name, api_key, description="", kind="agent", avatar_data_url=None: created.update(name=name, api_key=api_key, description=description, kind=kind, avatar_data_url=avatar_data_url))

    response = client.post("/admin/agents", json={"name": " athos "})

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "athos"
    assert body["api_key"].startswith("athos_")
    assert created == {"name": "athos", "api_key": body["api_key"], "description": "", "kind": "agent", "avatar_data_url": None}


def test_agent_create_accepts_pregenerated_key(monkeypatch):
    created = {}
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr("app.main.create_agent", lambda name, api_key, description="", kind="agent", avatar_data_url=None: created.update(name=name, api_key=api_key, description=description, kind=kind, avatar_data_url=avatar_data_url))

    response = client.post("/admin/agents", json={"name": "athos", "api_key": "hermes_pregenerated"})

    assert response.status_code == 200
    assert response.json()["api_key"] == "hermes_pregenerated"
    assert created["api_key"] == "hermes_pregenerated"


def test_agent_create_persists_profile_image(monkeypatch):
    created = {}
    avatar = VALID_AVATAR
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr(
        "app.main.create_agent",
        lambda name, api_key, description="", kind="agent", avatar_data_url=None: created.update(
            name=name,
            avatar_data_url=avatar_data_url,
        ),
    )

    response = client.post("/admin/agents", json={"name": "athos", "avatar_data_url": avatar})

    assert response.status_code == 200
    assert created == {"name": "athos", "avatar_data_url": avatar}


def test_agent_create_rejects_invalid_profile_image(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr("app.main.create_agent", lambda *args, **kwargs: None)

    response = client.post("/admin/agents", json={"name": "athos", "avatar_data_url": "https://example.com/avatar.png"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_avatar"


def test_agent_create_rejects_fake_image_data(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr("app.main.create_agent", lambda *args, **kwargs: None)

    response = client.post(
        "/admin/agents",
        json={"name": "athos", "avatar_data_url": "data:image/jpeg;base64,YXZhdGFy"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_avatar"


def test_agent_profile_image_can_be_updated_and_removed(monkeypatch):
    saved = []
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    monkeypatch.setattr("app.main.set_agent_avatar", lambda name, avatar: saved.append((name, avatar)) or True)
    avatar = VALID_AVATAR

    updated = client.put("/admin/agents/athos/avatar", json={"avatar_data_url": avatar})
    removed = client.put("/admin/agents/athos/avatar", json={"avatar_data_url": ""})

    assert updated.status_code == 200
    assert removed.status_code == 200
    assert saved == [("athos", avatar), ("athos", None)]


def test_agent_key_reveal_requires_admin_token(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "sekret" else None)
    monkeypatch.setattr("app.main.get_agent_api_key", lambda name: "hermes_supersecretvalue")

    denied = client.get("/admin/agents/athos/key")
    allowed = client.get("/admin/agents/athos/key", headers={"Authorization": "Bearer sekret"})

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["api_key"] == "hermes_supersecretvalue"


def test_agent_rotate_key_keeps_identity(monkeypatch):
    rotated = {}
    monkeypatch.setattr("app.main.get_agent_api_key", lambda name: "athos_oldkey")
    monkeypatch.setattr("app.main.get_agent_deploy_config", lambda name: None)
    monkeypatch.setattr("app.main.rotate_agent_key", lambda name, api_key: rotated.update(name=name, api_key=api_key) or True)

    response = client.post("/admin/agents/athos/rotate-key")

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"].startswith("athos_")
    assert rotated == {"name": "athos", "api_key": body["api_key"]}
    assert body["deploy"]["status"] == "no_config"


def test_agent_rotate_key_applies_deploy_config(monkeypatch, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("providers:\n  forgerouter:\n    api_key: athos_oldkey\n")
    monkeypatch.setattr("app.main.get_agent_api_key", lambda name: "athos_oldkey")
    monkeypatch.setattr(
        "app.main.get_agent_deploy_config",
        lambda name: {"config_path": str(config_file), "config_format": "yaml", "config_key": "providers.forgerouter.api_key", "restart_service": None},
    )
    monkeypatch.setattr("app.main.rotate_agent_key", lambda name, api_key: True)

    response = client.post("/admin/agents/athos/rotate-key")

    assert response.status_code == 200
    body = response.json()
    assert body["deploy"]["applied"] is True
    assert body["deploy"]["status"] == "written"
    assert body["api_key"] in config_file.read_text()
    assert "athos_oldkey" not in config_file.read_text()


def test_agent_duplicate_copies_controls(monkeypatch):
    duplicated = {}
    monkeypatch.setattr(
        "app.main.duplicate_agent",
        lambda source, new_name, api_key: duplicated.update(source=source, new_name=new_name, api_key=api_key) or True,
    )

    response = client.post("/admin/agents/athos/duplicate", json={"name": "athos-clone"})

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "athos-clone"
    assert body["source"] == "athos"
    assert duplicated["new_name"] == "athos-clone"


def test_agent_set_models(monkeypatch):
    saved = {}
    monkeypatch.setattr("app.main.set_agent_models", lambda name, models: saved.update(name=name, models=models) or True)

    response = client.put("/admin/agents/athos/models", json={"models": ["p2/model-b", " "]})

    assert response.status_code == 200
    assert saved == {"name": "athos", "models": ["p2/model-b"]}


def test_chat_respects_agent_model_controls(monkeypatch):
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "athos")
    monkeypatch.setattr("app.main.agent_allowed_models", lambda name: {"p2/model-b"})
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    # Known and above the 64k virtual-route floor — this test is about agent
    # model-control scoping, not context fit.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)

    calls = []

    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert calls == ["p2/model-b"]


def test_chat_attributes_agent_from_bearer_key(monkeypatch):
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "athos" if key == "hermes_k" else None)
    # Known and above the 64k virtual-route floor — this test is about agent
    # attribution, not context fit.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)

    recorded = {}

    def fake_persist(request_id, model_id, capability, status, error=None, usage=None, agent_name=None, tokens_raw=None, tokens_compacted=None, demand=None, provider_model=None, prompt_preview=None, messages_dropped=None, **kwargs):
        recorded["agent_name"] = agent_name

    monkeypatch.setattr("app.main.persist_route_event", fake_persist)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert recorded["agent_name"] == "athos"


def test_set_aux_tasks_agent_endpoint(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    assigned = []
    monkeypatch.setattr("app.main.set_aux_tasks_agent", lambda name: assigned.append(name) or True)

    denied = client.put("/admin/agents/aux-tasks/aux-tasks")
    allowed = client.put("/admin/agents/aux-tasks/aux-tasks", headers={"Authorization": "Bearer secret"})

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["aux_tasks_agent"] == "aux-tasks"
    assert assigned == ["aux-tasks"]


def test_set_agent_description_endpoint(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    saved = {}
    monkeypatch.setattr("app.main.set_agent_description", lambda name, description: saved.update({name: description}) or True)

    response = client.put(
        "/admin/agents/athos/description",
        headers={"Authorization": "Bearer secret"},
        json={"description": "used by the Hermes auxiliary tasks"},
    )

    assert response.status_code == 200
    assert saved == {"athos": "used by the Hermes auxiliary tasks"}


def test_associate_all_healthy_endpoint(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    called = []
    monkeypatch.setattr(
        "app.main.associate_all_healthy_models_to_all_agents",
        lambda: called.append(True) or {"agents_updated": 3, "models_count": 12, "associations_updated": 36},
    )

    denied = client.post("/admin/agents/associate-all-healthy")
    assert denied.status_code == 401

    allowed = client.post("/admin/agents/associate-all-healthy", headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200
    body = allowed.json()
    assert body["status"] == "ok"
    assert body["agents_updated"] == 3
    assert body["models_count"] == 12
    assert called == [True]
