"""Migration keeps existing browser ownership and CSRF identity."""
from test_prediction import system  # noqa: F401


def test_legacy_cookie_is_promoted_without_changing_owner(system):
    client, manager, policy = system
    session = client.cookies.get("forcevla_session")
    csrf = client.headers["X-CSRF-Token"]
    client.cookies.clear()
    client.cookies.set("shadow_session", session)
    assert client.get("/api/v1/status").status_code == 200
    response = client.get("/api/v1/session")
    assert response.json()["csrf_token"] == csrf
    assert client.cookies.get("forcevla_session") == session
    client.cookies.clear()
    client.cookies.set("shadow_session", session + "invalid")
    assert client.get("/api/v1/status").status_code == 401
