"""Guest accounts of the device and the terms they accept (ADR 60)."""

from conftest import db_run
from fastapi.testclient import TestClient
from test_auth import PASSWORD, bearer, register

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app


def open_client(settings: Settings, **overrides) -> TestClient:
    return TestClient(create_app(settings.model_copy(update={"guest_signup": True, **overrides})))


def test_guest_signup_is_closed_by_default(client: TestClient) -> None:
    response = client.post("/auth/guest")
    assert response.status_code == 403
    assert response.json()["detail"] == "guest_signup_closed"


def test_a_guest_gets_tokens_and_a_free_account_without_email(settings: Settings) -> None:
    with open_client(settings) as client:
        response = client.post("/auth/guest")
        assert response.status_code == 201
        tokens = response.json()

        me = client.get("/me", headers=bearer(tokens["access_token"])).json()
        assert me["email"] is None
        assert me["kind"] == "guest"
        assert me["tier"] == "free"
        assert me["terms_version"] is None

        # A guest's refresh token rotates like anyone's.
        refreshed = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert refreshed.status_code == 200


def test_every_guest_is_a_new_account(settings: Settings) -> None:
    with open_client(settings) as client:
        first = client.post("/auth/guest").json()["access_token"]
        second = client.post("/auth/guest").json()["access_token"]
        ids = {client.get("/me", headers=bearer(t)).json()["id"] for t in (first, second)}
    assert len(ids) == 2
    assert db_run("SELECT count(*) AS n FROM users WHERE kind = 'guest'")[0]["n"] == 2


def test_guests_are_limited_per_client_ip(settings: Settings) -> None:
    with open_client(settings, guest_signup_burst=2, guest_signup_per_minute=0.001) as client:
        codes = [client.post("/auth/guest").status_code for _ in range(3)]
    assert codes == [201, 201, 429]


def test_email_accounts_live_beside_guests(settings: Settings) -> None:
    with open_client(settings) as client:
        client.post("/auth/guest")
        # A guest has no email: it does not take one, and email accounts still sign in.
        assert register(client).status_code == 201
        login = client.post(
            "/auth/login", json={"email": "player@example.com", "password": PASSWORD}
        )
        assert login.status_code == 200


def test_consent_is_recorded_and_a_new_version_replaces_it(settings: Settings) -> None:
    with open_client(settings) as client:
        token = client.post("/auth/guest").json()["access_token"]

        first = client.post("/me/consent", json={"terms_version": "0-draft"}, headers=bearer(token))
        assert first.status_code == 200
        assert first.json()["terms_version"] == "0-draft"
        me = client.get("/me", headers=bearer(token)).json()
        assert me["terms_version"] == "0-draft"
        assert me["terms_accepted_at"] is not None

        client.post("/me/consent", json={"terms_version": "1"}, headers=bearer(token))
        assert client.get("/me", headers=bearer(token)).json()["terms_version"] == "1"


def test_consent_needs_a_signed_in_player_and_a_version(client: TestClient) -> None:
    assert client.post("/me/consent", json={"terms_version": "1"}).status_code == 401
    token = register(client).json()["access_token"]
    assert (
        client.post("/me/consent", json={"terms_version": ""}, headers=bearer(token)).status_code
        == 422
    )
    assert (
        client.post(
            "/me/consent", json={"terms_version": "x" * 33}, headers=bearer(token)
        ).status_code
        == 422
    )
