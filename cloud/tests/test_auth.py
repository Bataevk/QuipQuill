from datetime import UTC, datetime, timedelta

from conftest import INVITE, db_run
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

PASSWORD = "correct horse battery"


def register(client: TestClient, email: str = "Player@Example.com", **overrides) -> dict:
    body = {"email": email, "password": PASSWORD, "invite_code": INVITE, **overrides}
    return client.post("/auth/register", json=body)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_register_returns_tokens_and_me_works(client: TestClient) -> None:
    response = register(client)

    assert response.status_code == 201
    tokens = response.json()
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == 900

    me = client.get("/me", headers=bearer(tokens["access_token"]))
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "player@example.com"  # stored lowercased
    assert body["tier"] == "free"
    assert body["usage_today"] == {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": "0.000000",
    }
    # The password is stored as an argon2 hash, never in clear.
    (row,) = db_run("SELECT password_hash FROM users")
    assert row["password_hash"].startswith("$argon2id$")


def test_register_requires_valid_invite(client: TestClient) -> None:
    assert register(client, invite_code="nope").status_code == 403


def test_registration_closed_without_invite_codes(settings: Settings) -> None:
    closed = settings.model_copy(update={"invite_codes": []})
    with TestClient(create_app(closed)) as client:
        assert register(client).status_code == 403


def test_register_rejects_duplicate_email_case_insensitively(client: TestClient) -> None:
    assert register(client, "a@example.com").status_code == 201
    response = register(client, "A@Example.COM")
    assert response.status_code == 409
    assert response.json()["detail"] == "email_taken"


def test_register_validates_input(client: TestClient) -> None:
    assert register(client, password="short").status_code == 422
    assert register(client, email="not-an-email").status_code == 422


def test_login(client: TestClient) -> None:
    register(client, "p@example.com")

    ok = client.post("/auth/login", json={"email": "P@example.com", "password": PASSWORD})
    assert ok.status_code == 200
    assert client.get("/me", headers=bearer(ok.json()["access_token"])).status_code == 200

    wrong = client.post("/auth/login", json={"email": "p@example.com", "password": "wrong-pass"})
    unknown = client.post("/auth/login", json={"email": "x@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"] == "invalid_credentials"


def test_refresh_rotates_tokens(client: TestClient) -> None:
    first = register(client).json()

    second = client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert second.status_code == 200
    second = second.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert client.get("/me", headers=bearer(second["access_token"])).status_code == 200

    third = client.post("/auth/refresh", json={"refresh_token": second["refresh_token"]})
    assert third.status_code == 200


def test_reused_refresh_token_revokes_the_whole_family(client: TestClient) -> None:
    first = register(client).json()
    second = client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]}).json()

    # Someone replays the old (already rotated) token: treat it as stolen.
    replay = client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert replay.status_code == 401
    assert replay.json()["detail"] == "refresh_token_reused"

    # The legitimate newer token is revoked too; the user must log in again.
    after = client.post("/auth/refresh", json={"refresh_token": second["refresh_token"]})
    assert after.status_code == 401
    (row,) = db_run("SELECT count(*) AS active FROM refresh_tokens WHERE revoked_at IS NULL")
    assert row["active"] == 0


def test_expired_and_unknown_refresh_tokens(client: TestClient) -> None:
    tokens = register(client).json()
    db_run("UPDATE refresh_tokens SET expires_at = $1", datetime.now(UTC) - timedelta(seconds=1))

    expired = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    unknown = client.post("/auth/refresh", json={"refresh_token": "made-up"})
    assert expired.json()["detail"] == "refresh_token_expired"
    assert unknown.json()["detail"] == "invalid_refresh_token"
    assert expired.status_code == unknown.status_code == 401


def test_me_rejects_bad_credentials(client: TestClient, settings: Settings) -> None:
    tokens = register(client).json()

    assert client.get("/me").status_code == 401
    assert client.get("/me", headers=bearer("garbage")).status_code == 401
    # A refresh token is not an access token.
    assert client.get("/me", headers=bearer(tokens["refresh_token"])).status_code == 401

    expired = settings.model_copy(update={"access_token_ttl_seconds": -1})
    with TestClient(create_app(expired)) as other:
        stale = other.post(
            "/auth/login", json={"email": "player@example.com", "password": PASSWORD}
        ).json()
        assert other.get("/me", headers=bearer(stale["access_token"])).status_code == 401

    forged = settings.model_copy(
        update={"jwt_secret": SecretStr("another-secret-also-at-least-32-bytes")}
    )
    with TestClient(create_app(forged)) as other:
        foreign = other.post(
            "/auth/login", json={"email": "player@example.com", "password": PASSWORD}
        ).json()
    assert client.get("/me", headers=bearer(foreign["access_token"])).status_code == 401


def test_me_reports_todays_usage_only(client: TestClient) -> None:
    tokens = register(client).json()
    (user,) = db_run("SELECT id FROM users")
    insert = (
        "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens,"
        " output_tokens, cache_read_tokens, cost_usd, created_at)"
        " VALUES ($1, $2, 'gm', 'anthropic/claude-haiku-5.5', $3, $4, 0, $5, $6)"
    )
    now = datetime.now(UTC)
    db_run(insert, user["id"], "r1", 1000, 200, 0.0002, now)
    db_run(insert, user["id"], "r2", 500, 100, 0.0001, now)
    db_run(insert, user["id"], "old", 9999, 9999, 1, now - timedelta(days=2))

    usage = client.get("/me", headers=bearer(tokens["access_token"])).json()["usage_today"]
    assert usage == {
        "requests": 2,
        "input_tokens": 1500,
        "output_tokens": 300,
        "cost_usd": "0.000300",
    }
