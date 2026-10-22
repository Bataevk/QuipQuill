"""CORS for the web version (stage W.4, ADR 69): local dev servers and listed origins only."""

from fastapi.testclient import TestClient

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

PROD = {"env": "prod", "jwt_secret": "x" * 32, "openrouter_api_key": "sk-or-test"}


def preflight(client: TestClient, origin: str) -> dict[str, str]:
    response = client.options(
        "/auth/guest",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    return dict(response.headers)


def test_dev_allows_any_localhost_port() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    for origin in (
        "http://localhost:8081",
        "http://localhost:8090",
        "http://127.0.0.1:3000",
        "http://[::1]:8090",
    ):
        headers = preflight(client, origin)
        assert headers.get("access-control-allow-origin") == origin
        assert "authorization" in headers.get("access-control-allow-headers", "").lower()
    assert "access-control-allow-origin" not in preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in preflight(client, "http://localhost.evil.example")


def test_simple_request_gets_the_origin_and_request_id_exposed() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/health", headers={"Origin": "http://localhost:8090"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:8090"
    assert "x-request-id" in response.headers["access-control-expose-headers"].lower()


def test_prod_allows_only_listed_origins() -> None:
    same_origin = TestClient(create_app(Settings(_env_file=None, **PROD)))
    assert "access-control-allow-origin" not in preflight(same_origin, "http://localhost:8090")

    listed = TestClient(
        create_app(Settings(_env_file=None, cors_origins=["https://play.example"], **PROD))
    )
    assert preflight(listed, "https://play.example")["access-control-allow-origin"] == (
        "https://play.example"
    )
    assert "access-control-allow-origin" not in preflight(listed, "http://localhost:8090")
