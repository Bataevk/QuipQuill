"""The prototype limit of a guest and the release waitlist (ADR 66)."""

from decimal import Decimal

from conftest import db_run, ws_connect
from fake_openrouter import FakeOpenRouter, text_stream
from fastapi.testclient import TestClient
from test_auth import bearer, register
from test_guest import open_client
from test_llm_handler import gm_request

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app


def spend(user_id: str, usd: str) -> None:
    db_run(
        "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens, "
        "output_tokens, cache_read_tokens, cost_usd, created_at) "
        "VALUES ($1, 'r', 'gm', 'm', 0, 0, 0, $2, now() - interval '3 days')",
        user_id,
        Decimal(usd),
    )


def model_client(settings: Settings, fake: FakeOpenRouter) -> TestClient:
    """The real LLM handler (it checks the limits) over a fake OpenRouter."""
    live = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "guest_signup": True,
            "guest_total_cost_limit_usd": Decimal("0.10"),
        }
    )
    return TestClient(create_app(live, llm_client=fake))


def last_event(ws) -> dict:
    while True:
        event = ws.receive_json()
        if event["type"] in ("error", "done"):
            return event


def test_a_guest_who_spent_the_prototype_allowance_is_stopped_for_good(settings: Settings) -> None:
    fake = FakeOpenRouter()
    with model_client(settings, fake) as client:
        token = client.post("/auth/guest").json()["access_token"]
        me = client.get("/me", headers=bearer(token)).json()
        assert me["limits"]["total_cost_usd"] == "0.10"
        assert me["limits"]["spent_total_usd"] == "0.000000"

        spend(me["id"], "0.10")  # spent days ago: the daily cap would not stop it
        assert (
            client.get("/me", headers=bearer(token)).json()["limits"]["spent_total_usd"]
            == "0.100000"
        )
        with ws_connect(client, token) as ws:
            ws.send_json(gm_request())
            event = last_event(ws)
        assert event["type"] == "error"
        assert event["code"] == "prototype_limit"
        assert fake.requests == []  # stopped before the model

        embed = client.post(
            "/embeddings",
            json={"model": settings.embed_model, "input": ["x"]},
            headers=bearer(token),
        )
        assert embed.status_code == 429
        assert embed.json()["detail"] == "prototype_limit"


def test_the_prototype_limit_does_not_hold_email_accounts(settings: Settings) -> None:
    fake = FakeOpenRouter()
    fake.chunks = text_stream("Fine.")
    with model_client(settings, fake) as client:
        token = register(client).json()["access_token"]
        me = client.get("/me", headers=bearer(token)).json()
        assert me["limits"]["total_cost_usd"] is None
        spend(me["id"], "0.50")  # past the guests' allowance, days ago (no daily cap hit)
        with ws_connect(client, token) as ws:
            ws.send_json(gm_request())
            assert last_event(ws)["type"] == "done"


def test_the_waitlist_keeps_one_row_per_email(settings: Settings) -> None:
    with open_client(settings) as client:
        token = client.post("/auth/guest").json()["access_token"]
        body = {"email": "Reader@Example.com", "source": "limit"}
        assert client.post("/waitlist", json=body, headers=bearer(token)).status_code == 204
        again = {"email": "reader@example.com", "source": "profile"}
        assert client.post("/waitlist", json=again, headers=bearer(token)).status_code == 204
    rows = db_run("SELECT email, source, user_id IS NOT NULL AS linked FROM waitlist")
    assert [(r["email"], r["source"], r["linked"]) for r in rows] == [
        ("reader@example.com", "limit", True)
    ]


def test_the_waitlist_checks_its_input(client: TestClient) -> None:
    assert client.post("/waitlist", json={"email": "a@b.co", "source": "limit"}).status_code == 401
    token = register(client).json()["access_token"]
    for body in (
        {"email": "not an email", "source": "limit"},
        {"email": "a@b.co", "source": "ads"},
    ):
        assert client.post("/waitlist", json=body, headers=bearer(token)).status_code == 422
