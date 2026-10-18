"""Abuse limits, request ids, dependency health checks, JSON logs."""

import json
import logging
import time
from collections.abc import Iterator

import pytest
from conftest import INVITE, ws_connect
from fake_openrouter import FakeOpenRouter
from fastapi.testclient import TestClient

from quipquill_cloud import main
from quipquill_cloud.core import logs
from quipquill_cloud.core.config import Settings
from quipquill_cloud.core.ratelimit import RateLimiter
from quipquill_cloud.main import create_app


def make_client(settings: Settings, fake: FakeOpenRouter | None = None, **overrides):
    return TestClient(create_app(settings.model_copy(update=overrides), llm_client=fake))


def token_for(client: TestClient) -> str:
    response = client.post(
        "/auth/register",
        json={"email": "h@example.com", "password": "long enough pass", "invite_code": INVITE},
    )
    return response.json()["access_token"]


def llm_request(request_id: str, text: str = "hi", tools: list | None = None) -> dict:
    return {
        "v": 1,
        "type": "llm.request",
        "requestId": request_id,
        "sessionId": "s",
        "channel": "gm",
        "purpose": "gm",
        "messages": [{"role": "user", "content": text}],
        **({"tools": tools} if tools else {}),
    }


def first_terminal(ws) -> dict:
    while (event := ws.receive_json())["type"] not in ("done", "error"):
        pass
    return event


# --- rate limiting ------------------------------------------------------------------------------


def test_token_bucket() -> None:
    now = [0.0]
    limiter = RateLimiter(per_minute=30, burst=3, clock=lambda: now[0])
    assert [limiter.allow("u") for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("other")  # buckets are per user
    now[0] = 2.0  # 30/min = one token every 2 s
    assert limiter.allow("u") and not limiter.allow("u")


def test_websocket_requests_are_rate_limited(settings: Settings) -> None:
    with (
        make_client(settings, rate_limit_burst=2, rate_limit_per_minute=0.001) as client,
        ws_connect(client, token_for(client)) as ws,
    ):
        for rid in ("a", "b"):
            ws.send_json(llm_request(rid))
            assert first_terminal(ws)["type"] == "done"
        ws.send_json(llm_request("c"))
        refused = ws.receive_json()
    assert refused["type"] == "error"
    assert (refused["code"], refused["retryable"], refused["requestId"]) == (
        "rate_limited",
        True,
        "c",
    )


def test_embeddings_are_rate_limited(settings: Settings) -> None:
    fake = FakeOpenRouter()
    with make_client(settings, fake, rate_limit_burst=1, rate_limit_per_minute=0.001) as client:
        headers = {"Authorization": f"Bearer {token_for(client)}"}
        body = {"model": "openai/text-embedding-3-small", "input": ["x"]}
        assert client.post("/embeddings", json=body, headers=headers).status_code == 200
        second = client.post("/embeddings", json=body, headers=headers)
    assert (second.status_code, second.json()["detail"]) == (429, "rate_limited")


# --- size limits ---------------------------------------------------------------------------------


def test_oversized_llm_request_is_refused(settings: Settings) -> None:
    tool = {"type": "function", "function": {"name": "t", "parameters": {}}}
    with (
        make_client(settings, max_request_tokens=50, max_tools=1) as client,
        ws_connect(client, token_for(client)) as ws,
    ):
        ws.send_json(llm_request("big", text="word " * 200))
        too_large = ws.receive_json()
        ws.send_json(llm_request("tools", tools=[tool, tool]))
        too_many = ws.receive_json()
    assert (too_large["code"], too_large["retryable"]) == ("request_too_large", False)
    assert too_many["code"] == "too_many_tools"


def test_http_body_size_limit_and_request_id(settings: Settings) -> None:
    with make_client(settings, max_http_body_bytes=200) as client:
        big = client.post("/auth/login", json={"email": "a@example.com", "password": "x" * 500})
        echoed = client.get("/health", headers={"X-Request-ID": "trace-123"})
        generated = client.get("/health")
    assert (big.status_code, big.json()["detail"]) == (413, "request_too_large")
    assert echoed.headers["X-Request-ID"] == "trace-123"
    assert len(generated.headers["X-Request-ID"]) == 32


# --- health --------------------------------------------------------------------------------------


@pytest.fixture
def fake() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
def healthy(settings: Settings, fake: FakeOpenRouter) -> Iterator[TestClient]:
    with make_client(settings, fake) as client:
        yield client


def test_all_dependencies_healthy(healthy: TestClient) -> None:
    for path in ("/health/db", "/health/mq"):
        response = healthy.get(path)
        assert (response.status_code, response.json()["status"]) == (200, "ok"), path
    llm = healthy.get("/health/llm")
    assert llm.status_code == 200
    assert llm.json()["info"] == {"limitRemainingUsd": 4.9, "limitUsd": 5}


def test_llm_low_credit_is_degraded_and_failure_is_down(
    healthy: TestClient, fake: FakeOpenRouter
) -> None:
    fake.key_info = {"data": {"limit": 5, "limit_remaining": 0.2}}
    low = healthy.get("/health/llm")
    assert (low.status_code, low.json()["status"], low.json()["detail"]) == (
        200,
        "degraded",
        "low_credit",
    )
    fake.key_error = ConnectionError("no route")
    down = healthy.get("/health/llm")
    assert (down.status_code, down.json()["status"]) == (503, "down")


def test_database_down(settings: Settings) -> None:
    broken = "postgresql+asyncpg://nobody:nothing@localhost:1/none"
    with make_client(settings, database_url=broken) as client:
        response = client.get("/health/db")
    assert (response.status_code, response.json()["status"]) == (503, "down")


def test_queue_down(settings: Settings) -> None:
    broken = "amqp://nobody:nothing@localhost:1/"
    with make_client(settings, rabbitmq_url=broken, mq_connect_timeout=1) as client:
        response = client.get("/health/mq")
    assert response.status_code == 503
    assert response.json()["detail"] == "connecting"


def test_queue_comes_up_after_the_cloud(settings: Settings, monkeypatch) -> None:
    """After a reboot the cloud may start before RabbitMQ: it keeps connecting in the background."""
    real = main.Broker.connect
    failures = iter([OSError("connection refused"), OSError("connection refused")])

    async def flaky(cls, *args):
        error = next(failures, None)
        if error is not None:
            raise error
        return await real(*args)

    monkeypatch.setattr(main.Broker, "connect", classmethod(flaky))
    with make_client(settings, mq_retry_seconds=0.05) as client:
        assert client.get("/health/mq").json()["detail"] == "connecting"
        deadline = time.monotonic() + 10
        while client.get("/health/mq").status_code != 200:
            assert time.monotonic() < deadline, "the cloud never connected to RabbitMQ"
            time.sleep(0.1)
        assert client.get("/health/mq").json()["status"] == "ok"


# --- logs ----------------------------------------------------------------------------------------


def test_json_log_lines_carry_request_context() -> None:
    record = logging.LogRecord(
        "quipquill.test", logging.ERROR, __file__, 1, "boom %s", ("x",), None
    )
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()
    token = logs.request_id.set("req-42")
    try:
        line = json.loads(logs.JsonFormatter().format(record))
    finally:
        logs.request_id.reset(token)
    assert line["msg"] == "boom x"
    assert line["level"] == "error"
    assert line["requestId"] == "req-42"
    assert "ValueError: bad" in line["exc"]
