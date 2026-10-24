"""OpenRouter worker over the real queue and WebSocket, with a scripted model."""

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx2
import openai
import pytest
from conftest import INVITE, db_run, ws_connect
from fake_openrouter import USAGE, FakeOpenRouter, chunk, text_stream, tool_stream
from fastapi.testclient import TestClient

from quipquill_cloud.core.config import Settings
from quipquill_cloud.llm import usage
from quipquill_cloud.llm.policy import PURPOSES
from quipquill_cloud.main import create_app

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move to an adjacent location.",
            "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
        },
    }
]


@pytest.fixture
def fake() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
def app_client(settings: Settings, fake: FakeOpenRouter) -> Iterator[TestClient]:
    live = settings.model_copy(update={"llm_handler": "openrouter", "llm_timeout_seconds": 2.0})
    with TestClient(create_app(live, llm_client=fake)) as client:
        yield client


def token_for(client: TestClient) -> str:
    response = client.post(
        "/auth/register",
        json={"email": "gm@example.com", "password": "long enough pass", "invite_code": INVITE},
    )
    return response.json()["access_token"]


def gm_request(purpose: str = "gm", request_id: str = "req-1", tools=None) -> dict:
    return {
        "v": 1,
        "type": "llm.request",
        "requestId": request_id,
        "sessionId": "save-1",
        "channel": "gm",
        "purpose": purpose,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Rules.", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": "I walk north."},
        ],
        **({"tools": tools} if tools else {}),
    }


def run(client: TestClient, request: dict) -> list[dict]:
    with ws_connect(client, token_for(client)) as ws:
        ws.send_json(request)
        events = []
        while not events or events[-1]["type"] not in ("done", "error"):
            event = ws.receive_json()
            if event["type"] != "heartbeat":
                events.append(event)
        return events


def test_text_stream_and_usage(app_client: TestClient, fake: FakeOpenRouter) -> None:
    events = run(app_client, gm_request())

    assert [e["type"] for e in events] == ["status", "delta", "delta", "done"]
    assert [e["seq"] for e in events[1:3]] == [0, 1]
    done = events[-1]
    assert done["message"] == {"role": "assistant", "content": "Hello there"}
    assert done["finishReason"] == "stop"
    assert done["usage"] == {
        "input": 1200,
        "output": 80,
        "cacheRead": 1000,
        "cacheWrite": 0,
        "costUsd": 0.000123,
    }
    (row,) = db_run(
        "SELECT purpose, model, input_tokens, cache_read_tokens, cost_usd FROM usage_events"
    )
    assert row["purpose"] == "gm"
    assert row["model"] == "anthropic/claude-haiku-5.5"
    assert (row["input_tokens"], row["cache_read_tokens"]) == (1200, 1000)
    assert row["cost_usd"] == Decimal("0.000123")


def test_server_chooses_model_params_and_providers(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    run(app_client, gm_request(tools=TOOLS))
    (body,) = fake.requests

    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["max_tokens"] == PURPOSES["gm"].max_tokens
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"] == TOOLS
    assert body["extra_body"]["reasoning"] == {"effort": "high"}  # gm thinks at high (ADR 74)
    # Route "basic" (the default): Vertex global, then Bedrock (ADR 51).
    providers = ["google-vertex/global", "amazon-bedrock"]
    assert body["extra_body"]["provider"] == {
        "order": providers,
        "only": providers,
        "allow_fallbacks": False,
    }
    # The client's cache breakpoint is passed through untouched.
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_config_overrides_model_and_lifts_providers(
    settings: Settings, fake: FakeOpenRouter
) -> None:
    live = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "llm_model": "inception/mercury-3",
            "provider_order": [],
        }
    )
    with TestClient(create_app(live, llm_client=fake)) as client:
        run(client, gm_request())
    (body,) = fake.requests

    assert body["model"] == "inception/mercury-3"
    assert "provider" not in body["extra_body"]  # [] = OpenRouter picks the provider


def test_spare_model_answers_when_the_main_one_is_down(
    settings: Settings, fake: FakeOpenRouter
) -> None:
    live = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "llm_model": "inception/mercury-3",
            "provider_order": [],
            "llm_fallback_model": "anthropic/claude-haiku-5.5",
        }
    )
    fake.down_models = {"inception/mercury-3"}
    with TestClient(create_app(live, llm_client=fake)) as client:
        events = run(client, gm_request())

    assert [r["model"] for r in fake.requests] == [
        "inception/mercury-3",
        "anthropic/claude-haiku-5.5",
    ]
    # The spare keeps the route's own providers from the code (ADR 51).
    assert fake.requests[1]["extra_body"]["provider"]["only"] == [
        "google-vertex/global",
        "amazon-bedrock",
    ]
    assert events[-1]["type"] == "done"
    assert db_run("SELECT model FROM usage_events")[0]["model"] == "anthropic/claude-haiku-5.5"


def test_config_sets_effort_for_main_and_spare(settings: Settings, fake: FakeOpenRouter) -> None:
    live = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "llm_model": "inception/mercury-3",
            "llm_effort": {"gm": "medium"},
            "llm_fallback_model": "anthropic/claude-haiku-5.5",
            "llm_fallback_effort": {"gm": "low"},
        }
    )
    fake.down_models = {"inception/mercury-3"}
    with TestClient(create_app(live, llm_client=fake)) as client:
        run(client, gm_request())
    main, spare = fake.requests

    assert main["extra_body"]["reasoning"] == {"effort": "medium"}
    assert spare["extra_body"]["reasoning"] == {"effort": "low"}


def test_without_a_spare_a_failure_is_an_error(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    fake.down_models = {"anthropic/claude-haiku-5.5"}
    events = run(app_client, gm_request())

    assert len(fake.requests) == 1
    assert (events[-1]["type"], events[-1]["code"]) == ("error", "provider_error")


def test_route_is_named_by_the_client_and_chosen_by_the_server(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    events = run(app_client, {**gm_request(), "model": "anthropic/claude-opus"})
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_request"
    assert fake.requests == []  # an unknown route never reaches the provider


def test_npc_runs_at_medium_effort(app_client: TestClient, fake: FakeOpenRouter) -> None:
    run(app_client, gm_request(purpose="npc"))
    (body,) = fake.requests
    assert body["extra_body"]["reasoning"] == {"effort": "medium"}
    assert body["max_tokens"] == PURPOSES["npc"].max_tokens


def test_generator_runs_at_medium_effort(app_client: TestClient, fake: FakeOpenRouter) -> None:
    run(app_client, gm_request(purpose="generator"))
    (body,) = fake.requests
    assert body["extra_body"]["reasoning"] == {"effort": "medium"}
    assert body["max_tokens"] == PURPOSES["generator"].max_tokens


def test_tool_calls_are_assembled(app_client: TestClient, fake: FakeOpenRouter) -> None:
    fake.chunks = tool_stream()
    events = run(app_client, gm_request(tools=TOOLS))

    types = [e["type"] for e in events]
    assert types == ["status", "delta", "tool_call", "tool_call", "done"]
    first, second = events[2]["call"], events[3]["call"]
    assert first == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "move", "arguments": '{"to": "Torchlight Corridor"}'},
    }
    assert second["function"]["name"] == "inspect"
    done = events[-1]
    assert done["finishReason"] == "tool_calls"
    assert done["message"]["content"] == "Heading north."
    assert done["message"]["tool_calls"] == [first, second]


def test_daily_limit_blocks_before_calling_the_model(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    token_for(app_client)  # create the user
    (user,) = db_run("SELECT id FROM users")
    db_run(
        "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens,"
        " output_tokens, cache_read_tokens, cost_usd, created_at)"
        " VALUES ($1, 'old', 'gm', 'm', 0, 0, 0, 0.05, $2)",
        user["id"],
        datetime.now(UTC),
    )
    login = app_client.post(
        "/auth/login", json={"email": "gm@example.com", "password": "long enough pass"}
    ).json()
    with ws_connect(app_client, login["access_token"]) as ws:
        ws.send_json(gm_request())
        event = ws.receive_json()

    assert event["type"] == "error"
    assert event["code"] == "limit_exceeded"
    assert event["retryable"] is False
    assert fake.requests == []


def key_error(code: int) -> openai.APIStatusError:
    """OpenRouter refusing the shared key: 402 — no credits left, 401 — the key is not valid."""
    response = httpx2.Response(code, request=httpx2.Request("POST", "http://x"))
    kind = openai.AuthenticationError if code == 401 else openai.APIStatusError
    return kind("key", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (
            openai.BadRequestError(
                "bad",
                response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")),
                body=None,
            ),
            "bad_request",
            False,
        ),
        (
            openai.APIConnectionError(request=httpx2.Request("POST", "http://x")),
            "provider_error",
            True,
        ),
        (openai.APITimeoutError(request=httpx2.Request("POST", "http://x")), "timeout", True),
        (key_error(402), "budget_exhausted", False),
        (key_error(401), "budget_exhausted", False),
    ],
)
def test_provider_errors_become_error_events(
    app_client: TestClient, fake: FakeOpenRouter, error: Exception, code: str, retryable: bool
) -> None:
    fake.error = error
    events = run(app_client, gm_request())
    assert [e["type"] for e in events] == ["status", "error"]
    assert (events[-1]["code"], events[-1]["retryable"]) == (code, retryable)
    assert db_run("SELECT count(*) AS n FROM usage_events")[0]["n"] == 0


def test_done_arrives_even_if_usage_cannot_be_recorded(
    app_client: TestClient, fake: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_record(*args, **kwargs) -> None:
        raise ConnectionError("database is down")

    monkeypatch.setattr(usage, "record", broken_record)
    fake.chunks = tool_stream()
    events = run(app_client, gm_request(tools=TOOLS))

    assert [e["type"] for e in events] == ["status", "delta", "tool_call", "tool_call", "done"]
    assert events[-1]["usage"]["costUsd"] > 0


def test_slow_stream_times_out(app_client: TestClient, fake: FakeOpenRouter) -> None:
    fake.chunks = text_stream(*["word "] * 10)
    fake.delay = 0.5  # 10 chunks × 0.5 s > 2 s timeout
    events = run(app_client, gm_request())
    assert events[-1]["code"] == "timeout"


def test_cancel_stops_the_stream(app_client: TestClient, fake: FakeOpenRouter) -> None:
    fake.chunks = text_stream(*[f"w{i} " for i in range(30)])
    fake.delay = 0.05
    with ws_connect(app_client, token_for(app_client)) as ws:
        ws.send_json(gm_request())
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["type"] == "delta"
        ws.send_json({"v": 1, "type": "cancel", "requestId": "req-1"})
        while (event := ws.receive_json())["type"] != "done":
            pass
    assert event["finishReason"] == "cancelled"
    assert len(event["message"]["content"].split()) < 30


def test_embeddings_endpoint(app_client: TestClient, fake: FakeOpenRouter) -> None:
    headers = {"Authorization": f"Bearer {token_for(app_client)}"}
    body = {"model": "openai/text-embedding-3-small", "input": ["abc", "brass vault"]}

    response = app_client.post("/embeddings", json=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["vectors"] == [[3.0, 0.5], [11.0, 0.5]]  # back in input order
    (row,) = db_run("SELECT purpose, model, input_tokens FROM usage_events")
    assert (row["purpose"], row["input_tokens"]) == ("embed", 7)

    wrong = app_client.post("/embeddings", json={**body, "model": "other/model"}, headers=headers)
    assert wrong.status_code == 400
    assert app_client.post("/embeddings", json=body).status_code == 401
    too_many = app_client.post("/embeddings", json={**body, "input": ["x"] * 65}, headers=headers)
    assert too_many.status_code == 422


def test_embeddings_respect_the_daily_limit(app_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {token_for(app_client)}"}
    (user,) = db_run("SELECT id FROM users")
    db_run(
        "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens,"
        " output_tokens, cache_read_tokens, cost_usd) VALUES ($1, 'x', 'gm', 'm', 0, 0, 0, 1)",
        user["id"],
    )
    body = {"model": "openai/text-embedding-3-small", "input": ["abc"]}
    assert app_client.post("/embeddings", json=body, headers=headers).status_code == 429


def test_daily_limit_can_be_raised_for_everyone(settings: Settings) -> None:
    fake = FakeOpenRouter()
    raised = settings.model_copy(update={"daily_cost_limit_usd": Decimal("5")})
    with TestClient(create_app(raised, llm_client=fake)) as client:
        headers = {"Authorization": f"Bearer {token_for(client)}"}
        (user,) = db_run("SELECT id FROM users")
        db_run(
            "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens,"
            " output_tokens, cache_read_tokens, cost_usd) VALUES ($1, 'x', 'gm', 'm', 0, 0, 0, 1)",
            user["id"],
        )
        body = {"model": "openai/text-embedding-3-small", "input": ["abc"]}
        assert client.post("/embeddings", json=body, headers=headers).status_code == 200
        assert client.get("/me", headers=headers).json()["limits"]["daily_cost_usd"] == "5"


def test_me_reports_limits(app_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {token_for(app_client)}"}
    assert app_client.get("/me", headers=headers).json()["limits"]["daily_cost_usd"] == "0.05"


# --- prompts filled in by the cloud (ADR 57) ---


def test_the_worker_sends_the_filled_in_prompt(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    request = gm_request()
    request["messages"][0]["content"] = [
        {
            "type": "prompt",
            "name": "summary",
            "vars": {},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    run(app_client, request)
    (body,) = fake.requests
    (part,) = body["messages"][0]["content"]
    assert part["text"].startswith("You keep the Game Master's memory of an interactive story.")
    assert part["cache_control"] == {"type": "ephemeral"}


def test_a_bad_prompt_is_a_bad_request(app_client: TestClient, fake: FakeOpenRouter) -> None:
    request = gm_request()
    request["messages"][0]["content"] = [{"type": "prompt", "name": "gm", "vars": {}}]
    events = run(app_client, request)
    assert [e["type"] for e in events] == ["status", "error"]
    assert events[-1]["code"] == "bad_request"
    assert fake.requests == []


def test_health_reports_missing_prompts(settings: Settings, tmp_path: Path) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/prompts").json()["status"] == "ok"
    (tmp_path / "gm.md").write_text("Rules.")
    with TestClient(create_app(settings.model_copy(update={"prompts_dir": tmp_path}))) as client:
        response = client.get("/health/prompts")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("missing: actor, guard")


def guard_request(user_text: str = "I notice a stick in my pocket.") -> dict:
    request = gm_request(purpose="guard")
    request["channel"] = "system"
    request["messages"] = [
        {
            "role": "system",
            "content": [{"type": "prompt", "name": "guard", "vars": {"WORLD_RULES": "No magic."}}],
        },
        {"role": "user", "content": [{"type": "text", "text": f"# Player input\n{user_text}"}]},
    ]
    return request


def test_guard_is_a_decision_model_call(app_client: TestClient, fake: FakeOpenRouter) -> None:
    """ADR 70: the Decisions API, not chat; the template's questions; answers as JSON in done."""
    events = run(app_client, guard_request())

    assert [e["type"] for e in events] == ["status", "done"]
    (body,) = fake.requests
    assert body["path"] == "https://openrouter.ai/api/alpha/decisions"
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"].startswith("You check one player input of an interactive story")
    assert "# World rules\nNo magic." in body["state"]
    assert body["state"].endswith("# Player input\nI notice a stick in my pocket.")
    assert "## decision" not in body["state"]
    assert list(body["questions"]) == ["decision", "warning_type", "threat_level"]
    decision = body["questions"]["decision"]
    assert decision["type"] == "choice"
    assert list(decision["criteria"]) == ["ALLOW", "PASS_TO_GM", "REJECT"]
    assert decision["instructions"] == "Where should this player input go?"

    done = events[-1]
    answers = json.loads(done["message"]["content"])
    assert answers["decision"]["choice"] == "PASS_TO_GM"
    assert done["usage"] == {
        "input": 800,
        "output": 120,
        "cacheRead": 0,
        "cacheWrite": 0,
        "costUsd": 0.0000336,
    }
    (row,) = db_run("SELECT purpose, model, input_tokens, cost_usd FROM usage_events")
    assert (row["purpose"], row["model"], row["input_tokens"]) == (
        "guard",
        "typesafe/jev-1.14",
        800,
    )
    assert row["cost_usd"] == Decimal("0.000034")  # the column keeps micro-dollars


@pytest.mark.parametrize(
    ("reply", "code"),
    [({"error": "nothing"}, "provider_error"), ({"answers": "yes"}, "provider_error")],
)
def test_a_decision_without_answers_is_a_provider_error(
    app_client: TestClient, fake: FakeOpenRouter, reply: dict, code: str
) -> None:
    fake.decision = reply
    events = run(app_client, guard_request())
    assert [e["type"] for e in events] == ["status", "error"]
    assert events[-1]["code"] == code
    assert db_run("SELECT count(*) AS n FROM usage_events")[0]["n"] == 0


def test_a_failed_decision_call_is_an_error_event(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    request = httpx2.Request("POST", "https://openrouter.ai/api/alpha/decisions")
    fake.decision_error = openai.APIConnectionError(request=request)
    events = run(app_client, guard_request())
    assert [e["type"] for e in events] == ["status", "error"]
    assert (events[-1]["code"], events[-1]["retryable"]) == ("provider_error", True)


def test_a_decision_template_without_questions_is_a_bad_request(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    request = guard_request()
    request["messages"][0]["content"] = [{"type": "text", "text": "Just text, no questions."}]
    events = run(app_client, request)
    assert [e["type"] for e in events] == ["status", "error"]
    assert events[-1]["code"] == "bad_request"
    assert fake.requests == []


def test_the_items_of_a_request_become_questions(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    """Stage T.7: the sheet's sentences, one `noul` question each."""
    request = guard_request("# The sheet\nName: Kai. A bard.")
    request["messages"][0]["content"] = [{"type": "prompt", "name": "guard_sheet", "vars": {}}]
    request["items"] = ["Name: Kai.", "A bard."]
    events = run(app_client, request)
    assert events[-1]["type"] == "done"
    (body,) = fake.requests
    assert list(body["questions"]) == ["decision", "reason", "sentence_0", "sentence_1"]
    assert body["questions"]["sentence_1"]["instructions"].endswith("\nText: A bard.")


def test_a_spent_key_skips_the_spare_model(settings: Settings, fake: FakeOpenRouter) -> None:
    live = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "llm_model": "inception/mercury-3",
            "llm_fallback_model": "anthropic/claude-haiku-5.5",
        }
    )
    fake.error = key_error(402)
    with TestClient(create_app(live, llm_client=fake)) as client:
        events = run(client, gm_request())
    assert len(fake.requests) == 1  # the spare goes through the same key
    assert (events[-1]["code"], events[-1]["retryable"]) == ("budget_exhausted", False)


def test_a_spent_key_stops_decisions_and_embeddings(
    app_client: TestClient, fake: FakeOpenRouter
) -> None:
    fake.error = key_error(402)
    headers = {"Authorization": f"Bearer {token_for(app_client)}"}
    body = {"model": "openai/text-embedding-3-small", "input": ["abc"]}
    response = app_client.post("/embeddings", json=body, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "budget_exhausted"

    fake.decision_error = key_error(402)
    with ws_connect(app_client, headers["Authorization"].split()[1]) as ws:
        ws.send_json(guard_request())
        while (event := ws.receive_json())["type"] != "error":
            pass
    assert (event["code"], event["retryable"]) == ("budget_exhausted", False)


def test_an_empty_answer_is_logged_with_its_finish_reason(
    app_client: TestClient, fake: FakeOpenRouter, caplog: pytest.LogCaptureFixture
) -> None:
    usage = {**USAGE, "completion_tokens_details": {"reasoning_tokens": 80}}
    fake.chunks = [chunk("", finish="length", usage=usage)]
    with caplog.at_level(logging.INFO, logger="quipquill_cloud.llm.handler"):
        events = run(app_client, gm_request())
    assert events[-1]["type"] == "done"
    text = caplog.text
    assert "reasoning=80 finish=length text=0 tools=0" in text
    assert "answered req-1 with nothing (finish=length)" in text
