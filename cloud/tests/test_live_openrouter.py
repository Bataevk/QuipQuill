"""Live OpenRouter checks (cost a fraction of a cent). Run: make cloud-test-live."""

import os
from collections.abc import Iterator

import pytest
from conftest import INVITE, db_run
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
    ),
]

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Move the player to an adjacent location by name.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string", "description": "Location name"}},
            "required": ["to"],
        },
    },
}
# Long enough to be cacheable (Haiku 5.5 caches prefixes from 512 tokens).
RULES = " ".join(
    f"Rule {i}: you are the game master; every change of the world goes through a tool call."
    for i in range(80)
)


@pytest.fixture
def live(settings: Settings) -> Iterator[TestClient]:
    real = settings.model_copy(
        update={
            "llm_handler": "openrouter",
            "openrouter_api_key": SecretStr(os.environ["OPENROUTER_API_KEY"]),
        }
    )
    with TestClient(create_app(real)) as client:
        yield client


def connect(client: TestClient):
    token = client.post(
        "/auth/register",
        json={"email": "live@example.com", "password": "long enough pass", "invite_code": INVITE},
    ).json()["access_token"]
    return client.websocket_connect(f"/ws?token={token}"), token


def ask(ws, request_id: str, purpose: str, messages: list, tools=None) -> list[dict]:
    ws.send_json(
        {
            "v": 1,
            "type": "llm.request",
            "requestId": request_id,
            "sessionId": "live-1",
            "channel": "gm",
            "purpose": purpose,
            "messages": messages,
            **({"tools": tools} if tools else {}),
        }
    )
    events = []
    while not events or events[-1]["type"] not in ("done", "error"):
        event = ws.receive_json()
        if event["type"] != "heartbeat":
            events.append(event)
    return events


def test_streaming_tool_calls_usage_cache_and_limit(live: TestClient) -> None:
    socket, token = connect(live)
    system = {
        "role": "system",
        "content": [{"type": "text", "text": RULES, "cache_control": {"type": "ephemeral"}}],
    }
    with socket as ws:
        # 1. Text streams as deltas.
        talk = ask(ws, "t1", "npc", [{"role": "user", "content": "Greet a traveller in one line."}])
        assert talk[-1]["type"] == "done", talk[-1]
        assert any(e["type"] == "delta" for e in talk)
        assert talk[-1]["message"]["content"]

        # 2. A tool call arrives whole; usage and cost are reported.
        moves = []
        for i in range(3):
            events = ask(
                ws,
                f"m{i}",
                "gm",
                [system, {"role": "user", "content": "I go to the Torchlight Corridor."}],
                tools=[MOVE_TOOL],
            )
            moves.append(events)
        done = moves[0][-1]
        assert done["type"] == "done", done
        assert done["finishReason"] == "tool_calls"
        call = next(e["call"] for e in moves[0] if e["type"] == "tool_call")
        assert call["function"]["name"] == "move"
        assert "corridor" in call["function"]["arguments"].lower()
        assert done["usage"]["costUsd"] > 0

        # 3. The cached system prompt is read back on a later call (Bedrock first, ADR 20).
        assert any(m[-1]["usage"]["cacheRead"] > 0 for m in moves[1:]), [
            m[-1]["usage"] for m in moves
        ]

    rows = db_run("SELECT count(*) AS n, sum(cost_usd) AS cost FROM usage_events")[0]
    assert rows["n"] == 4 and rows["cost"] > 0

    # 4. Over the daily limit the model is not called at all.
    (user,) = db_run("SELECT id FROM users")
    db_run(
        "INSERT INTO usage_events (user_id, request_id, purpose, model, input_tokens,"
        " output_tokens, cache_read_tokens, cost_usd) VALUES ($1, 'x', 'gm', 'm', 0, 0, 0, 1)",
        user["id"],
    )
    with live.websocket_connect(f"/ws?token={token}") as ws:
        blocked = ask(ws, "b1", "npc", [{"role": "user", "content": "Hello?"}])
    assert [e["type"] for e in blocked] == ["error"]
    assert blocked[0]["code"] == "limit_exceeded"


def test_live_embeddings(live: TestClient) -> None:
    _, token = connect(live)
    response = live.post(
        "/embeddings",
        json={"model": "openai/text-embedding-3-small", "input": ["brass vault", "copper torch"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    vectors = response.json()["vectors"]
    assert len(vectors) == 2 and len(vectors[0]) == 1536
