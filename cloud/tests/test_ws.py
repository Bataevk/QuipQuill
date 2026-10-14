"""WebSocket + RabbitMQ round trips with the echo handler (needs make infra-up)."""

import asyncio

import aio_pika
import pytest
from conftest import INVITE, RABBITMQ_URL, ws_connect
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

TEXT = "the brass vault opens at midnight under a comet"


def token_for(client: TestClient, email: str = "ws@example.com") -> str:
    response = client.post(
        "/auth/register",
        json={"email": email, "password": "long enough pass", "invite_code": INVITE},
    )
    return response.json()["access_token"]


def request(text: str = TEXT, request_id: str = "req-1", session_id: str = "save-1") -> dict:
    return {
        "v": 1,
        "type": "llm.request",
        "requestId": request_id,
        "sessionId": session_id,
        "channel": "npc:innkeeper",
        "purpose": "npc",
        "messages": [{"role": "user", "content": text}],
        "meta": {"speaker": "Innkeeper"},
    }


def receive_until_done(ws) -> list[dict]:
    events = []
    while True:
        event = ws.receive_json()
        if event["type"] == "heartbeat":
            continue
        events.append(event)
        if event["type"] in ("done", "error"):
            return events


def test_echo_streams_status_deltas_done(client: TestClient) -> None:
    with ws_connect(client, token_for(client)) as ws:
        ws.send_json(request())
        events = receive_until_done(ws)

    status, *deltas, done = events
    assert status["type"] == "status" and status["status"] == "thinking"
    assert status["meta"] == {"speaker": "Innkeeper"}
    assert status["channel"] == "npc:innkeeper"
    assert [e["type"] for e in deltas] == ["delta"] * len(TEXT.split())
    assert [e["seq"] for e in deltas] == list(range(len(deltas)))
    assert "".join(e["text"] for e in deltas) == TEXT
    assert done["message"] == {"role": "assistant", "content": TEXT}
    assert done["finishReason"] == "stop"
    assert all(e["requestId"] == "req-1" and e["sessionId"] == "save-1" for e in events)
    assert all("userId" not in e for e in events)  # internal routing never leaks


def test_resume_replays_text_in_flight(settings: Settings) -> None:
    slow = settings.model_copy(update={"echo_delay_seconds": 0.1})
    long_text = " ".join(f"word{i}" for i in range(20))
    with TestClient(create_app(slow)) as client:
        token = token_for(client)
        with ws_connect(client, token) as ws:
            ws.send_json(request(long_text))
            assert ws.receive_json()["type"] == "status"
            seen = ws.receive_json()["text"] + ws.receive_json()["text"]
        # Connection dropped mid-generation; the worker keeps going.
        with ws_connect(client, token) as ws:
            ws.send_json({"v": 1, "type": "resume", "sessionId": "save-1"})
            replay = ws.receive_json()
            rest = receive_until_done(ws)

    assert replay["type"] == "replay"
    assert replay["requestId"] == "req-1"
    assert replay["meta"] == {"speaker": "Innkeeper"}
    assert replay["text"].startswith(seen)
    later = "".join(e["text"] for e in rest if e["type"] == "delta")
    assert replay["text"] + later == long_text
    assert rest[-1]["message"]["content"] == long_text


def test_resume_after_finish_repeats_done(client: TestClient) -> None:
    token = token_for(client)
    with ws_connect(client, token) as ws:
        ws.send_json(request())
        receive_until_done(ws)
    with ws_connect(client, token) as ws:
        ws.send_json({"v": 1, "type": "resume", "sessionId": "save-1"})
        repeated = ws.receive_json()
    assert repeated["type"] == "done" and repeated["requestId"] == "req-1"


def test_bad_message_keeps_socket_open(client: TestClient) -> None:
    with ws_connect(client, token_for(client)) as ws:
        ws.send_text("not json")
        assert ws.receive_json() == {
            "v": 1,
            "type": "error",
            "code": "bad_request",
            "statusKey": "error.bad_request",
            "retryable": False,
        }
        ws.send_json({"v": 1, "type": "llm.request", "requestId": "x"})  # missing fields
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_json({"v": 1, "type": "ping"})
        assert ws.receive_json()["type"] == "heartbeat"


@pytest.mark.parametrize(
    "first",
    [
        {"v": 1, "type": "auth", "token": "garbage"},
        {"v": 1, "type": "ping"},  # anything before auth
        "not json",
    ],
)
def test_rejects_unauthenticated(client: TestClient, first: dict | str) -> None:
    with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect("/ws") as ws:
        ws.send_text(first) if isinstance(first, str) else ws.send_json(first)
        ws.receive_json()
    assert closed.value.code == 4401


def test_closes_a_socket_that_never_authenticates(settings: Settings) -> None:
    quick = settings.model_copy(update={"ws_auth_timeout": 0.2})
    with (
        TestClient(create_app(quick)) as client,
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect("/ws") as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4401


def test_token_in_the_url_is_not_accepted(client: TestClient) -> None:
    """ADR 22: tokens in query strings end up in access logs."""
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect(f"/ws?token={token_for(client)}") as ws,
    ):
        ws.send_json({"v": 1, "type": "ping"})
        ws.receive_json()
    assert closed.value.code == 4401


def test_sessions_are_isolated_between_users(settings: Settings) -> None:
    slow = settings.model_copy(update={"echo_delay_seconds": 0.1})
    with TestClient(create_app(slow)) as client:
        alice, bob = token_for(client, "alice@example.com"), token_for(client, "bob@example.com")
        with ws_connect(client, alice) as ws_alice:
            ws_alice.send_json(request(" ".join(["secret"] * 20)))
            assert ws_alice.receive_json()["type"] == "status"
            with ws_connect(client, bob) as ws_bob:
                # Same session id, different user: nothing to replay, nothing delivered.
                ws_bob.send_json({"v": 1, "type": "resume", "sessionId": "save-1"})
                ws_bob.send_json({"v": 1, "type": "ping"})
                assert ws_bob.receive_json()["type"] == "heartbeat"
            ws_alice.send_json({"v": 1, "type": "cancel", "requestId": "req-1"})
            receive_until_done(ws_alice)


def test_handler_failure_becomes_error_event(client: TestClient) -> None:
    with ws_connect(client, token_for(client)) as ws:
        ws.send_json(request("!fail"))
        status, error = receive_until_done(ws)
    assert status["type"] == "status"
    assert error["type"] == "error"
    assert error["code"] == "provider_error"
    assert error["retryable"] is True


def test_cancel_stops_generation(settings: Settings) -> None:
    slow = settings.model_copy(update={"echo_delay_seconds": 0.1})
    with (
        TestClient(create_app(slow)) as client,
        ws_connect(client, token_for(client)) as ws,
    ):
        ws.send_json(request(" ".join(["word"] * 50)))
        assert ws.receive_json()["type"] == "status"
        ws.receive_json()  # first delta
        ws.send_json({"v": 1, "type": "cancel", "requestId": "req-1"})
        done = receive_until_done(ws)[-1]
    assert done["finishReason"] == "cancelled"
    assert len(done["message"]["content"].split()) < 50


def test_malformed_command_is_dropped_and_worker_survives(
    client: TestClient, settings: Settings
) -> None:
    async def publish_garbage() -> None:
        async with await aio_pika.connect(RABBITMQ_URL) as connection:
            channel = await connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(b"{not json"), routing_key=f"{settings.queue_prefix}.commands"
            )

    asyncio.run(publish_garbage())
    with ws_connect(client, token_for(client)) as ws:
        ws.send_json(request("still alive"))
        assert receive_until_done(ws)[-1]["message"]["content"] == "still alive"


def test_heartbeat_while_generation_runs(settings: Settings) -> None:
    # Words arrive slower than the heartbeat interval, so heartbeats fill the gaps.
    slow = settings.model_copy(update={"echo_delay_seconds": 0.5, "ws_heartbeat_interval": 0.1})
    with (
        TestClient(create_app(slow)) as client,
        ws_connect(client, token_for(client)) as ws,
    ):
        ws.send_json(request("one two three"))
        types = []
        while not types or types[-1] != "done":
            types.append(ws.receive_json()["type"])
    assert "heartbeat" in types


def test_websocket_refused_when_queue_is_down(settings: Settings) -> None:
    down = settings.model_copy(
        update={"rabbitmq_url": "amqp://nobody:nothing@localhost:1/", "mq_connect_timeout": 1}
    )
    with TestClient(create_app(down)) as client:
        assert client.get("/health").status_code == 200  # REST keeps working
        token = token_for(client)
        with (
            pytest.raises(WebSocketDisconnect) as closed,
            ws_connect(client, token),
        ):
            pass
    assert closed.value.code == 1013
