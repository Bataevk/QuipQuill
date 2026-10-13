"""WebSocket endpoint: `/ws?token=<access JWT>`. Contract: docs/architecture/protocol.md."""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from quipquill_cloud.core import logs
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.core.security import InvalidToken, decode_access_token
from quipquill_cloud.db.models import User
from quipquill_cloud.mq.dispatcher import SessionKey
from quipquill_cloud.mq.schemas import (
    Cancel,
    Command,
    Event,
    LlmRequest,
    Ping,
    Resume,
    client_message,
    error_event,
)

logger = logging.getLogger(__name__)
router = APIRouter()

CLOSE_UNAUTHORIZED = 4401
CLOSE_TRY_AGAIN_LATER = 1013
MAX_MESSAGE_BYTES = 1_000_000


class Connection:
    """One client socket: serialized sends, the sessions it follows, last activity."""

    def __init__(self, websocket: WebSocket, user: User) -> None:
        self.websocket = websocket
        self.user_id = str(user.id)
        self.tier = user.tier
        self.sessions: set[SessionKey] = set()
        self.last_send = time.monotonic()
        self._lock = asyncio.Lock()

    async def send(self, event: Event) -> None:
        async with self._lock:
            await self.websocket.send_json(event.wire())
            self.last_send = time.monotonic()


async def _authenticate(websocket: WebSocket, token: str | None, settings: Settings) -> User | None:
    if not token:
        return None
    try:
        user_id = uuid.UUID(decode_access_token(token, settings)["sub"])
    except (InvalidToken, ValueError):
        return None
    async with websocket.app.state.sessionmaker() as session:
        return await session.get(User, user_id)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Query()] = None,
) -> None:
    user = await _authenticate(websocket, token, settings)
    if user is None:
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="not_authenticated")
        return
    state = websocket.app.state
    if getattr(state, "broker", None) is None:
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason="queue_unavailable")
        return

    await websocket.accept()
    logs.user_id.set(str(user.id))
    conn = Connection(websocket, user)
    heartbeat = asyncio.create_task(_heartbeat(conn, settings.ws_heartbeat_interval))
    try:
        while True:
            raw = await websocket.receive_text()
            await _handle_client_message(conn, raw)
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        state.dispatcher.unsubscribe(conn)


async def _handle_client_message(conn: Connection, raw: str) -> None:
    state = conn.websocket.app.state
    if len(raw.encode()) > MAX_MESSAGE_BYTES:
        await conn.send(error_event("message_too_large"))
        return
    try:
        message = client_message.validate_python(json.loads(raw))
    except (ValueError, ValidationError):
        # The socket stays open: a bad message is the client's bug, not a reason to disconnect.
        await conn.send(error_event("bad_request"))
        return

    match message:
        case LlmRequest():
            logs.request_id.set(message.request_id)
            logs.session_id.set(message.session_id)
            refusal = _refuse(conn, message, state.settings)
            if refusal is not None:
                await conn.send(
                    error_event(
                        refusal,
                        request_id=message.request_id,
                        session_id=message.session_id,
                        channel=message.channel,
                        retryable=refusal == "rate_limited",
                    )
                )
                return
            key = (conn.user_id, message.session_id)
            if key not in conn.sessions:
                await state.dispatcher.subscribe(key, conn, replay=False)
                conn.sessions.add(key)
            try:
                await state.broker.publish_command(
                    Command(user_id=conn.user_id, tier=conn.tier, request=message)
                )
            except Exception:
                logger.exception("Could not enqueue request %s", message.request_id)
                # No preceding `status`: the generation never started.
                await conn.send(
                    error_event(
                        "queue_unavailable",
                        request_id=message.request_id,
                        session_id=message.session_id,
                        channel=message.channel,
                        retryable=True,
                    )
                )
        case Resume():
            key = (conn.user_id, message.session_id)
            await state.dispatcher.subscribe(key, conn, replay=True)
            conn.sessions.add(key)
        case Cancel():
            state.cancels.cancel(conn.user_id, message.request_id)
        case Ping():
            await conn.send(Event(type="heartbeat"))


def _refuse(conn: Connection, message: LlmRequest, settings: Settings) -> str | None:
    """A reason to reject the request before it reaches the queue, or None."""
    if len(message.tools or []) > settings.max_tools:
        return "too_many_tools"
    approx_tokens = len(json.dumps(message.messages, ensure_ascii=False)) // 4
    if approx_tokens > settings.max_request_tokens:
        return "request_too_large"
    if not conn.websocket.app.state.rate_limiter.allow(conn.user_id):
        return "rate_limited"
    return None


async def _heartbeat(conn: Connection, interval: float) -> None:
    """While a generation is running, never stay silent longer than `interval` seconds."""
    dispatcher = conn.websocket.app.state.dispatcher
    while True:
        await asyncio.sleep(interval)
        busy = any(dispatcher.has_active(key) for key in conn.sessions)
        if busy and time.monotonic() - conn.last_send >= interval:
            try:
                await conn.send(Event(type="heartbeat"))
            except Exception:
                return
