"""WebSocket endpoint `/ws`: first message `auth` (ADR 22), then docs/architecture/protocol.md."""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from quipquill_cloud.core import logs
from quipquill_cloud.core.security import InvalidToken, decode_access_token
from quipquill_cloud.db.models import User
from quipquill_cloud.mq.dispatcher import SessionKey
from quipquill_cloud.mq.schemas import (
    Auth,
    Cancel,
    Command,
    Event,
    LlmRequest,
    Ping,
    Resume,
    client_message,
    error_event,
)
from quipquill_cloud.services import Services, get_services

logger = logging.getLogger(__name__)
router = APIRouter()

CLOSE_UNAUTHORIZED = 4401
CLOSE_TRY_AGAIN_LATER = 1013
MAX_MESSAGE_BYTES = 1_000_000


class Connection:
    """One client socket: serialized sends, the sessions it follows, last activity."""

    def __init__(self, websocket: WebSocket, user: User, services: Services) -> None:
        self.websocket = websocket
        self.services = services
        self.user_id = str(user.id)
        self.tier = user.tier
        self.sessions: set[SessionKey] = set()
        self.last_send = time.monotonic()
        self._lock = asyncio.Lock()

    async def send(self, event: Event) -> None:
        async with self._lock:
            await self.websocket.send_json(event.wire())
            self.last_send = time.monotonic()


async def _authenticate(websocket: WebSocket, services: Services) -> User | None:
    """The first message must be `auth` with a valid access token (ADR 22), within the timeout.

    The token is never read from the URL: query strings end up in access logs (uvicorn, Caddy).
    """
    try:
        async with asyncio.timeout(services.settings.ws_auth_timeout):
            raw = await websocket.receive_text()
        message = Auth.model_validate_json(raw)
        user_id = uuid.UUID(decode_access_token(message.token, services.settings)["sub"])
    except (TimeoutError, ValidationError, InvalidToken, ValueError):
        return None
    async with services.sessionmaker() as session:
        return await session.get(User, user_id)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    services: Annotated[Services, Depends(get_services)],
) -> None:
    await websocket.accept()
    if services.broker is None:
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason="queue_unavailable")
        return
    try:
        user = await _authenticate(websocket, services)
    except WebSocketDisconnect:
        return
    if user is None:
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="not_authenticated")
        return

    logs.user_id.set(str(user.id))
    conn = Connection(websocket, user, services)
    await conn.send(Event(type="ready"))
    heartbeat = asyncio.create_task(_heartbeat(conn, services.settings.ws_heartbeat_interval))
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
        services.dispatcher.unsubscribe(conn)


async def _handle_client_message(conn: Connection, raw: str) -> None:
    services = conn.services
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
            refusal = _refuse(conn, message)
            if refusal is not None:
                await conn.send(
                    error_event(refusal, retryable=refusal == "rate_limited", **message.address())
                )
                return
            key = (conn.user_id, message.session_id)
            if key not in conn.sessions:
                await services.dispatcher.subscribe(key, conn, replay=False)
                conn.sessions.add(key)
            try:
                if services.broker is None:
                    raise RuntimeError("the queue is not connected")
                await services.broker.publish_command(
                    Command(user_id=conn.user_id, tier=conn.tier, request=message)
                )
            except Exception:
                logger.exception("Could not enqueue request %s", message.request_id)
                # No preceding `status`: the generation never started.
                await conn.send(
                    error_event("queue_unavailable", retryable=True, **message.address())
                )
        case Resume():
            key = (conn.user_id, message.session_id)
            await services.dispatcher.subscribe(key, conn, replay=True)
            conn.sessions.add(key)
        case Cancel():
            services.cancels.cancel(conn.user_id, message.request_id)
        case Ping():
            await conn.send(Event(type="heartbeat"))


def _refuse(conn: Connection, message: LlmRequest) -> str | None:
    """A reason to reject the request before it reaches the queue, or None."""
    settings = conn.services.settings
    if len(message.tools or []) > settings.max_tools:
        return "too_many_tools"
    approx_tokens = len(json.dumps(message.messages, ensure_ascii=False)) // 4
    if approx_tokens > settings.max_request_tokens:
        return "request_too_large"
    if not conn.services.rate_limiter.allow(conn.user_id):
        return "rate_limited"
    return None


async def _heartbeat(conn: Connection, interval: float) -> None:
    """While a generation is running, never stay silent longer than `interval` seconds."""
    dispatcher = conn.services.dispatcher
    while True:
        await asyncio.sleep(interval)
        busy = any(dispatcher.has_active(key) for key in conn.sessions)
        if busy and time.monotonic() - conn.last_send >= interval:
            try:
                await conn.send(Event(type="heartbeat"))
            except Exception:
                return
