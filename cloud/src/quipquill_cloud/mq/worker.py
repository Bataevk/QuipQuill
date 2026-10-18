"""Command worker: one LLM request in, a stream of events out.

Error policy (from AI Samba): a malformed command is rejected without requeue (it will not get
better on retry); a failure while handling becomes an `error` event and the command is still
acked, so a poison message never loops.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from aio_pika.abc import AbstractIncomingMessage
from pydantic import ValidationError

from quipquill_cloud.core import logs as log_context
from quipquill_cloud.mq.schemas import NO_USAGE, Command, Event, error_event

logger = logging.getLogger(__name__)

PublishEvent = Callable[[Event], Awaitable[None]]


class CancelRegistry:
    """Requests the client asked to stop. In-process: API and workers share one process.

    A cancel can arrive before its command reaches a worker, or after it finished (then nobody
    calls `forget`), so entries also expire after `ttl` seconds.
    """

    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl
        self._clock = clock
        self._cancelled: dict[tuple[str, str], float] = {}

    def cancel(self, user_id: str, request_id: str) -> None:
        now = self._clock()
        deadline = now - self._ttl
        for key in [k for k, at in self._cancelled.items() if at < deadline]:
            del self._cancelled[key]
        self._cancelled[(user_id, request_id)] = now

    def is_cancelled(self, user_id: str, request_id: str) -> bool:
        return (user_id, request_id) in self._cancelled

    def forget(self, user_id: str, request_id: str) -> None:
        self._cancelled.pop((user_id, request_id), None)

    def __len__(self) -> int:
        return len(self._cancelled)


class Handler(Protocol):
    def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]: ...


class EchoHandler:
    """Development stand-in for the LLM (stage 2.2): streams the last user message back."""

    FAIL_TRIGGER = "!fail"

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]:
        request = command.request
        common = command.address()
        yield Event(type="status", status="thinking", meta=request.meta, **common)
        text = _last_user_text(request.messages)
        if text.strip() == self.FAIL_TRIGGER:
            raise RuntimeError("echo failure requested")
        sent = ""
        finish = "stop"
        for seq, word in enumerate(text.split(" ")):
            if is_cancelled():
                finish = "cancelled"
                break
            piece = word if seq == 0 else " " + word
            sent += piece
            yield Event(type="delta", seq=seq, text=piece, **common)
            await asyncio.sleep(self._delay)
        yield Event(
            type="done",
            message={"role": "assistant", "content": sent},
            finish_reason=finish,
            usage=dict(NO_USAGE),
            meta=request.meta,
            **common,
        )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class CommandWorker:
    def __init__(self, handler: Handler, publish: PublishEvent, cancels: CancelRegistry) -> None:
        self._handler = handler
        self._publish = publish
        self._cancels = cancels

    async def on_message(self, message: AbstractIncomingMessage) -> None:
        try:
            command = Command.model_validate_json(message.body)
        except (ValidationError, ValueError):
            logger.error("Rejecting malformed command: %r", message.body[:200])
            await message.reject(requeue=False)
            return

        request = command.request
        log_context.request_id.set(request.request_id)
        log_context.session_id.set(request.session_id)
        log_context.user_id.set(command.user_id)
        try:
            await self.handle(command)
        except Exception:
            logger.exception("Command %s failed", request.request_id)
            await self._publish(error_event("provider_error", retryable=True, **command.address()))
        finally:
            self._cancels.forget(command.user_id, request.request_id)
            await message.ack()

    async def handle(self, command: Command) -> None:
        def is_cancelled() -> bool:
            return self._cancels.is_cancelled(command.user_id, command.request.request_id)

        async for event in self._handler.run(command, is_cancelled):
            await self._publish(event)
