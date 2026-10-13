"""Command worker: one LLM request in, a stream of events out.

Error policy (from AI Samba): a malformed command is rejected without requeue (it will not get
better on retry); a failure while handling becomes an `error` event and the command is still
acked, so a poison message never loops.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from aio_pika.abc import AbstractIncomingMessage
from pydantic import ValidationError

from quipquill_cloud.mq.schemas import Command, Event, error_event

logger = logging.getLogger(__name__)

PublishEvent = Callable[[Event], Awaitable[None]]


class CancelRegistry:
    """Requests the client asked to stop. In-process: API and workers share one process."""

    def __init__(self) -> None:
        self._cancelled: set[tuple[str, str]] = set()

    def cancel(self, user_id: str, request_id: str) -> None:
        self._cancelled.add((user_id, request_id))

    def is_cancelled(self, user_id: str, request_id: str) -> bool:
        return (user_id, request_id) in self._cancelled

    def forget(self, user_id: str, request_id: str) -> None:
        self._cancelled.discard((user_id, request_id))


class Handler(Protocol):
    def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]: ...


class EchoHandler:
    """Development stand-in for the LLM (stage 2.2): streams the last user message back."""

    FAIL_TRIGGER = "!fail"

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]:
        request = command.request
        common: dict[str, Any] = {
            "user_id": command.user_id,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "channel": request.channel,
        }
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
            usage={"input": 0, "output": 0, "cacheRead": 0, "costUsd": 0},
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
        try:
            await self.handle(command)
        except Exception:
            logger.exception("Command %s failed", request.request_id)
            await self._publish(
                error_event(
                    "provider_error",
                    user_id=command.user_id,
                    request_id=request.request_id,
                    session_id=request.session_id,
                    channel=request.channel,
                    retryable=True,
                )
            )
        finally:
            self._cancels.forget(command.user_id, request.request_id)
            await message.ack()

    async def handle(self, command: Command) -> None:
        def is_cancelled() -> bool:
            return self._cancels.is_cancelled(command.user_id, command.request.request_id)

        async for event in self._handler.run(command, is_cancelled):
            await self._publish(event)
