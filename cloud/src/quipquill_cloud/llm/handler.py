"""LLM worker: one command → a streamed OpenRouter call → protocol events.

Order of events: status → delta* → tool_call* → done (or a single error). `done.message` is the
assembled assistant message (text + complete tool calls) and is the agent's source of truth.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quipquill_cloud.core.config import Settings
from quipquill_cloud.llm import usage
from quipquill_cloud.llm.policy import PURPOSES, route
from quipquill_cloud.mq.schemas import Command, Event, error_event

logger = logging.getLogger(__name__)


class _ToolCallBuffer:
    """Accumulates streamed tool-call fragments by their index."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def add(self, fragments: list[Any]) -> None:
        for fragment in fragments:
            call = self._calls.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
            if fragment.id:
                call["id"] = fragment.id
            function = fragment.function
            if function is not None:
                call["name"] += function.name or ""
                call["arguments"] += function.arguments or ""

    def complete(self) -> list[dict[str, Any]]:
        return [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
            }
            for _, call in sorted(self._calls.items())
        ]


class OpenRouterHandler:
    def __init__(
        self,
        client: AsyncOpenAI,
        sessionmaker: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._client = client
        self._sessionmaker = sessionmaker
        self._settings = settings

    async def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]:
        request = command.request
        user_id = uuid.UUID(command.user_id)
        common: dict[str, Any] = {
            "user_id": command.user_id,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "channel": request.channel,
        }
        cap = self._settings.daily_cost_limit_usd
        if await usage.over_limit(self._sessionmaker, user_id, command.tier, cap):
            yield error_event("limit_exceeded", **common)
            return

        yield Event(type="status", status="thinking", meta=request.meta, **common)

        chosen = route(request.model)
        if chosen is None:
            logger.warning("Unknown route %r in %s", request.model, request.request_id)
            yield error_event("bad_request", **common)
            return
        model = chosen.model
        providers = self._settings.provider_order or list(chosen.providers)
        params = PURPOSES[request.purpose]
        body: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "max_tokens": params.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {
                "reasoning": {"effort": params.effort},
                # Tried in this order, only these: the next one when one is unavailable.
                "provider": {"order": providers, "only": providers, "allow_fallbacks": False},
            },
        }
        if request.tools:
            body["tools"] = request.tools
        if request.response_format:
            body["response_format"] = request.response_format

        text = ""
        seq = 0
        tool_calls = _ToolCallBuffer()
        finish = "stop"
        reported: Any = None
        provider: str | None = None
        started = time.monotonic()
        first: float | None = None
        try:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                stream = await self._client.chat.completions.create(**body)
                async for chunk in stream:
                    if is_cancelled():
                        finish = "cancelled"
                        await stream.close()
                        break
                    if chunk.usage is not None:
                        reported = chunk.usage
                    # OpenRouter names the provider that served the call (not in the SDK model).
                    provider = getattr(chunk, "provider", None) or provider
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if first is None and (delta.content or delta.tool_calls):
                        first = time.monotonic()
                    if delta.content:
                        text += delta.content
                        yield Event(type="delta", seq=seq, text=delta.content, **common)
                        seq += 1
                    if delta.tool_calls:
                        tool_calls.add(delta.tool_calls)
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except TimeoutError:
            yield error_event("timeout", retryable=True, **common)
            return
        except openai.APITimeoutError:
            yield error_event("timeout", retryable=True, **common)
            return
        except openai.BadRequestError as error:
            logger.warning("Provider rejected request %s: %s", request.request_id, error)
            yield error_event("bad_request", **common)
            return
        except openai.APIError as error:
            logger.warning("Provider error for %s: %s", request.request_id, error)
            yield error_event("provider_error", retryable=True, **common)
            return

        calls = tool_calls.complete() if finish != "cancelled" else []
        for call in calls:
            yield Event(type="tool_call", call=call, **common)

        spent = _usage(reported)
        # Which provider served each purpose and what it cached (ADR 49): cache misses show here.
        done_at = time.monotonic()
        logger.info(
            "llm %s purpose=%s provider=%s ttft=%.2f total=%.2f in=%s cached=%s written=%s cost=%s",
            request.request_id,
            request.purpose,
            provider,
            (first or done_at) - started,
            done_at - started,
            spent["input"],
            spent["cacheRead"],
            spent["cacheWrite"],
            spent["costUsd"],
        )
        try:
            await usage.record(
                self._sessionmaker,
                user_id=user_id,
                request_id=request.request_id,
                session_id=request.session_id,
                purpose=request.purpose,
                model=model,
                input_tokens=spent["input"],
                output_tokens=spent["output"],
                cache_read_tokens=spent["cacheRead"],
                cost_usd=spent["costUsd"],
            )
        except Exception:
            # The answer is already generated and paid for: losing the accounting row is
            # cheaper than leaving the client without `done` after deltas and tool calls.
            logger.exception("Could not record usage for %s: %s", request.request_id, spent)
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = calls
        yield Event(
            type="done",
            message=message,
            finish_reason=finish,
            usage=spent,
            meta=request.meta,
            **common,
        )


def _usage(reported: Any) -> dict[str, Any]:
    """Protocol usage from OpenRouter's final chunk (zeros if the stream was cut short)."""
    if reported is None:
        return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "costUsd": 0}
    data = reported.model_dump()
    details = data.get("prompt_tokens_details") or {}
    return {
        "input": data.get("prompt_tokens") or 0,
        "output": data.get("completion_tokens") or 0,
        "cacheRead": details.get("cached_tokens") or 0,
        "cacheWrite": details.get("cache_write_tokens") or 0,
        "costUsd": data.get("cost") or 0,
    }
