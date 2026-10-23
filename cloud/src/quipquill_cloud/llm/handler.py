"""LLM worker: one command → a streamed OpenRouter call → protocol events.

Order of events: status → delta* → tool_call* → done (or a single error). `done.message` is the
assembled assistant message (text + complete tool calls) and is the agent's source of truth.
A decision model (ADR 70) answers without streaming: status → done, the answers as JSON text.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quipquill_cloud.core.config import Settings
from quipquill_cloud.llm import usage
from quipquill_cloud.llm.policy import DECISIONS, PURPOSES, Route, route
from quipquill_cloud.llm.prompts import PromptError, PromptStore, split_decision
from quipquill_cloud.mq.schemas import NO_USAGE, Command, Event, LlmRequest, Usage, error_event

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
        prompts: PromptStore,
    ) -> None:
        self._client = client
        self._sessionmaker = sessionmaker
        self._settings = settings
        self._prompts = prompts

    async def run(self, command: Command, is_cancelled: Callable[[], bool]) -> AsyncIterator[Event]:
        request = command.request
        common = command.address()
        reached = await usage.limit_reached(
            self._sessionmaker,
            uuid.UUID(command.user_id),
            command.tier,
            self._settings.daily_cost_limit_usd,
            self._settings.guest_total_cost_limit_usd,
        )
        if reached:
            yield error_event(reached, **common)
            return

        yield Event(type="status", status="thinking", meta=request.meta, **common)

        decision_model = DECISIONS.get(request.purpose)
        if decision_model is not None:
            async for event in self._decide(command, decision_model):
                yield event
            return

        chosen = route(request.model)
        if chosen is None:
            logger.warning("Unknown route %r in %s", request.model, request.request_id)
            yield error_event("bad_request", **common)
            return
        attempts = [self._settings.override(chosen)]
        spare = self._settings.fallback(chosen)
        if spare is not None and spare != attempts[0]:
            attempts.append(spare)

        for number, attempt in enumerate(attempts, start=1):
            try:
                body = self._body(request, attempt)
            except PromptError as error:
                logger.warning("Bad prompt in %s: %s", request.request_id, error)
                yield error_event("bad_request", **common)
                return

            answer = _Answer()
            failure: str | None = None
            try:
                async with asyncio.timeout(self._settings.llm_timeout_seconds):
                    async for event in self._stream(body, answer, is_cancelled):
                        yield Event(**event, **common)
            except (TimeoutError, openai.APITimeoutError):
                failure = "timeout"
            except openai.BadRequestError as error:
                logger.warning(
                    "%s rejected request %s: %s", attempt.model, request.request_id, error
                )
                failure = "bad_request"
            except openai.APIError as error:
                logger.warning("%s error for %s: %s", attempt.model, request.request_id, error)
                failure = "provider_error"

            if failure is None:
                async for event in self._finish(command, attempt.model, answer):
                    yield event
                return
            # The spare model only while the player has seen nothing of this answer.
            if number < len(attempts) and not answer.streamed:
                logger.warning(
                    "%s failed (%s) for %s; trying %s",
                    attempt.model,
                    failure,
                    request.request_id,
                    attempts[number].model,
                )
                continue
            yield error_event(failure, retryable=failure != "bad_request", **common)
            return

    async def _decide(self, command: Command, model: str) -> AsyncIterator[Event]:
        """A decision model (ADR 70): the template's opening text and the messages become the
        state, the template's questions go as they are; `done.message` is the answers as JSON."""
        request = command.request
        common = command.address()
        try:
            state, questions = self._decision_input(request)
        except PromptError as error:
            logger.warning("Bad decision prompt in %s: %s", request.request_id, error)
            yield error_event("bad_request", **common)
            return
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._settings.decision_timeout_seconds):
                reply = await self._client.post(
                    self._settings.openrouter_decisions_url,
                    cast_to=object,
                    body={"model": model, "state": state, "questions": questions},
                )
        except (TimeoutError, openai.APITimeoutError):
            yield error_event("timeout", retryable=True, **common)
            return
        except openai.BadRequestError as error:
            logger.warning("Decision rejected %s: %s", request.request_id, error)
            yield error_event("bad_request", **common)
            return
        except openai.APIError as error:
            logger.warning("Decision error for %s: %s", request.request_id, error)
            yield error_event("provider_error", retryable=True, **common)
            return
        answers = reply.get("answers") if isinstance(reply, dict) else None
        if not isinstance(answers, dict):
            logger.warning("Decision without answers for %s: %r", request.request_id, reply)
            yield error_event("provider_error", retryable=True, **common)
            return
        reported = reply.get("usage") or {}
        spent: Usage = {
            "input": reported.get("input_tokens") or 0,
            "output": reported.get("output_tokens") or 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "costUsd": reported.get("cost") or 0,
        }
        logger.info(
            "decision %s purpose=%s provider=%s total=%.2f in=%s cost=%s",
            request.request_id,
            request.purpose,
            reply.get("provider"),
            time.monotonic() - started,
            spent["input"],
            spent["costUsd"],
        )
        await self._record(command, model, spent)
        yield Event(
            type="done",
            message={"role": "assistant", "content": json.dumps(answers)},
            finish_reason="stop",
            usage=dict(spent),
            meta=request.meta,
            **common,
        )

    def _decision_input(self, request: LlmRequest) -> tuple[str, dict[str, dict[str, Any]]]:
        """System messages: the decision template (opening text and questions); the rest: what
        is checked now. Cache marks are dropped: decision models are not cached (ADR 70)."""
        messages = self._prompts.expand(request.messages)
        system = "\n\n".join(_text(m) for m in messages if m.get("role") == "system")
        rest = "\n\n".join(_text(m) for m in messages if m.get("role") != "system")
        opening, questions = split_decision(system, request.items)
        if not questions:
            raise PromptError("the decision has no questions to ask")
        return "\n\n".join(part for part in (opening, rest) if part), questions

    def _body(self, request: LlmRequest, chosen: Route) -> dict[str, Any]:
        """The OpenRouter call: the route's model and providers, the purpose's effort and limit;
        the prompts the messages name, filled in (ADR 57)."""
        params = PURPOSES[request.purpose]
        extra: dict[str, Any] = {"reasoning": {"effort": params.effort}}
        if chosen.providers:
            # Tried in this order, only these: the next one when one is unavailable.
            providers = list(chosen.providers)
            extra["provider"] = {"order": providers, "only": providers, "allow_fallbacks": False}
        body: dict[str, Any] = {
            "model": chosen.model,
            "messages": self._prompts.expand(request.messages),
            "max_tokens": params.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": extra,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.response_format:
            body["response_format"] = request.response_format
        return body

    async def _stream(
        self, body: dict[str, Any], answer: "_Answer", is_cancelled: Callable[[], bool]
    ) -> AsyncIterator[dict[str, Any]]:
        """Reads the stream into `answer`; yields the fields of each `delta` event."""
        stream = await self._client.chat.completions.create(**body)
        seq = 0
        async for chunk in stream:
            if is_cancelled():
                answer.finish = "cancelled"
                await stream.close()
                break
            if chunk.usage is not None:
                answer.reported = chunk.usage
            # OpenRouter names the provider that served the call (not in the SDK model).
            answer.provider = getattr(chunk, "provider", None) or answer.provider
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if answer.first is None and (delta.content or delta.tool_calls):
                answer.first = time.monotonic()
            if delta.content:
                answer.streamed = True
                answer.text += delta.content
                yield {"type": "delta", "seq": seq, "text": delta.content}
                seq += 1
            if delta.tool_calls:
                answer.tool_calls.add(delta.tool_calls)
            if choice.finish_reason:
                answer.finish = choice.finish_reason

    async def _finish(
        self, command: Command, model: str, answer: "_Answer"
    ) -> AsyncIterator[Event]:
        """Complete tool calls, the usage row and `done` with the assembled message."""
        request = command.request
        common = command.address()
        calls = answer.tool_calls.complete() if answer.finish != "cancelled" else []
        for call in calls:
            yield Event(type="tool_call", call=call, **common)

        spent = _usage(answer.reported)
        # Which provider served each purpose and what it cached (ADR 49): cache misses show here.
        done_at = time.monotonic()
        logger.info(
            "llm %s purpose=%s model=%s provider=%s ttft=%.2f total=%.2f in=%s cached=%s "
            "written=%s cost=%s",
            request.request_id,
            request.purpose,
            model,
            answer.provider,
            (answer.first or done_at) - answer.started,
            done_at - answer.started,
            spent["input"],
            spent["cacheRead"],
            spent["cacheWrite"],
            spent["costUsd"],
        )
        await self._record(command, model, spent)
        message: dict[str, Any] = {"role": "assistant", "content": answer.text or None}
        if calls:
            message["tool_calls"] = calls
        yield Event(
            type="done",
            message=message,
            finish_reason=answer.finish,
            usage=dict(spent),
            meta=request.meta,
            **common,
        )

    async def _record(self, command: Command, model: str, spent: Usage) -> None:
        request = command.request
        try:
            await usage.record(
                self._sessionmaker,
                user_id=uuid.UUID(command.user_id),
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


def _text(message: dict[str, Any]) -> str:
    """A message's text: the string content, or its text parts joined."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


@dataclass
class _Answer:
    """What the stream has said so far."""

    text: str = ""
    tool_calls: _ToolCallBuffer = field(default_factory=_ToolCallBuffer)
    finish: str = "stop"
    reported: Any = None
    provider: str | None = None
    # A delta reached the player: too late to switch to the spare model.
    streamed: bool = False
    started: float = field(default_factory=time.monotonic)
    # When the first text or tool call arrived (time to first token).
    first: float | None = None


def _usage(reported: Any) -> Usage:
    """Protocol usage from OpenRouter's final chunk (zeros if the stream was cut short)."""
    if reported is None:
        return NO_USAGE.copy()
    data = reported.model_dump()
    details = data.get("prompt_tokens_details") or {}
    return {
        "input": data.get("prompt_tokens") or 0,
        "output": data.get("completion_tokens") or 0,
        "cacheRead": details.get("cached_tokens") or 0,
        "cacheWrite": details.get("cache_write_tokens") or 0,
        "costUsd": data.get("cost") or 0,
    }
