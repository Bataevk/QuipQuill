"""Chat clients. The extractor needs only "system + user → text", one at a time or in bulk."""

from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from quipquill_builder.batch import BatchError, BatchRunner, api_key
from quipquill_builder.config import Settings

Prompt = tuple[str, str]  # (system, user)


class ChatClient(Protocol):
    model: str

    async def complete(self, system: str, user: str) -> str: ...


@runtime_checkable
class BulkChatClient(Protocol):
    """A client that answers many prompts in one job (e.g. a provider batch)."""

    model: str

    async def complete_many(self, prompts: list[Prompt]) -> list[str | Exception]: ...


def _request_body(settings: Settings, system: str, user: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": settings.max_tokens,
    }
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.reasoning_effort:
        body["reasoning"] = {"effort": settings.reasoning_effort}
    return body


class OpenRouterChatClient:
    """Synchronous chat completions through OpenRouter's OpenAI-compatible API."""

    def __init__(self, settings: Settings, model: str) -> None:
        self.model = model
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=api_key(settings),
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def complete(self, system: str, user: str) -> str:
        body = _request_body(self._settings, system, user)
        messages = body.pop("messages")
        max_tokens = body.pop("max_tokens")
        extra_body = {k: body.pop(k) for k in ("reasoning",) if k in body}
        if self._settings.provider_only:
            extra_body["provider"] = {
                "only": self._settings.provider_only,
                "allow_fallbacks": False,
            }
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
            **body,  # temperature, if set
        )
        return response.choices[0].message.content or ""


class OpenRouterBatchClient:
    """Chat completions through the Batch API (see batch.py): one job for all prompts."""

    def __init__(self, settings: Settings, model: str, **runner_options: Any) -> None:
        self.model = model
        self._settings = settings
        self._runner = BatchRunner(settings, **runner_options)

    async def complete(self, system: str, user: str) -> str:
        (result,) = await self.complete_many([(system, user)])
        if isinstance(result, Exception):
            raise result
        return result

    async def complete_many(self, prompts: list[Prompt]) -> list[str | Exception]:
        bodies = [_request_body(self._settings, system, user) for system, user in prompts]
        results = await self._runner.run("/v1/chat/completions", self.model, bodies)
        return [_completion_text(result) for result in results]


def _completion_text(result: dict[str, Any] | Exception) -> str | Exception:
    if isinstance(result, Exception):
        return result
    try:
        return result["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return BatchError(f"Unexpected completion shape: {str(result)[:200]}")
