"""Minimal chat client. Everything the extractor needs is one system+user call returning text."""

from typing import Protocol

from openai import AsyncOpenAI

from quipquill_builder.config import Settings


class ChatClient(Protocol):
    model: str

    async def complete(self, system: str, user: str) -> str: ...


class OpenRouterChatClient:
    """Chat completions through OpenRouter's OpenAI-compatible API."""

    def __init__(self, settings: Settings, model: str) -> None:
        if settings.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see builder/.env.example)")
        self.model = model
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def complete(self, system: str, user: str) -> str:
        options: dict = {"max_tokens": self._settings.max_tokens}
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        if self._settings.reasoning_effort:
            options["extra_body"] = {"reasoning": {"effort": self._settings.reasoning_effort}}
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **options,
        )
        return response.choices[0].message.content or ""
