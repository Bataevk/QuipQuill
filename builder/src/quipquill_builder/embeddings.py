"""Embeddings via OpenRouter and float32 BLOB packing shared with the mobile core."""

import struct
from typing import Protocol

from openai import AsyncOpenAI

from quipquill_builder.config import Settings


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenRouterEmbedder:
    def __init__(self, settings: Settings, model: str, batch_size: int = 64) -> None:
        if settings.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see builder/.env.example)")
        self.model = model
        self._batch_size = batch_size
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = await self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))
