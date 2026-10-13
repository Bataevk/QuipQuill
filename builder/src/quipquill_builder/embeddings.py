"""Embeddings via OpenRouter and float32 BLOB packing shared with the mobile core."""

import struct
from typing import Any, Protocol

from openai import AsyncOpenAI

from quipquill_builder.batch import BatchError, BatchRunner, api_key
from quipquill_builder.config import Settings

BATCH_SIZE = 64  # texts per embeddings request


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _chunks(texts: list[str]) -> list[list[str]]:
    return [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]


def _check_count(vectors: list[list[float]], texts: list[str]) -> list[list[float]]:
    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
    return vectors


class OpenRouterEmbedder:
    """Direct embeddings calls (no provider restriction for this model)."""

    def __init__(self, settings: Settings, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key(settings),
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for chunk in _chunks(texts):
            response = await self._client.embeddings.create(model=self.model, input=chunk)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return _check_count(vectors, texts)


class OpenRouterBatchEmbedder:
    """Embeddings through the Batch API: one job, one request per 64 texts."""

    def __init__(self, settings: Settings, model: str, **runner_options: Any) -> None:
        self.model = model
        self._runner = BatchRunner(settings, **runner_options)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        bodies = [{"input": chunk} for chunk in _chunks(texts)]
        results = await self._runner.run("/v1/embeddings", self.model, bodies)
        vectors: list[list[float]] = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            try:
                data = sorted(result["data"], key=lambda d: d["index"])
                vectors.extend(item["embedding"] for item in data)
            except (KeyError, TypeError) as error:
                raise BatchError(f"Unexpected embeddings shape: {str(result)[:200]}") from error
        return _check_count(vectors, texts)


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))
