"""A stand-in for AsyncOpenAI: scripted chat streams and embeddings, no network."""

import asyncio
from typing import Any

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk


def chunk(content: str | None = None, *, tool_calls=None, finish: str | None = None, usage=None):
    delta: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    data: dict[str, Any] = {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "anthropic/claude-haiku-5.5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 0},
    "cost": 0.000123,
}


def text_stream(*pieces: str) -> list:
    return [chunk(p) for p in pieces] + [chunk("", finish="stop", usage=USAGE)]


def tool_stream() -> list:
    return [
        chunk("Heading north."),
        chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "move", "arguments": ""},
                }
            ]
        ),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"to": "Torch'}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": 'light Corridor"}'}}]),
        chunk(
            tool_calls=[
                {
                    "index": 1,
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": '{"target": "door"}'},
                }
            ]
        ),
        chunk("", finish="tool_calls", usage=USAGE),
    ]


class _Stream:
    def __init__(self, chunks: list, delay: float) -> None:
        self._chunks = chunks
        self._delay = delay
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self._chunks:
            if self.closed:
                return
            await asyncio.sleep(self._delay)
            yield item

    async def close(self) -> None:
        self.closed = True


class _Completions:
    def __init__(self, owner: "FakeOpenRouter") -> None:
        self._owner = owner

    async def create(self, **body: Any) -> _Stream:
        self._owner.requests.append(body)
        if self._owner.error is not None:
            raise self._owner.error
        return _Stream(self._owner.chunks, self._owner.delay)


class _Chat:
    def __init__(self, owner: "FakeOpenRouter") -> None:
        self.completions = _Completions(owner)


class _Embeddings:
    def __init__(self, owner: "FakeOpenRouter") -> None:
        self._owner = owner

    async def create(self, *, model: str, input: list[str]) -> CreateEmbeddingResponse:
        self._owner.requests.append({"model": model, "input": input})
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": model,
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(len(text)), 0.5]}
                    for i, text in reversed(list(enumerate(input)))
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 0.0000002},
            }
        )


class FakeOpenRouter:
    def __init__(self) -> None:
        self.chunks: list = text_stream("Hello", " there")
        self.delay = 0.0
        self.error: Exception | None = None
        self.requests: list[dict[str, Any]] = []
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    async def close(self) -> None:
        return None
