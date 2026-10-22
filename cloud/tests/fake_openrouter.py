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


DECISION = {
    "model": "typesafe/jev-1.14-20261014",
    "answers": {
        "decision": {
            "type": "choice",
            "choice": "PASS_TO_GM",
            "probabilities": {},
            "confidence": 0.9,
        },
        "warning_type": {
            "type": "choice",
            "choice": "GHOST_ITEM",
            "probabilities": {},
            "confidence": 0.8,
        },
        "threat_level": {
            "type": "choice",
            "choice": "MEDIUM",
            "probabilities": {},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 800, "output_tokens": 120, "cost": 0.0000336},
    "provider": "TypeSafe",
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
        self.key_info: dict[str, Any] = {"data": {"limit": 5, "limit_remaining": 4.9}}
        self.key_error: Exception | None = None
        # The Decisions API (ADR 70): what `post` answers.
        self.decision: Any = DECISION
        self.decision_error: Exception | None = None

    async def post(self, path: str, *, cast_to: Any, body: dict[str, Any]) -> Any:
        self.requests.append({"path": path, **body})
        if self.decision_error is not None:
            raise self.decision_error
        return self.decision

    async def get(self, path: str, *, cast_to: Any) -> Any:
        assert path == "/key"
        if self.key_error is not None:
            raise self.key_error
        return self.key_info

    async def close(self) -> None:
        return None
