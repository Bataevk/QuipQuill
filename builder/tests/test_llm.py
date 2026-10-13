import asyncio
import json

import httpx2
import pytest

from quipquill_builder.config import Settings
from quipquill_builder.extractor import ExtractionError, ExtractionParams, GraphExtractor
from quipquill_builder.llm import BatchError, OpenRouterBatchClient, OpenRouterChatClient
from quipquill_builder.text import load_documents

SETTINGS = Settings(_env_file=None, openrouter_api_key="test-key", batch_poll_interval=0)


def completion(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class FakeBatchApi:
    """In-memory stand-in for POST /batches + GET /batches/:id."""

    def __init__(self, final_status: str = "completed", polls_before_done: int = 2) -> None:
        self.final_status = final_status
        self.polls_left = polls_before_done
        self.submitted: dict | None = None
        self.raw_body = b""

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            self.raw_body = request.content
            self.submitted = json.loads(request.content)
            return httpx2.Response(202, json={"id": "batch-1", "status": "validating"})
        if self.polls_left > 0:
            self.polls_left -= 1
            return httpx2.Response(200, json={"id": "batch-1", "status": "in_progress"})
        results = []
        for item in self.submitted["requests"]:
            user = item["body"]["messages"][1]["content"]
            if "fail" in user:
                results.append({"custom_id": item["custom_id"], "error": {"message": "boom"}})
            elif "wrapped" in user:
                results.append(
                    {
                        "custom_id": item["custom_id"],
                        "response": {"status_code": 200, "body": completion("W:" + user)},
                    }
                )
            else:
                results.append(
                    {"custom_id": item["custom_id"], "response": completion("R:" + user)}
                )
        return httpx2.Response(
            200, json={"id": "batch-1", "status": self.final_status, "results": results[::-1]}
        )


def make_client(api: FakeBatchApi) -> OpenRouterBatchClient:
    async def no_sleep(_: float) -> None:
        return None

    return OpenRouterBatchClient(
        SETTINGS,
        "anthropic/claude-haiku-5.5",
        transport=httpx2.MockTransport(api.handler),
        sleep=no_sleep,
    )


def test_batch_submits_once_and_maps_results_by_id() -> None:
    api = FakeBatchApi()
    results = asyncio.run(
        make_client(api).complete_many([("sys", "one"), ("sys", "fail me"), ("sys", "wrapped")])
    )

    assert results[0] == "R:one"
    assert isinstance(results[1], BatchError)
    assert results[2] == "W:wrapped"

    body = api.submitted
    assert body["endpoint"] == "/v1/chat/completions"
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert list(body)[-1] == "requests"  # the API requires `requests` last
    first = body["requests"][0]["body"]
    assert first["reasoning"] == {"effort": "low"}
    assert "temperature" not in first
    assert "provider" not in body  # batches run on Anthropic's :batch endpoint


@pytest.mark.parametrize("status", ["failed", "expired"])
def test_batch_failure_raises(status: str) -> None:
    with pytest.raises(BatchError, match=status):
        asyncio.run(make_client(FakeBatchApi(final_status=status)).complete_many([("s", "u")]))


def test_extractor_sends_all_chunks_in_one_batch(lore_dir) -> None:
    api = FakeBatchApi(polls_before_done=0)
    extractor = GraphExtractor(
        make_client(api), params=ExtractionParams(chunk_size=600, chunk_overlap=0)
    )
    # The fake replies are not JSON, so extraction fails — what matters is the single submit.
    with pytest.raises(ExtractionError):
        asyncio.run(extractor.extract(load_documents(lore_dir)))

    assert len(api.submitted["requests"]) == 2


def test_sync_client_routes_only_to_allowed_providers() -> None:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                **completion("hi"),
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
            },
        )

    client = OpenRouterChatClient(SETTINGS, "anthropic/claude-haiku-5.5")
    client._client = client._client.with_options(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    assert asyncio.run(client.complete("sys", "user")) == "hi"
    assert seen["provider"] == {
        "only": ["amazon-bedrock", "google-vertex/global"],
        "allow_fallbacks": False,
    }
    assert seen["reasoning"] == {"effort": "low"}
    assert "temperature" not in seen


def test_batch_embedder_splits_into_requests_of_64() -> None:
    from quipquill_builder.embeddings import OpenRouterBatchEmbedder

    submitted: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            submitted.update(json.loads(request.content))
            return httpx2.Response(202, json={"id": "b", "status": "validating"})
        results = []
        for item in submitted["requests"]:
            texts = item["body"]["input"]
            data = [{"index": i, "embedding": [float(len(t)), 0.0]} for i, t in enumerate(texts)]
            results.append({"custom_id": item["custom_id"], "response": {"data": data[::-1]}})
        return httpx2.Response(200, json={"id": "b", "status": "completed", "results": results})

    embedder = OpenRouterBatchEmbedder(
        SETTINGS, "openai/text-embedding-3-small", transport=httpx2.MockTransport(handler)
    )
    texts = ["x" * n for n in range(1, 101)]
    vectors = asyncio.run(embedder.embed(texts))

    assert submitted["endpoint"] == "/v1/embeddings"
    assert [len(r["body"]["input"]) for r in submitted["requests"]] == [64, 36]
    assert [v[0] for v in vectors] == [float(n) for n in range(1, 101)]
