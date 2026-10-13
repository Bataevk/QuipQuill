"""OpenRouter Batch API: submit many requests as one job, poll until it finishes.

Half the per-token price; results arrive within a 24h window (minutes to hours in practice).
Docs: https://openrouter.ai/docs/projects/docs/batch-quickstart
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2

from quipquill_builder.config import Settings

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("completed", "failed", "expired", "cancelled")
# A freshly submitted batch can return 404 on GET for a short while (observed live).
NOT_FOUND_GRACE_SECONDS = 120.0
NOT_FOUND_RETRY_SECONDS = 5.0


class BatchError(RuntimeError):
    pass


def api_key(settings: Settings) -> str:
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see builder/.env.example)")
    return settings.openrouter_api_key.get_secret_value()


class BatchRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._key = api_key(settings)
        self._transport = transport
        self._sleep = sleep

    async def run(
        self,
        endpoint: str,
        model: str,
        bodies: list[dict[str, Any]],
        *,
        resume_id: str | None = None,
    ) -> list[dict[str, Any] | Exception]:
        """Response body for each request (in order), or the error for that request.

        `resume_id` attaches to a batch submitted earlier with the same requests instead of
        submitting (and paying for) a new one.
        """
        if not bodies:
            return []
        payload = {
            "endpoint": endpoint,
            "model": model,
            # `requests` must be the last key in the body (the API rejects it otherwise).
            "requests": [{"custom_id": f"r{i}", "body": body} for i, body in enumerate(bodies)],
        }
        async with httpx2.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._settings.request_timeout,
            transport=self._transport,
        ) as http:
            if resume_id:
                batch_id = resume_id
                logger.info("Resuming batch %s (%d requests expected)", batch_id, len(bodies))
            else:
                response = await http.post("/batches", json=payload)
                if response.status_code != 202:
                    raise BatchError(f"Batch submit failed: {response.status_code} {response.text}")
                batch_id = response.json()["id"]
                logger.info(
                    "Batch %s submitted: %d requests to %s", batch_id, len(bodies), endpoint
                )
            batch = await self._wait(http, batch_id)

        total = (batch.get("request_counts") or {}).get("total")
        if total is not None and total != len(bodies):
            raise BatchError(
                f"Batch {batch_id} has {total} requests, expected {len(bodies)}: "
                "it was submitted for different input"
            )

        if batch["status"] != "completed":
            raise BatchError(f"Batch {batch_id} ended as {batch['status']}: {batch.get('error')}")
        by_id = {item["custom_id"]: item for item in batch.get("results") or []}
        return [_response_body(by_id.get(f"r{i}")) for i in range(len(bodies))]

    async def _wait(self, http: httpx2.AsyncClient, batch_id: str) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            response = await http.get(f"/batches/{batch_id}")
            elapsed = time.monotonic() - started
            if response.status_code == 404 and elapsed < NOT_FOUND_GRACE_SECONDS:
                await self._sleep(NOT_FOUND_RETRY_SECONDS)
                continue
            response.raise_for_status()
            batch = response.json()
            status = batch.get("status")
            if status in TERMINAL_STATUSES:
                return batch
            if elapsed > self._settings.batch_timeout:
                raise BatchError(
                    f"Batch {batch_id} still {status} after {elapsed:.0f}s; "
                    "it keeps running on OpenRouter and can be resumed by its id"
                )
            logger.info(
                "Batch %s: %s %s (%.0fs)", batch_id, status, batch.get("request_counts"), elapsed
            )
            await self._sleep(self._settings.batch_poll_interval)


def _response_body(item: dict[str, Any] | None) -> dict[str, Any] | Exception:
    if item is None:
        return BatchError("No result returned for this request")
    if item.get("error"):
        return BatchError(f"Request failed: {item['error']}")
    response = item.get("response")
    if not isinstance(response, dict):
        return BatchError(f"Unexpected result shape: {str(item)[:200]}")
    body = response.get("body", response)  # tolerate a {status_code, body} wrapper
    return body if isinstance(body, dict) else BatchError(f"Unexpected body: {str(body)[:200]}")
