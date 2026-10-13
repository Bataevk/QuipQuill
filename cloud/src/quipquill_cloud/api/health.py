"""Liveness (/health) and dependency checks (/health/db, /health/mq, /health/llm).

A dependency check answers 200 when the dependency works and 503 when it does not, so a
deployment probe or a human can tell which part is down.
"""

import asyncio
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from quipquill_cloud import __version__
from quipquill_cloud.core.config import Settings, get_settings

router = APIRouter(tags=["health"])

CHECK_TIMEOUT_SECONDS = 5.0
# Below this OpenRouter credit the gateway still works but someone should top it up.
LOW_CREDIT_USD = 0.5


class Health(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    env: str


class Check(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None
    info: dict[str, Any] | None = None


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> Health:
    """Liveness only: the process answers."""
    return Health(status="ok", service="quipquill-cloud", version=__version__, env=settings.env)


def _answer(response: Response, check: Check) -> Check:
    if check.status == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return check


@router.get("/health/db")
async def health_db(request: Request, response: Response) -> Check:
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            async with request.app.state.sessionmaker() as session:
                await session.execute(text("SELECT 1"))
    except Exception as error:
        return _answer(response, Check(status="down", detail=type(error).__name__))
    return Check(status="ok")


@router.get("/health/mq")
async def health_mq(request: Request, response: Response) -> Check:
    broker = getattr(request.app.state, "broker", None)
    if broker is None:
        return _answer(response, Check(status="down", detail="not_connected_at_startup"))
    if not await broker.ping(CHECK_TIMEOUT_SECONDS / 2):
        return _answer(response, Check(status="down", detail="broker_unreachable"))
    return Check(status="ok")


@router.get("/health/llm")
async def health_llm(request: Request, response: Response) -> Check:
    """OpenRouter reachable and the key valid (free call); reports the remaining credit."""
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            data = await request.app.state.llm_client.get("/key", cast_to=object)
    except Exception as error:
        return _answer(response, Check(status="down", detail=type(error).__name__))
    key = data.get("data", {}) if isinstance(data, dict) else {}
    remaining = key.get("limit_remaining")
    info = {"limitRemainingUsd": remaining, "limitUsd": key.get("limit")}
    if remaining is not None and remaining < LOW_CREDIT_USD:
        return Check(status="degraded", detail="low_credit", info=info)
    return Check(status="ok", info=info)
