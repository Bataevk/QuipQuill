"""POST /embeddings — vectors for lore search at play time (same model as the story file)."""

import uuid
from typing import Annotated

import openai
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.llm import usage
from quipquill_cloud.services import Services, get_services

router = APIRouter(tags=["llm"])

MAX_INPUTS = 64
MAX_INPUT_CHARS = 8000


class EmbeddingsIn(BaseModel):
    model: str
    input: list[Annotated[str, Field(min_length=1, max_length=MAX_INPUT_CHARS)]] = Field(
        min_length=1, max_length=MAX_INPUTS
    )


class EmbeddingsOut(BaseModel):
    model: str
    vectors: list[list[float]]
    usage: dict[str, float | int]


@router.post("/embeddings")
async def embeddings(
    body: EmbeddingsIn,
    user: CurrentUser,
    services: Annotated[Services, Depends(get_services)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingsOut:
    if body.model != settings.embed_model:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="model_not_allowed")
    if not services.rate_limiter.allow(str(user.id)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limited")
    sessionmaker = services.sessionmaker
    reached = await usage.limit_reached(
        sessionmaker,
        user.id,
        user.tier,
        settings.daily_cost_limit_usd,
        settings.guest_total_cost_limit_usd,
    )
    if reached:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=reached)
    try:
        response = await services.llm_client.embeddings.create(model=body.model, input=body.input)
    except openai.APIError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="provider_error") from error

    data = response.model_dump()
    vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
    reported = data.get("usage") or {}
    spent = {"input": reported.get("prompt_tokens") or 0, "costUsd": reported.get("cost") or 0}
    await usage.record(
        sessionmaker,
        user_id=user.id,
        request_id=uuid.uuid4().hex,
        session_id=None,
        purpose="embed",
        model=body.model,
        input_tokens=spent["input"],
        output_tokens=0,
        cache_read_tokens=0,
        cost_usd=spent["costUsd"],
    )
    return EmbeddingsOut(model=body.model, vectors=vectors, usage=spent)
