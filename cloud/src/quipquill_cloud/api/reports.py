"""POST /reports — the game's Report button: what the player says, the hero and the transcript."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.db.models import Report
from quipquill_cloud.db.session import get_session
from quipquill_cloud.services import Services, get_services

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])

MAX_MESSAGE_CHARS = 4000
MAX_LINES = 2000
MAX_LINE_CHARS = 20_000
# The whole body: a long game is a few hundred KB of text.
MAX_BODY_BYTES = 2_000_000


class ReportLine(BaseModel):
    """One transcript line as the player saw it."""

    seq: int
    speaker: Annotated[str, Field(max_length=128)]
    text: Annotated[str, Field(max_length=MAX_LINE_CHARS)]
    ts: Annotated[str, Field(max_length=64)] = ""


class ReportIn(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]
    story_id: Annotated[str, Field(max_length=64)] | None = None
    save_id: Annotated[str, Field(max_length=64)] | None = None
    character: dict[str, Any] | None = None
    transcript: Annotated[list[ReportLine], Field(max_length=MAX_LINES)] = []
    # The app's platform and version.
    client: dict[str, Any] | None = None


class ReportOut(BaseModel):
    id: int


@router.post("/reports", status_code=status.HTTP_201_CREATED)
async def send_report(
    request: Request,
    user: CurrentUser,
    services: Annotated[Services, Depends(get_services)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ReportOut:
    """Stored as sent; the team reads them from the database."""
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="report_too_large")
    try:
        body = ReportIn.model_validate_json(raw)
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    if not services.form_limiter.allow(f"report:{user.id}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limited")
    report = Report(
        user_id=user.id,
        message=body.message.strip(),
        story_id=body.story_id,
        save_id=body.save_id,
        character=body.character,
        transcript=[line.model_dump() for line in body.transcript],
        client=body.client,
    )
    session.add(report)
    await session.commit()
    logger.info("report %s from %s (%s lines)", report.id, user.id, len(body.transcript))
    return ReportOut(id=report.id)
