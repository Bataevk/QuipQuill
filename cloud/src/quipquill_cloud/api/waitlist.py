"""The release waitlist (ADR 66): a player leaves an email to hear when foolore ships."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from quipquill_cloud.api.auth import client_ip
from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.db.models import WaitlistEntry
from quipquill_cloud.db.session import get_session
from quipquill_cloud.services import Services, get_services

router = APIRouter(tags=["waitlist"])


class WaitlistIn(BaseModel):
    email: EmailStr
    # Where the player left it: the prototype limit, the profile or the onboarding.
    source: Literal["limit", "profile", "onboarding"]


@router.post("/waitlist", status_code=status.HTTP_204_NO_CONTENT)
async def join_waitlist(
    body: WaitlistIn,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Idempotent: the same email again changes nothing (the first source and time stay)."""
    await session.execute(
        insert(WaitlistEntry)
        .values(email=body.email.lower(), user_id=user.id, source=body.source)
        .on_conflict_do_nothing(index_elements=["email"])
    )
    await session.commit()


class SiteWaitlistIn(BaseModel):
    email: EmailStr


@router.post("/waitlist/site", status_code=status.HTTP_204_NO_CONTENT)
async def join_waitlist_from_site(
    body: SiteWaitlistIn,
    request: Request,
    services: Annotated[Services, Depends(get_services)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """The landing page's form: no account, so limited per client address. Idempotent too."""
    if not services.form_limiter.allow(f"waitlist:{client_ip(request)}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limited")
    await session.execute(
        insert(WaitlistEntry)
        .values(email=body.email.lower(), user_id=None, source="site")
        .on_conflict_do_nothing(index_elements=["email"])
    )
    await session.commit()
