"""The release waitlist (ADR 66): a player leaves an email to hear when foolore ships."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.db.models import WaitlistEntry
from quipquill_cloud.db.session import get_session

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
