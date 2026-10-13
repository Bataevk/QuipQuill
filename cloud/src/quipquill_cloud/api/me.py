import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.db.models import UsageEvent
from quipquill_cloud.db.session import get_session

router = APIRouter(tags=["account"])

COST_QUANTUM = Decimal("0.000001")  # matches usage_events.cost_usd NUMERIC(12, 6)


class UsageToday(BaseModel):
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


class Me(BaseModel):
    id: uuid.UUID
    email: str
    tier: str
    created_at: datetime
    usage_today: UsageToday  # UTC day; per-tier limits are added in stage 2.3


@router.get("/me")
async def me(user: CurrentUser, session: Annotated[AsyncSession, Depends(get_session)]) -> Me:
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    row = (
        await session.execute(
            select(
                func.count(UsageEvent.id),
                func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                func.coalesce(func.sum(UsageEvent.cost_usd), 0),
            ).where(UsageEvent.user_id == user.id, UsageEvent.created_at >= day_start)
        )
    ).one()
    return Me(
        id=user.id,
        email=user.email,
        tier=user.tier,
        created_at=user.created_at,
        usage_today=UsageToday(
            requests=row[0],
            input_tokens=row[1],
            output_tokens=row[2],
            cost_usd=Decimal(row[3]).quantize(COST_QUANTUM),
        ),
    )
