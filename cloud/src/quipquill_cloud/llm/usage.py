"""Spend accounting: every model or embeddings call is a row in usage_events."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quipquill_cloud.db.models import UsageEvent
from quipquill_cloud.llm.policy import daily_cap


def utc_day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def spent_today(session: AsyncSession, user_id: uuid.UUID) -> Decimal:
    total = await session.scalar(
        select(func.coalesce(func.sum(UsageEvent.cost_usd), 0)).where(
            UsageEvent.user_id == user_id, UsageEvent.created_at >= utc_day_start()
        )
    )
    return Decimal(total)


async def over_limit(
    sessionmaker: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
    tier_name: str,
    override: Decimal | None = None,
) -> bool:
    cap = daily_cap(tier_name, override)
    if cap is None:
        return False
    async with sessionmaker() as session:
        return await spent_today(session, user_id) >= cap


async def record(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    user_id: uuid.UUID,
    request_id: str,
    session_id: str | None,
    purpose: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cost_usd: float | Decimal,
) -> None:
    async with sessionmaker() as session:
        session.add(
            UsageEvent(
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                purpose=purpose,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cost_usd=Decimal(str(cost_usd)),
            )
        )
        await session.commit()
