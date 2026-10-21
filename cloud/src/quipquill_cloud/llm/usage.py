"""Spend accounting: every model or embeddings call is a row in usage_events."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quipquill_cloud.db.models import UsageEvent, User
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


async def spent_total(session: AsyncSession, user_id: uuid.UUID) -> Decimal:
    total = await session.scalar(
        select(func.coalesce(func.sum(UsageEvent.cost_usd), 0)).where(UsageEvent.user_id == user_id)
    )
    return Decimal(total)


# Error codes of a spent limit (protocol.md): the guest's prototype allowance (ADR 66) is spent
# for good; the daily cap comes back the next UTC day.
PROTOTYPE_LIMIT = "prototype_limit"
DAILY_LIMIT = "limit_exceeded"


async def limit_reached(
    sessionmaker: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
    tier_name: str,
    override: Decimal | None = None,
    guest_total: Decimal | None = None,
) -> str | None:
    """Which limit stops this user's next call, if any: `prototype_limit` or `limit_exceeded`."""
    cap = daily_cap(tier_name, override)
    async with sessionmaker() as session:
        if guest_total is not None:
            kind = await session.scalar(select(User.kind).where(User.id == user_id))
            if kind == "guest" and await spent_total(session, user_id) >= guest_total:
                return PROTOTYPE_LIMIT
        if cap is not None and await spent_today(session, user_id) >= cap:
            return DAILY_LIMIT
    return None


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
