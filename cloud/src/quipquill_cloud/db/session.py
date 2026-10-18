from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from quipquill_cloud.services import Services, get_services


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(
    services: Annotated[Services, Depends(get_services)],
) -> AsyncIterator[AsyncSession]:
    """Request-scoped session from the engine created in the app lifespan."""
    async with services.sessionmaker() as session:
        yield session
