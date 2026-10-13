"""Email + password auth: register (invite only), login, refresh with rotation."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.core.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    invite_code_valid,
    new_refresh_token,
    verify_password,
)
from quipquill_cloud.db.models import RefreshToken, User
from quipquill_cloud.db.session import get_session

router = APIRouter(prefix="/auth", tags=["auth"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    invite_code: str = Field(min_length=1, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class RefreshIn(BaseModel):
    refresh_token: str = Field(max_length=256)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail)


async def _issue_tokens(session: AsyncSession, user: User, settings: Settings) -> TokenPair:
    raw, token_hash = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_ttl_days),
        )
    )
    await session.commit()
    return TokenPair(
        access_token=create_access_token(user.id, user.tier, settings),
        refresh_token=raw,
        expires_in=settings.access_token_ttl_seconds,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    if not invite_code_valid(body.invite_code, settings):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="invalid_invite_code")
    email = body.email.lower()
    if await session.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="email_taken")
    user = User(email=email, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as error:  # a concurrent registration won the race
        raise HTTPException(status.HTTP_409_CONFLICT, detail="email_taken") from error
    return await _issue_tokens(session, user, settings)


@router.post("/login")
async def login(body: LoginIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    user = await session.scalar(select(User).where(User.email == body.email.lower()))
    if not verify_password(body.password, user.password_hash if user else None) or not user:
        raise _unauthorized("invalid_credentials")
    return await _issue_tokens(session, user, settings)


@router.post("/refresh")
async def refresh(body: RefreshIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    """Exchange a refresh token for a new pair; the old token is revoked (rotation).

    Presenting an already revoked token means it leaked: every token of that user is revoked.
    """
    now = datetime.now(UTC)
    token = await session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(body.refresh_token))
        .with_for_update()
    )
    if token is None:
        raise _unauthorized("invalid_refresh_token")
    if token.revoked_at is not None:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == token.user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        await session.commit()
        raise _unauthorized("refresh_token_reused")
    if token.expires_at <= now:
        raise _unauthorized("refresh_token_expired")

    token.revoked_at = now
    user = await session.get(User, token.user_id)
    if user is None:
        raise _unauthorized("invalid_refresh_token")
    return await _issue_tokens(session, user, settings)
