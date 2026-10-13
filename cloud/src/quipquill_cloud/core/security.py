"""Passwords (argon2 via pwdlib), JWT access tokens, opaque refresh tokens."""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash

from quipquill_cloud.core.config import Settings

JWT_ALGORITHM = "HS256"
_password_hash = PasswordHash.recommended()
# Verified against when the user does not exist, so both paths cost the same time.
_DUMMY_HASH = _password_hash.hash("dummy-password-for-timing")


class InvalidToken(Exception):
    pass


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if password_hash is None:
        _password_hash.verify(password, _DUMMY_HASH)
        return False
    return _password_hash.verify(password, password_hash)


def create_access_token(user_id: uuid.UUID, tier: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "tier": tier,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(seconds=settings.access_token_ttl_seconds),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp", "type"]},
        )
    except jwt.PyJWTError as error:
        raise InvalidToken(str(error)) from error
    if claims.get("type") != "access":
        raise InvalidToken("not an access token")
    return claims


def new_refresh_token() -> tuple[str, str]:
    """(raw token for the client, hash for the database)."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def invite_code_valid(code: str, settings: Settings) -> bool:
    return any(secrets.compare_digest(code, valid) for valid in settings.invite_codes)
