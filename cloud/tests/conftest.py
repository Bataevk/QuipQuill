"""Test fixtures. DB tests run against `quipquill_test` in the compose Postgres (make infra-up)."""

import asyncio
import faulthandler
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aio_pika
import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

CLOUD_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://quipquill:quipquill@localhost:5432/quipquill_test",
)
INVITE = "TEST-INVITE"
RABBITMQ_URL = os.environ.get("TEST_RABBITMQ_URL", "amqp://quipquill:quipquill@localhost:5672/")
# Each test run gets its own queues so it never interferes with a running dev server.
QUEUE_PREFIX = f"qq.test.{uuid.uuid4().hex[:8]}"
HANG_TIMEOUT_SECONDS = 60


def _dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


def db_run(sql: str, *args: Any) -> list[asyncpg.Record]:
    async def run() -> list[asyncpg.Record]:
        conn = await asyncpg.connect(_dsn(TEST_DATABASE_URL))
        try:
            return await conn.fetch(sql, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


async def _ensure_database() -> None:
    admin_dsn = _dsn(TEST_DATABASE_URL).rsplit("/", 1)[0] + "/postgres"
    name = TEST_DATABASE_URL.rsplit("/", 1)[1]
    conn = await asyncpg.connect(admin_dsn)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def migrated_db() -> str:
    try:
        asyncio.run(_ensure_database())
    except (OSError, asyncpg.PostgresError) as error:
        pytest.fail(f"Postgres is not reachable ({error}); start it with `make infra-up`")
    config = Config(str(CLOUD_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(CLOUD_DIR / "migrations"))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(config, "head")
    return TEST_DATABASE_URL


@pytest.fixture(autouse=True)
def _fail_on_hang() -> Iterator[None]:
    """WebSocket receives have no timeout; dump stacks and exit instead of hanging forever."""
    faulthandler.dump_traceback_later(HANG_TIMEOUT_SECONDS, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session", autouse=True)
def _delete_test_queues() -> Iterator[None]:
    yield

    async def delete() -> None:
        try:
            connection = await aio_pika.connect(RABBITMQ_URL, timeout=5)
        except Exception:
            return
        async with connection:
            channel = await connection.channel()
            for name in (f"{QUEUE_PREFIX}.commands", f"{QUEUE_PREFIX}.events"):
                await channel.queue_delete(name)

    asyncio.run(delete())


@pytest.fixture
def settings(migrated_db: str) -> Settings:
    db_run("TRUNCATE users, refresh_tokens, usage_events RESTART IDENTITY CASCADE")
    return Settings(
        _env_file=None,
        env="test",
        database_url=migrated_db,
        jwt_secret="test-secret-that-is-at-least-32-bytes-long",
        invite_codes=[INVITE],
        rabbitmq_url=RABBITMQ_URL,
        queue_prefix=QUEUE_PREFIX,
        llm_handler="echo",
        echo_delay_seconds=0.01,
        ws_heartbeat_interval=0.2,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@contextmanager
def ws_connect(client: TestClient, token: str) -> Iterator[Any]:
    """Open /ws and authenticate with the first message (ADR 22); yields the ready socket."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"v": 1, "type": "auth", "token": token})
        ready = ws.receive_json()
        assert ready == {"v": 1, "type": "ready"}, ready
        yield ws
