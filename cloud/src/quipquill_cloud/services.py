"""What the app runs on, built once in the lifespan (main.py) and handed to endpoints by
`get_services` — instead of untyped attributes on `app.state`."""

import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import HTTPConnection

from quipquill_cloud.core.config import Settings
from quipquill_cloud.core.ratelimit import RateLimiter
from quipquill_cloud.llm.prompts import PromptStore
from quipquill_cloud.mq.broker import Broker
from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.worker import CancelRegistry
from quipquill_cloud.stories import StoryCatalog


@dataclass
class Services:
    settings: Settings
    sessionmaker: async_sessionmaker[AsyncSession]
    llm_client: AsyncOpenAI
    rate_limiter: RateLimiter
    # The site's waitlist form and reports.
    form_limiter: RateLimiter
    dispatcher: EventDispatcher
    cancels: CancelRegistry
    prompts: PromptStore
    # New guest accounts per client IP (ADR 60).
    guest_limiter: RateLimiter
    stories: StoryCatalog
    # None until RabbitMQ is connected: REST works, /ws answers 1013 (main._start_queue).
    broker: Broker | None = None
    # The background reconnection while the broker is not up yet.
    mq_task: asyncio.Task[None] | None = None


def get_services(connection: HTTPConnection) -> Services:
    """Dependency for HTTP and WebSocket endpoints alike."""
    return connection.app.state.services
