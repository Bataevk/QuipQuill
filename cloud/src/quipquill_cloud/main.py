import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from quipquill_cloud import __version__
from quipquill_cloud.api import auth, embeddings, health, me, ws
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.db.session import make_engine, make_sessionmaker
from quipquill_cloud.llm.handler import OpenRouterHandler
from quipquill_cloud.mq.broker import Broker
from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.worker import CancelRegistry, CommandWorker, EchoHandler, Handler

logger = logging.getLogger(__name__)


async def _start_queue(app: FastAPI, settings: Settings) -> None:
    """Connect to RabbitMQ; on failure degrade (REST keeps working, /ws answers 1013)."""
    app.state.dispatcher = EventDispatcher(settings.generation_ttl_seconds)
    app.state.cancels = CancelRegistry()
    app.state.broker = None
    try:
        broker = await Broker.connect(
            settings.rabbitmq_url, settings.queue_prefix, settings.mq_connect_timeout
        )
    except Exception as error:
        logger.warning("RabbitMQ unavailable (%s); WebSocket chat is disabled", error)
        return
    await broker.start_dispatcher(app.state.dispatcher)
    if settings.run_workers:
        worker = CommandWorker(_handler(app, settings), broker.publish_event, app.state.cancels)
        await broker.start_workers(worker, settings.worker_concurrency)
    app.state.broker = broker


def _handler(app: FastAPI, settings: Settings) -> Handler:
    if settings.llm_handler == "echo":
        return EchoHandler(settings.echo_delay_seconds)
    return OpenRouterHandler(app.state.llm_client, app.state.sessionmaker, settings)


def _llm_client(settings: Settings) -> AsyncOpenAI:
    key = settings.openrouter_api_key
    return AsyncOpenAI(
        # Without a key the client still builds; calls then fail as provider errors (dev only).
        api_key=key.get_secret_value() if key else "missing-openrouter-key",
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )


def create_app(settings: Settings | None = None, llm_client: AsyncOpenAI | None = None) -> FastAPI:
    """`llm_client` replaces the OpenRouter client (tests pass a fake)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.llm_client = llm_client or _llm_client(settings)
        if settings.llm_handler == "openrouter" and settings.openrouter_api_key is None:
            logger.warning("OPENROUTER_API_KEY is not set: LLM calls will fail")
        await _start_queue(app, settings)
        yield
        if app.state.broker is not None:
            await app.state.broker.close()
        await app.state.llm_client.close()
        await engine.dispose()

    app = FastAPI(title="QuipQuill Cloud", version=__version__, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(ws.router)
    app.include_router(embeddings.router)
    return app


app = create_app()
