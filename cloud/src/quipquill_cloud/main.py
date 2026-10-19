import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from quipquill_cloud import __version__
from quipquill_cloud.api import auth, embeddings, health, me, ws
from quipquill_cloud.core import logs
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.core.ratelimit import RateLimiter
from quipquill_cloud.db.session import make_engine, make_sessionmaker
from quipquill_cloud.llm.handler import OpenRouterHandler
from quipquill_cloud.llm.prompts import PromptStore
from quipquill_cloud.mq.broker import Broker
from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.worker import CancelRegistry, CommandWorker, EchoHandler, Handler
from quipquill_cloud.services import Services

logger = logging.getLogger(__name__)


# Retry pauses while RabbitMQ is not up yet: doubling up to this.
MQ_RETRY_MAX_SECONDS = 30.0


async def _start_queue(services: Services) -> None:
    """Connect to RabbitMQ; on failure degrade (REST keeps working, /ws answers 1013) and keep
    trying in the background: after a reboot Docker starts the containers in no particular order,
    so the cloud may come up before the broker."""
    if not await _connect_queue(services):
        services.mq_task = asyncio.create_task(_keep_connecting(services))


async def _connect_queue(services: Services) -> bool:
    settings = services.settings
    try:
        broker = await Broker.connect(
            settings.rabbitmq_url, settings.queue_prefix, settings.mq_connect_timeout
        )
    except Exception as error:
        logger.warning("RabbitMQ unavailable (%s); WebSocket chat is disabled", error)
        return False
    await broker.start_dispatcher(services.dispatcher)
    if settings.run_workers:
        worker = CommandWorker(_handler(services), broker.publish_event, services.cancels)
        await broker.start_workers(worker, settings.worker_concurrency)
    services.broker = broker
    return True


async def _keep_connecting(services: Services) -> None:
    delay = services.settings.mq_retry_seconds
    while True:
        await asyncio.sleep(delay)
        if await _connect_queue(services):
            logger.info("RabbitMQ connected; WebSocket chat is enabled")
            return
        delay = min(delay * 2, MQ_RETRY_MAX_SECONDS)


def _handler(services: Services) -> Handler:
    settings = services.settings
    if settings.llm_handler == "echo":
        return EchoHandler(settings.echo_delay_seconds)
    return OpenRouterHandler(services.llm_client, services.sessionmaker, settings, services.prompts)


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
        logs.configure(settings.log_level, settings.log_format)
        engine = make_engine(settings.database_url)
        services = Services(
            settings=settings,
            sessionmaker=make_sessionmaker(engine),
            llm_client=llm_client or _llm_client(settings),
            rate_limiter=RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
            dispatcher=EventDispatcher(settings.generation_ttl_seconds),
            cancels=CancelRegistry(settings.generation_ttl_seconds),
            prompts=PromptStore(settings.prompts_dir),
        )
        app.state.services = services
        if settings.llm_handler == "openrouter" and settings.openrouter_api_key is None:
            logger.warning("OPENROUTER_API_KEY is not set: LLM calls will fail")
        if missing := services.prompts.missing():
            logger.warning("Prompts missing in %s: %s", settings.prompts_dir, ", ".join(missing))
        await _start_queue(services)
        yield
        if services.mq_task is not None:
            services.mq_task.cancel()
        if services.broker is not None:
            await services.broker.close()
        await services.llm_client.close()
        await engine.dispose()

    app = FastAPI(title="QuipQuill Cloud", version=__version__, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """X-Request-ID in and out (log correlation) and a cap on the request body size."""
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        logs.request_id.set(rid[:64])
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_http_body_bytes:
            response: Response = JSONResponse({"detail": "request_too_large"}, status_code=413)
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = rid[:64]
        return response

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(ws.router)
    app.include_router(embeddings.router)
    return app


app = create_app()
