import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from quipquill_cloud import __version__
from quipquill_cloud.api import auth, health, me, ws
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.db.session import make_engine, make_sessionmaker
from quipquill_cloud.mq.broker import Broker
from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.worker import CancelRegistry, CommandWorker, EchoHandler

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
        handler = EchoHandler(settings.echo_delay_seconds)
        worker = CommandWorker(handler, broker.publish_event, app.state.cancels)
        await broker.start_workers(worker, settings.worker_concurrency)
    app.state.broker = broker


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        await _start_queue(app, settings)
        yield
        if app.state.broker is not None:
            await app.state.broker.close()
        await engine.dispose()

    app = FastAPI(title="QuipQuill Cloud", version=__version__, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(ws.router)
    return app


app = create_app()
