from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from quipquill_cloud import __version__
from quipquill_cloud.api import auth, health, me
from quipquill_cloud.core.config import Settings, get_settings
from quipquill_cloud.db.session import make_engine, make_sessionmaker


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        yield
        await engine.dispose()

    app = FastAPI(title="QuipQuill Cloud", version=__version__, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)
    return app


app = create_app()
