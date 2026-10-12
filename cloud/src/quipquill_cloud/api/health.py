from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from quipquill_cloud import __version__
from quipquill_cloud.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    env: str


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> Health:
    """Liveness only; dependency checks (/health/db, /health/mq, /health/llm) come in 2.4."""
    return Health(status="ok", service="quipquill-cloud", version=__version__, env=settings.env)
