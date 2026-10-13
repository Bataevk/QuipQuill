"""Logging: one JSON object per line in stdout, with the request context attached.

`request_id`, `session_id` and `user_id` live in contextvars: set them once (HTTP middleware,
WebSocket message, queue command) and every log line below carries them, including lines from
libraries (uvicorn, aio-pika, openai).
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
user_id: ContextVar[str | None] = ContextVar("user_id", default=None)

_CONTEXT = {"requestId": request_id, "sessionId": session_id, "userId": user_id}
_HANDLER_NAME = "quipquill"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, var in _CONTEXT.items():
            value = var.get()
            if value:
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Readable lines for local development; context appended in brackets."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        context = " ".join(f"{k}={v.get()}" for k, v in _CONTEXT.items() if v.get())
        return f"{line} [{context}]" if context else line


def configure(level: str, fmt: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.set_name(_HANDLER_NAME)
    root = logging.getLogger()
    # Replace only our own handler, so others (e.g. pytest's capture) keep working.
    root.handlers[:] = [h for h in root.handlers if h.get_name() != _HANDLER_NAME] + [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    # One line per HTTP request to OpenRouter is noise at INFO.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
