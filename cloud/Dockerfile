# QuipQuill cloud: FastAPI + in-process workers (ADR 18). Build context: cloud/.

# --- build: resolve and install everything into /app/.venv ---------------------------------------
FROM python:3.13.16-slim-bookworm AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first: this layer is reused until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
# The project itself is installed into site-packages (no editable link to /app/src).
RUN uv sync --frozen --no-dev --no-editable

# --- runtime: just the virtualenv and the migrations ---------------------------------------------
FROM python:3.13.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations

# The app only reads /app; files stay owned by root.
RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# One process: the event dispatcher and resume buffer live in memory (ADR 18).
CMD ["uvicorn", "quipquill_cloud.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
