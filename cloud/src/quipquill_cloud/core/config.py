from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository's prompts/ (development); production sets PROMPTS_DIR (ADR 57).
REPO_PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts"
# The repository's stories/ (development); production sets STORIES_DIR (ADR 61).
REPO_STORIES_DIR = Path(__file__).resolve().parents[4] / "stories"

# HS256 needs a key of at least 32 bytes (RFC 7518 §3.2).
MIN_JWT_SECRET_BYTES = 32
DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-change-me"


class Settings(BaseSettings):
    """Service configuration, read from the environment and `cloud/.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "info"
    # "text" reads well in a terminal; containers use "json" (docker-compose.yml).
    log_format: Literal["text", "json"] = "text"

    # Abuse limits on top of the daily spend cap (owner, 2026-10-14).
    rate_limit_per_minute: float = 30
    rate_limit_burst: int = 10
    # Overrides the tier's spend cap per UTC day for everyone (local test runs, demos);
    # unset = each tier's own cap (llm/policy.py). The product cap stays in code.
    daily_cost_limit_usd: Decimal | None = None
    # The prototype's limit for a guest, for all time (ADR 66): ~30 turns. Email accounts (the
    # team's) are not held by it. Unset = no such limit.
    guest_total_cost_limit_usd: Decimal | None = Decimal("0.10")
    max_http_body_bytes: int = 1_000_000
    # Rough guard (chars / 4) keeping a request under Haiku's 100K-token price step.
    max_request_tokens: int = 95_000
    max_tools: int = 32

    database_url: str = "postgresql+asyncpg://quipquill:quipquill@localhost:5432/quipquill"
    rabbitmq_url: str = "amqp://quipquill:quipquill@localhost:5672/"
    queue_prefix: str = "qq.dev"
    mq_connect_timeout: float = 10.0
    # First pause before connecting again when RabbitMQ is down at startup (then doubling to 30 s).
    mq_retry_seconds: float = 2.0
    # Workers run inside the API process (ADR 18); False leaves only the API and dispatcher.
    run_workers: bool = True
    worker_concurrency: int = 4
    # "openrouter" calls the model; "echo" streams the user text back (tests, offline dev).
    llm_handler: Literal["openrouter", "echo"] = "openrouter"
    echo_delay_seconds: float = 0.05
    llm_timeout_seconds: float = 120.0
    ws_heartbeat_interval: float = 2.0
    # The first WebSocket message must be `auth` within this time (ADR 22).
    ws_auth_timeout: float = 10.0
    # How long in-flight and finished generations are kept for `resume`.
    generation_ttl_seconds: float = 900.0

    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_days: int = 30
    # Registration requires one of these codes (JSON list in env); empty = registration closed.
    invite_codes: list[str] = []
    # A guest account per device, no email or password (ADR 60); off unless the env opens it.
    guest_signup: bool = False
    # New guests per client IP: on average per minute, at most `burst` in a row.
    guest_signup_per_minute: float = 2
    guest_signup_burst: int = 5

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Overrides the providers of every route (JSON list in env), e.g. while one is down;
    # empty = each route's own order (llm/policy.py, ADR 50).
    provider_order: list[str] = []
    embed_model: str = "openai/text-embedding-3-small"
    # The LLM prompt templates `<name>.md` (llm/prompts.py, ADR 57).
    prompts_dir: Path = REPO_PROMPTS_DIR
    # The story catalogue: every `*.qqstory` under this folder (stories.py, ADR 61).
    stories_dir: Path = REPO_STORIES_DIR

    @model_validator(mode="after")
    def _require_real_secrets_in_prod(self) -> "Settings":
        secret = self.jwt_secret.get_secret_value()
        if self.env == "prod" and (
            secret == DEV_JWT_SECRET or len(secret.encode()) < MIN_JWT_SECRET_BYTES
        ):
            raise ValueError(
                f"JWT_SECRET must be a real secret of at least {MIN_JWT_SECRET_BYTES} bytes "
                "when ENV=prod"
            )
        if self.env == "prod" and self.llm_handler == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when ENV=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
