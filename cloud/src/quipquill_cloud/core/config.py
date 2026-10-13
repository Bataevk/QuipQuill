from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# HS256 needs a key of at least 32 bytes (RFC 7518 §3.2).
MIN_JWT_SECRET_BYTES = 32
DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-change-me"


class Settings(BaseSettings):
    """Service configuration, read from the environment and `cloud/.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    database_url: str = "postgresql+asyncpg://quipquill:quipquill@localhost:5432/quipquill"
    rabbitmq_url: str = "amqp://quipquill:quipquill@localhost:5672/"
    queue_prefix: str = "qq.dev"
    mq_connect_timeout: float = 10.0
    # Workers run inside the API process (ADR 18); False leaves only the API and dispatcher.
    run_workers: bool = True
    worker_concurrency: int = 4
    # "openrouter" calls the model; "echo" streams the user text back (tests, offline dev).
    llm_handler: Literal["openrouter", "echo"] = "openrouter"
    echo_delay_seconds: float = 0.05
    llm_timeout_seconds: float = 120.0
    ws_heartbeat_interval: float = 2.0
    # How long in-flight and finished generations are kept for `resume`.
    generation_ttl_seconds: float = 900.0

    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_days: int = 30
    # Registration requires one of these codes (JSON list in env); empty = registration closed.
    invite_codes: list[str] = []

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Game calls: only providers that do not log prompts, Bedrock first (its cache works),
    # Vertex as the fallback (ADR 16, ADR 20).
    provider_order: list[str] = ["amazon-bedrock", "google-vertex/global"]
    embed_model: str = "openai/text-embedding-3-small"

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
