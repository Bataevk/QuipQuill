from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "change-me"


class Settings(BaseSettings):
    """Service configuration, read from the environment and `cloud/.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    database_url: str = "postgresql+asyncpg://quipquill:quipquill@localhost:5432/quipquill"
    rabbitmq_url: str = "amqp://quipquill:quipquill@localhost:5672/"
    queue_prefix: str = "qq.dev"

    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model_free: str = "anthropic/claude-haiku-5.5"
    embed_model: str = "openai/text-embedding-3-small"

    @model_validator(mode="after")
    def _require_real_secrets_in_prod(self) -> "Settings":
        if self.env == "prod" and self.jwt_secret.get_secret_value() == DEV_JWT_SECRET:
            raise ValueError("JWT_SECRET must be set to a real secret when ENV=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
