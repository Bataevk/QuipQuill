from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Builder configuration, read from the environment and `builder/.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    extract_model: str = "anthropic/claude-haiku-5.5"
    # Summarization merges duplicate descriptions; defaults to the extraction model.
    summarize_model: str | None = None
    embed_model: str = "openai/text-embedding-3-small"

    # Unset by default: Claude 5.x models reject non-default sampling parameters.
    temperature: float | None = None
    max_tokens: int = 8192
    request_timeout: float = 120.0
    max_retries: int = 2
    # OpenRouter unified reasoning control; None leaves the provider default.
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    # Sync chat calls only go to providers that do not log prompts (owner's requirement).
    # Batch requests run on Anthropic's :batch endpoint and ignore this list.
    provider_only: list[str] = ["amazon-bedrock", "google-vertex/global"]

    # Graph building uses OpenRouter's Batch API by default (50% price, minutes to hours).
    use_batch: bool = True
    batch_poll_interval: float = 30.0
    batch_timeout: float = 6 * 3600.0

    chunk_size: int = 2000
    chunk_overlap: int = 200
    merge_threshold: float = 0.85
    max_concurrency: int = 4
