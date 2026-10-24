from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quipquill_cloud.llm.policy import PURPOSES, Effort, Route

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

    # Requests to the models per user (chat over /ws, /embeddings); 0 = no limit, the default
    # (owner, 2026-10-24): one turn is many calls, and the spend caps already hold the cost.
    rate_limit_per_minute: float = 0
    rate_limit_burst: int = 10
    # Forms without a model behind them: the site's waitlist (per IP) and reports (per user).
    form_rate_limit_per_minute: float = 10
    form_rate_limit_burst: int = 5
    # Overrides the tier's spend cap per UTC day for everyone (local test runs, demos);
    # unset = each tier's own cap (llm/policy.py). The product cap stays in code.
    daily_cost_limit_usd: Decimal | None = None
    # The prototype's limit for a guest, for all time (ADR 66): ~30 turns. Email accounts (the
    # team's) are not held by it. Unset = no such limit.
    guest_total_cost_limit_usd: Decimal | None = Decimal("0.10")
    max_http_body_bytes: int = 1_000_000
    # Browser origins of the web version served elsewhere (JSON list in env). The web build is
    # served by the cloud's own host (ADR 69), so production needs none; in dev every localhost
    # port is allowed besides (Expo web's dev server is not the cloud).
    cors_origins: list[str] = []
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
    # Decision models (ADR 70): OpenRouter's Decisions API, outside the chat API's /v1.
    openrouter_decisions_url: str = "https://openrouter.ai/api/alpha/decisions"
    # A decision is a short call without streaming: a tighter limit than the chat models'.
    decision_timeout_seconds: float = 15.0
    # Overrides the chat model of every route, e.g. to try another model; empty = each route's
    # own model (llm/policy.py, ADR 50).
    llm_model: str = ""
    # The chat model of single purposes (JSON object in env), over llm_model, e.g.
    # {"generator": "anthropic/claude-haiku-5.5"} (ADR 82). Such a model keeps the route's
    # providers from the code whatever provider_order says: a model chosen for one purpose
    # is chosen with its providers.
    llm_model_by_purpose: dict[str, str] = {}
    # Overrides the providers of every route (JSON list in env), e.g. while one is down;
    # unset = each route's own order (ADR 50); [] = no restriction, OpenRouter picks.
    provider_order: list[str] | None = None
    # A spare model when the main one fails before its first token (error or timeout);
    # empty = no spare. Its providers: unset = the route's own from the code, [] = any.
    llm_fallback_model: str = ""
    llm_fallback_providers: list[str] | None = None
    # Reasoning effort per purpose (JSON object in env, e.g. {"gm": "high"}); a purpose not
    # listed keeps the code's (llm/policy.py). The spare's: unset = the same as the main model's.
    llm_effort: dict[str, Effort] = {}
    llm_fallback_effort: dict[str, Effort] | None = None
    embed_model: str = "openai/text-embedding-3-small"
    # The LLM prompt templates `<name>.md` (llm/prompts.py, ADR 57).
    prompts_dir: Path = REPO_PROMPTS_DIR
    # The story catalogue: every `*.qqstory` under this folder (stories.py, ADR 61).
    stories_dir: Path = REPO_STORIES_DIR

    def override(self, chosen: Route, purpose: str = "") -> Route:
        """The route with the model and providers from the config, where they are set."""
        own = self.llm_model_by_purpose.get(purpose)
        if own:
            return Route(model=own, providers=chosen.providers)
        return Route(
            model=self.llm_model or chosen.model,
            providers=chosen.providers
            if self.provider_order is None
            else tuple(self.provider_order),
        )

    def effort(self, purpose: str, *, spare: bool = False) -> Effort:
        """Reasoning effort of a purpose for the main model or the spare (ADR 75)."""
        main = self.llm_effort.get(purpose, PURPOSES[purpose].effort)
        if spare and self.llm_fallback_effort is not None:
            return self.llm_fallback_effort.get(purpose, main)
        return main

    def fallback(self, chosen: Route) -> Route | None:
        """The spare route for `chosen` (as in the code, before `override`), if one is set."""
        if not self.llm_fallback_model:
            return None
        providers = self.llm_fallback_providers
        return Route(
            model=self.llm_fallback_model,
            providers=chosen.providers if providers is None else tuple(providers),
        )

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
