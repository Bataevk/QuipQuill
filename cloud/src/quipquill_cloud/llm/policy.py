"""What the server sends to the model. The client names a route ("basic") and a purpose; the
server chooses model, providers, effort and max_tokens."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

HAIKU = "anthropic/claude-haiku-5.5"


Effort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class CallParams:
    effort: Effort
    # The call's limit: reasoning and the answer together (ADR 76).
    max_tokens: int


# Owner's decision (2026-10-14, raised 2026-10-24, ADR 74): the game master thinks at "high",
# the actor (npc) and the generator at "medium", everything else at "low". The config may
# override effort per purpose (LLM_EFFORT, LLM_FALLBACK_EFFORT; ADR 75).
PURPOSES: dict[str, CallParams] = {
    "gm": CallParams(effort="high", max_tokens=6000),
    "npc": CallParams(effort="medium", max_tokens=1600),
    "summary": CallParams(effort="low", max_tokens=3000),
    "character": CallParams(effort="low", max_tokens=3000),
    # Stage 6.2d (ADR 42): the background world generator after each turn, a JSON list of changes.
    # ADR 55: up to ten expected additions a turn, so room for a long answer.
    "generator": CallParams(effort="medium", max_tokens=6000),
    # Stage T.6 (ADR 70): a short in-story answer to a rejected input; no tools, no world.
    "mini_gm": CallParams(effort="low", max_tokens=300),
    # Stage T.8: before the master — drafts of the place and the player's unsupported claims.
    "pregen": CallParams(effort="low", max_tokens=800),
}


# Purposes answered by a decision model, not a chat model (ADR 70): typed choices with
# probabilities, no text. Called through OpenRouter's Decisions API; no prompt cache.
# The input check: TypeSafe Jev (owner, 2026-10-22: it does not log prompts; Luna Decisions does).
DECISIONS: dict[str, str] = {
    "guard": "typesafe/jev-1.14",
}


@dataclass(frozen=True)
class Route:
    """A model class the client names (`model: "basic"`): the server picks model and providers."""

    model: str
    # OpenRouter provider slugs, tried in this order and only these (ADR 50).
    providers: tuple[str, ...]


# ADR 51 (owner, 2026-10-18): Vertex global, then Bedrock. Vertex in Europe was dropped: in the
# live runs it was faster but not worth its price. Only providers that do not log prompts (ADR 16).
ROUTES: dict[str, Route] = {
    "basic": Route(model=HAIKU, providers=("google-vertex/global", "amazon-bedrock")),
}
DEFAULT_ROUTE = "basic"


def route(name: str) -> Route | None:
    return ROUTES.get(name)


@dataclass(frozen=True)
class Tier:
    # Spend cap per UTC day in USD (OpenRouter's reported cost); None = not limited.
    daily_cost_usd: Decimal | None


# Prototype: everyone is on "free" ($0.05/day). Caps (and routes) for pro/max are a product
# decision for later (backlog); until then they behave like free.
TIERS: dict[str, Tier] = {
    "free": Tier(daily_cost_usd=Decimal("0.05")),
    "pro": Tier(daily_cost_usd=Decimal("0.05")),
    "max": Tier(daily_cost_usd=Decimal("0.05")),
}


def tier(name: str) -> Tier:
    return TIERS.get(name, TIERS["free"])


def daily_cap(tier_name: str, override: Decimal | None = None) -> Decimal | None:
    """The spend cap per UTC day: `DAILY_COST_LIMIT_USD` if set, else the tier's own."""
    return override if override is not None else tier(tier_name).daily_cost_usd
