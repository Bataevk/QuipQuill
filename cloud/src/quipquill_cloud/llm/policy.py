"""What the server sends to the model. The client never chooses model, effort or max_tokens."""

from dataclasses import dataclass
from decimal import Decimal

HAIKU = "anthropic/claude-haiku-5.5"


@dataclass(frozen=True)
class CallParams:
    effort: str
    max_tokens: int


# Owner's decision (2026-10-14): the game master thinks at "medium", everything else at "low".
PURPOSES: dict[str, CallParams] = {
    "gm": CallParams(effort="medium", max_tokens=2000),
    "npc": CallParams(effort="low", max_tokens=800),
    # Stage 6.2k (ADR 48): the verdict also lists the place's expected gaps (drafts).
    "guard": CallParams(effort="low", max_tokens=400),
    "summary": CallParams(effort="low", max_tokens=1500),
    "character": CallParams(effort="low", max_tokens=1500),
    # Stage 6.2d (ADR 42): the background world generator after each turn, a JSON list of changes.
    "generator": CallParams(effort="low", max_tokens=1200),
}


@dataclass(frozen=True)
class Tier:
    model: str
    # Spend cap per UTC day in USD (OpenRouter's reported cost); None = not limited.
    daily_cost_usd: Decimal | None


# Prototype: everyone is on "free" (Haiku 5.5, $0.05/day). Models and caps for pro/max are a
# product decision for later (backlog); until then they behave like free.
TIERS: dict[str, Tier] = {
    "free": Tier(model=HAIKU, daily_cost_usd=Decimal("0.05")),
    "pro": Tier(model=HAIKU, daily_cost_usd=Decimal("0.05")),
    "max": Tier(model=HAIKU, daily_cost_usd=Decimal("0.05")),
}


def tier(name: str) -> Tier:
    return TIERS.get(name, TIERS["free"])


def daily_cap(tier_name: str, override: Decimal | None = None) -> Decimal | None:
    """The spend cap per UTC day: `DAILY_COST_LIMIT_USD` if set, else the tier's own."""
    return override if override is not None else tier(tier_name).daily_cost_usd
