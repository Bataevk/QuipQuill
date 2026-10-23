import pytest
from pydantic import ValidationError

from quipquill_cloud.core.config import Settings
from quipquill_cloud.llm.policy import Route


def test_dev_defaults_load_without_env_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.env == "dev"
    assert settings.provider_order is None  # each route has its own order (ADR 50)
    assert settings.llm_model == ""


def test_prod_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, env="prod")


def test_prod_rejects_short_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="32 bytes"):
        Settings(_env_file=None, env="prod", jwt_secret="too-short")


def test_prod_requires_openrouter_key() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(_env_file=None, env="prod", jwt_secret="x" * 32)


def test_prod_accepts_real_secrets() -> None:
    settings = Settings(
        _env_file=None, env="prod", jwt_secret="x" * 32, openrouter_api_key="sk-or-test"
    )

    assert settings.env == "prod"


def test_config_overrides_route_model_and_providers() -> None:
    basic = Route(model="a/model", providers=("p1", "p2"))

    assert Settings(_env_file=None).override(basic) == basic
    assert Settings(_env_file=None, llm_model="b/model", provider_order=[]).override(
        basic
    ) == Route(model="b/model", providers=())
    assert Settings(_env_file=None, provider_order=["p2"]).override(basic).providers == ("p2",)


def test_fallback_keeps_route_providers_unless_set() -> None:
    basic = Route(model="a/model", providers=("p1", "p2"))

    assert Settings(_env_file=None).fallback(basic) is None
    spare = Settings(_env_file=None, llm_fallback_model="h/model").fallback(basic)
    assert spare == Route(model="h/model", providers=("p1", "p2"))
    anywhere = Settings(_env_file=None, llm_fallback_model="h/model", llm_fallback_providers=[])
    assert anywhere.fallback(basic) == Route(model="h/model", providers=())


def test_effort_from_config_per_purpose() -> None:
    plain = Settings(_env_file=None)
    assert (plain.effort("gm"), plain.effort("npc"), plain.effort("pregen")) == (
        "high",
        "medium",
        "low",
    )
    assert plain.effort("gm", spare=True) == "high"  # the spare thinks like the main model

    tuned = Settings(
        _env_file=None, llm_effort={"gm": "medium"}, llm_fallback_effort={"npc": "low"}
    )
    assert (tuned.effort("gm"), tuned.effort("gm", spare=True)) == ("medium", "medium")
    assert (tuned.effort("npc"), tuned.effort("npc", spare=True)) == ("medium", "low")


def test_unknown_effort_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_effort={"gm": "max"})
