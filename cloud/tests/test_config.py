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
