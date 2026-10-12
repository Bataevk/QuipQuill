import pytest
from pydantic import ValidationError

from quipquill_cloud.core.config import Settings


def test_dev_defaults_load_without_env_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.env == "dev"
    assert settings.llm_model_free == "anthropic/claude-haiku-5.5"


def test_prod_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, env="prod")


def test_prod_accepts_real_jwt_secret() -> None:
    settings = Settings(_env_file=None, env="prod", jwt_secret="a-real-secret")

    assert settings.env == "prod"
