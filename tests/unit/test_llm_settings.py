"""Tests for explicit, secret-safe language-model settings."""

import pytest

from personal_knowledge_agent.domain.errors import ConfigurationError
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings


def test_settings_require_api_key_and_apply_deepseek_defaults() -> None:
    settings = load_llm_settings({"PKA_LLM_API_KEY": "  test-secret  "})

    assert settings.api_key == "test-secret"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model_id == "deepseek-v4-flash"
    assert "test-secret" not in repr(settings)


def test_settings_preserve_non_blank_overrides_after_trimming() -> None:
    settings = load_llm_settings(
        {
            "PKA_LLM_API_KEY": "key",
            "PKA_LLM_BASE_URL": " https://provider.example/v1 ",
            "PKA_LLM_MODEL": " provider-model ",
        }
    )

    assert settings.base_url == "https://provider.example/v1"
    assert settings.model_id == "provider-model"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_settings_reject_missing_or_blank_api_key(value: str | None) -> None:
    environment = {} if value is None else {"PKA_LLM_API_KEY": value}

    with pytest.raises(ConfigurationError, match="PKA_LLM_API_KEY"):
        load_llm_settings(environment)


@pytest.mark.parametrize("name", ["PKA_LLM_BASE_URL", "PKA_LLM_MODEL"])
def test_settings_reject_blank_optional_overrides(name: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        load_llm_settings({"PKA_LLM_API_KEY": "key", name: "   "})
