"""Load explicit, secret-safe language-model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from personal_knowledge_agent.domain.errors import ConfigurationError

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL_ID = "deepseek-v4-flash"


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Configuration for one OpenAI-compatible language-model endpoint."""

    api_key: str = field(repr=False)
    base_url: str = _DEFAULT_BASE_URL
    model_id: str = _DEFAULT_MODEL_ID


def load_llm_settings(environment: Mapping[str, str]) -> LLMSettings:
    """Build settings from a supplied environment mapping without global reads."""
    api_key = _required_setting(environment, "PKA_LLM_API_KEY")
    base_url = _optional_setting(
        environment,
        "PKA_LLM_BASE_URL",
        _DEFAULT_BASE_URL,
    )
    model_id = _optional_setting(
        environment,
        "PKA_LLM_MODEL",
        _DEFAULT_MODEL_ID,
    )
    return LLMSettings(api_key, base_url, model_id)


def _required_setting(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise ConfigurationError(f"Missing required environment variable {name}")
    return value.strip()


def _optional_setting(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    value = environment.get(name)
    if value is None:
        return default
    if not value.strip():
        raise ConfigurationError(f"Environment variable {name} must not be blank")
    return value.strip()
