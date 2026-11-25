"""Settings: where the keys and the defaults come from.

Read from the environment only; nothing else in the project reads it. Keys are
never written to a file or printed. A provider may have several keys, separated
by commas, and they are used in turn (see providers.KeyRing).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from .providers import Anthropic, FallbackProvider, Gemini, OpenAICompatible, Provider

KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-2.5-flash",
    "openrouter": "google/gemini-2.5-flash",
    "ollama": "qwen2.5:14b",
}
BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def keys_for(provider: str, env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    raw = env.get(KEY_VARIABLES[provider], "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def make_provider(
    name: str, model: str | None = None, env: Mapping[str, str] | None = None
) -> Provider:
    """One provider by name, or several joined with commas ("gemini,openai") as a fallback chain."""
    if "," in name:
        return FallbackProvider([make_provider(n.strip(), None, env) for n in name.split(",")])
    if name not in KEY_VARIABLES:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(KEY_VARIABLES)}")
    keys = keys_for(name, env)
    if not keys and name != "ollama":
        raise ValueError(f"no API key: set ${KEY_VARIABLES[name]}")
    model = model or DEFAULT_MODELS[name]
    if name == "anthropic":
        return Anthropic(keys, model)
    if name == "gemini":
        return Gemini(keys, model)
    return OpenAICompatible(keys, model, BASE_URLS[name])
