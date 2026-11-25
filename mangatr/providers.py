"""Translation providers: one `complete(system, messages)` call each.

`messages` is a list of {"role": "user"|"assistant", "content": str}. Retries
on 429 and 5xx with backoff (the Retry-After header wins), rotates API keys
when one is rate limited, and `FallbackProvider` tries the next provider when
one is exhausted. The network client is injectable so tests use a fake.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> str: ...


class ProviderError(RuntimeError):
    """The provider failed for good (bad key, bad request, retries exhausted)."""


class KeyRing:
    """API keys used in turn; a rate-limited key rests for `cooldown` seconds."""

    def __init__(self, keys: Sequence[str], cooldown: float = 60.0, clock=time.monotonic) -> None:
        if not keys:
            raise ValueError("at least one API key is required")
        self._keys = list(keys)
        self._until = dict.fromkeys(self._keys, 0.0)
        self._cycle = itertools.cycle(self._keys)
        self._cooldown = cooldown
        self._clock = clock

    def next(self) -> tuple[str, float]:
        """A usable key and how long to wait before using it (0 when ready)."""
        now = self._clock()
        for _ in self._keys:
            key = next(self._cycle)
            if self._until[key] <= now:
                return key, 0.0
        key = min(self._keys, key=lambda k: self._until[k])
        return key, self._until[key] - now

    def rest(self, key: str, seconds: float | None = None) -> None:
        self._until[key] = self._clock() + (seconds if seconds is not None else self._cooldown)


class _Http:
    def __init__(
        self,
        keys: Sequence[str],
        client: httpx.Client | None,
        retries: int,
        sleep: Callable[[float], None],
    ) -> None:
        self.ring = KeyRing(keys)
        self.client = client or httpx.Client(timeout=120)
        self.retries = retries
        self.sleep = sleep

    def post(self, url: str, build: Callable[[str], dict[str, Any]], pick) -> str:
        last = "no attempt"
        for attempt in range(self.retries + 1):
            key, wait = self.ring.next()
            if wait:
                self.sleep(min(wait, 30.0))
            request = build(key)
            try:
                response = self.client.post(url, **request)
            except httpx.TransportError as exc:
                last = f"network error: {exc}"
                self.sleep(min(2.0**attempt, 20.0))
                continue
            if response.status_code in RETRY_STATUS:
                after = response.headers.get("retry-after")
                numeric = bool(after) and after.replace(".", "", 1).isdigit()
                delay = float(after) if numeric else 2.0**attempt
                if response.status_code == 429:
                    self.ring.rest(key, delay)
                last = f"HTTP {response.status_code}"
                self.sleep(min(delay, 30.0))
                continue
            if response.status_code >= 400:
                raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
            try:
                return pick(response.json())
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ProviderError(f"unexpected reply shape: {exc!r}") from exc
        raise ProviderError(f"retries exhausted ({last})")


class OpenAICompatible:
    """OpenAI, OpenRouter, DeepSeek, Ollama (/v1) and anything with the same API."""

    def __init__(
        self,
        keys: Sequence[str],
        model: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        temperature: float = 0.3,
    ) -> None:
        self.name = f"openai:{model}"
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self._http = _Http(keys or ["none"], client, retries, sleep)

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> str:
        def build(key: str) -> dict[str, Any]:
            body = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, *messages],
            }
            return {"headers": {"Authorization": f"Bearer {key}"}, "json": body}

        return self._http.post(
            f"{self.base_url}/chat/completions",
            build,
            lambda data: data["choices"][0]["message"]["content"],
        )


class Anthropic:
    def __init__(
        self,
        keys: Sequence[str],
        model: str,
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        max_tokens: int = 4096,
    ) -> None:
        self.name = f"anthropic:{model}"
        self.model, self.max_tokens = model, max_tokens
        self._http = _Http(keys, client, retries, sleep)

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> str:
        def build(key: str) -> dict[str, Any]:
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": list(messages),
            }
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            return {"headers": headers, "json": body}

        return self._http.post(
            "https://api.anthropic.com/v1/messages",
            build,
            lambda data: "".join(p["text"] for p in data["content"] if p.get("type") == "text"),
        )


class Gemini:
    def __init__(
        self,
        keys: Sequence[str],
        model: str,
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = f"gemini:{model}"
        self.model = model
        self._http = _Http(keys, client, retries, sleep)

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> str:
        def build(key: str) -> dict[str, Any]:
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [
                    {
                        "role": "user" if m["role"] == "user" else "model",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                ],
            }
            return {"headers": {"x-goog-api-key": key}, "json": body}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        return self._http.post(
            url,
            build,
            lambda data: "".join(p["text"] for p in data["candidates"][0]["content"]["parts"]),
        )


class FallbackProvider:
    """Tries each provider in order; the first that answers wins."""

    def __init__(self, providers: Sequence[Provider]) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = list(providers)
        self.name = "fallback:" + ",".join(p.name for p in providers)

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> str:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.complete(system, messages)
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
        raise ProviderError("; ".join(errors))
