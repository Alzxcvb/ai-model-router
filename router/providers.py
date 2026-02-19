"""API wrappers for model providers (OpenRouter + direct APIs)."""

from __future__ import annotations

import os
import time

from openai import OpenAI

from .types import ModelInfo


class OpenRouterProvider:
    """Sends prompts to models via the OpenRouter API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY env var or pass api_key."
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def call(
        self,
        model: ModelInfo,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Send a prompt to a model and return (response_text, latency_ms)."""
        return self.call_raw(
            model.id, prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )

    def call_raw(
        self,
        model_id: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Send a prompt to a model by ID string. Returns (response_text, latency_ms)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        content = response.choices[0].message.content or ""
        return content, latency_ms
