"""Prompt classifier — determines task type for a given prompt.

Supports two modes:
- "rules": Fast, free keyword matching (v0.1)
- "llm": Richer classification via a cheap LLM call (v0.2)
"""

from __future__ import annotations

from .rules import classify_by_rules
from .types import ClassificationResult


def classify(
    prompt: str,
    *,
    method: str = "rules",
    provider=None,
) -> ClassificationResult:
    """Classify a prompt into a task type.

    Args:
        prompt: The user prompt to classify.
        method: "rules" for keyword matching, "llm" for LLM-based classification.
        provider: OpenRouterProvider instance (only needed for method="llm").
    """
    if method == "llm":
        from .llm_classifier import classify_by_llm
        return classify_by_llm(prompt, provider=provider)

    return classify_by_rules(prompt)
