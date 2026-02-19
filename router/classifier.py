"""Prompt classifier — determines task type for a given prompt.

v0.1: Uses the rules engine (keyword matching).
v0.2 (future): Will use a small/fast LLM for classification.
"""

from __future__ import annotations

from .rules import classify_by_rules
from .types import ClassificationResult


def classify(prompt: str) -> ClassificationResult:
    """Classify a prompt into a task type.

    Currently delegates to the rules engine.
    In Phase 2 this will be replaced with an LLM-based classifier.
    """
    return classify_by_rules(prompt)
