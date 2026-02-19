"""LLM-based prompt classifier (Phase 2).

Uses a cheap, fast model (Gemini Flash) to classify prompts into structured
categories with richer metadata than keyword matching alone.
"""

from __future__ import annotations

import json
import logging

from .providers import OpenRouterProvider
from .types import ClassificationResult, Complexity, TaskType

logger = logging.getLogger(__name__)

# The model used for classification — should be fast and cheap
CLASSIFIER_MODEL = "google/gemini-2.0-flash-001"

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a prompt classifier. Given a user prompt, classify it into exactly one category \
and return structured JSON. Do NOT include any text outside the JSON object.

Categories:
- code: programming, debugging, code review, algorithms
- writing: essays, creative writing, blog posts, copywriting
- reasoning: math, logic, proofs, analysis, step-by-step problem solving
- summarization: condensing text, extracting key points, TL;DR
- conversation: casual chat, opinions, recommendations, Q&A
- research: factual lookup, comparisons, literature review
- translation: language translation, localization
- data: CSV/JSON parsing, data analysis, statistics, visualization

Return ONLY this JSON (no markdown, no backticks):
{
  "task_type": "<one of the categories above>",
  "confidence": <0.0-1.0>,
  "complexity": "<low|medium|high>",
  "needs_reasoning": <true|false>,
  "needs_creativity": <true|false>
}"""


def classify_by_llm(
    prompt: str,
    provider: OpenRouterProvider | None = None,
) -> ClassificationResult:
    """Classify a prompt using an LLM call.

    Args:
        prompt: The user prompt to classify.
        provider: An OpenRouterProvider instance (creates one if not given).

    Returns:
        ClassificationResult with rich metadata from the LLM.
    """
    if provider is None:
        provider = OpenRouterProvider()

    response_text, _ = provider.call_raw(
        CLASSIFIER_MODEL,
        f"Classify this prompt:\n\n{prompt}",
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
        max_tokens=150,
    )

    return _parse_classification(response_text)


def _parse_classification(response_text: str) -> ClassificationResult:
    """Parse the LLM's JSON response into a ClassificationResult.

    Handles common formatting issues (markdown fences, extra whitespace).
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM classifier returned invalid JSON: %s", text[:200])
        return ClassificationResult(
            task_type=TaskType.CONVERSATION,
            confidence=0.2,
            method="llm_fallback",
        )

    # Parse task_type
    raw_type = data.get("task_type", "conversation").lower().strip()
    try:
        task_type = TaskType(raw_type)
    except ValueError:
        logger.warning("LLM returned unknown task_type: %s", raw_type)
        task_type = TaskType.CONVERSATION

    # Parse complexity
    raw_complexity = data.get("complexity", "medium").lower().strip()
    try:
        complexity = Complexity(raw_complexity)
    except ValueError:
        complexity = Complexity.MEDIUM

    confidence = float(data.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    return ClassificationResult(
        task_type=task_type,
        confidence=confidence,
        complexity=complexity,
        needs_reasoning=bool(data.get("needs_reasoning", False)),
        needs_creativity=bool(data.get("needs_creativity", False)),
        method="llm",
    )
