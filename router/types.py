"""Shared types and data classes for the AI Model Router."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    CODE = "code"
    WRITING = "writing"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    RESEARCH = "research"
    TRANSLATION = "translation"
    DATA = "data"


@dataclass
class ClassificationResult:
    """Output of prompt classification."""
    task_type: TaskType
    confidence: float  # 0.0 - 1.0
    keywords_matched: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Information about an AI model."""
    id: str                          # e.g. "anthropic/claude-sonnet-4-5"
    name: str                        # e.g. "Claude Sonnet 4.5"
    provider: str                    # e.g. "anthropic"
    scores: dict[str, float]         # task_type → score (0-10)
    cost_per_million_input: float    # USD per 1M input tokens
    cost_per_million_output: float   # USD per 1M output tokens
    max_context: int = 128_000       # context window size
    supports_images: bool = False
    supports_tools: bool = True


@dataclass
class RoutingDecision:
    """Explains why a model was selected."""
    model: ModelInfo
    task_type: TaskType
    score: float
    reasoning: str
    alternatives: list[tuple[ModelInfo, float]] = field(default_factory=list)


@dataclass
class RouterResponse:
    """Full response from the router."""
    content: str
    decision: RoutingDecision
    latency_ms: float
    estimated_cost: float | None = None
