"""Tests for the LLM-based classifier (Phase 2).

Uses mock API responses to test classification parsing without real API calls.
"""

import pytest

from router.llm_classifier import _parse_classification
from router.types import Complexity, TaskType


class TestParseClassification:
    def test_valid_json(self):
        result = _parse_classification(
            '{"task_type": "code", "confidence": 0.95, "complexity": "high", '
            '"needs_reasoning": true, "needs_creativity": false}'
        )
        assert result.task_type == TaskType.CODE
        assert result.confidence == 0.95
        assert result.complexity == Complexity.HIGH
        assert result.needs_reasoning is True
        assert result.needs_creativity is False
        assert result.method == "llm"

    def test_markdown_fenced_json(self):
        result = _parse_classification(
            '```json\n{"task_type": "writing", "confidence": 0.8, '
            '"complexity": "medium", "needs_reasoning": false, "needs_creativity": true}\n```'
        )
        assert result.task_type == TaskType.WRITING
        assert result.confidence == 0.8
        assert result.needs_creativity is True

    def test_invalid_json_falls_back(self):
        result = _parse_classification("This is not JSON at all")
        assert result.task_type == TaskType.CONVERSATION
        assert result.confidence == 0.2
        assert result.method == "llm_fallback"

    def test_unknown_task_type_falls_back(self):
        result = _parse_classification(
            '{"task_type": "banana", "confidence": 0.9, "complexity": "low"}'
        )
        assert result.task_type == TaskType.CONVERSATION

    def test_confidence_clamped(self):
        result = _parse_classification(
            '{"task_type": "reasoning", "confidence": 1.5, "complexity": "medium"}'
        )
        assert result.confidence == 1.0

        result = _parse_classification(
            '{"task_type": "reasoning", "confidence": -0.5, "complexity": "medium"}'
        )
        assert result.confidence == 0.0

    def test_missing_optional_fields(self):
        result = _parse_classification('{"task_type": "data"}')
        assert result.task_type == TaskType.DATA
        assert result.confidence == 0.7  # default
        assert result.complexity == Complexity.MEDIUM  # default
        assert result.needs_reasoning is False
        assert result.needs_creativity is False

    def test_all_task_types_parseable(self):
        for task_type in TaskType:
            result = _parse_classification(
                f'{{"task_type": "{task_type.value}", "confidence": 0.8}}'
            )
            assert result.task_type == task_type
