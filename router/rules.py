"""Rules engine v0.1 — keyword/pattern matching for prompt classification."""

from __future__ import annotations

import re

from .types import ClassificationResult, TaskType

# Keywords/phrases mapped to task types, ordered by specificity
RULES: dict[TaskType, list[str]] = {
    TaskType.CODE: [
        "write a function", "write a script", "implement", "debug", "fix this bug",
        "refactor", "code review", "unit test", "regex", "algorithm",
        "python", "javascript", "typescript", "java", "rust", "golang",
        "html", "css", "sql", "api endpoint", "class", "function",
        "compile", "runtime error", "syntax error", "stack trace",
    ],
    TaskType.WRITING: [
        "essay", "blog post", "article", "creative writing", "story",
        "poem", "rewrite", "proofread", "persuasive", "narrative",
        "write me", "draft", "compose", "copywriting", "slogan",
    ],
    TaskType.REASONING: [
        "solve", "calculate", "prove", "logic", "math", "equation",
        "why does", "explain why", "what would happen if", "probability",
        "derive", "reasoning", "step by step", "analyze the argument",
    ],
    TaskType.SUMMARIZATION: [
        "summarize", "tldr", "tl;dr", "key points", "condense",
        "brief overview", "main ideas", "recap", "in short",
    ],
    TaskType.CONVERSATION: [
        "chat", "tell me about yourself", "how are you", "what do you think",
        "let's talk", "opinion on", "recommend",
    ],
    TaskType.RESEARCH: [
        "research", "find information", "what is", "who is", "when did",
        "compare and contrast", "pros and cons", "sources", "evidence",
        "literature review", "state of the art",
    ],
    TaskType.TRANSLATION: [
        "translate", "translation", "in spanish", "in french", "in german",
        "in japanese", "in chinese", "in arabic", "to english", "from english",
        "localize", "multilingual",
    ],
    TaskType.DATA: [
        "csv", "json", "parse", "data analysis", "spreadsheet",
        "table", "dataset", "extract data", "structured data",
        "visualization", "chart", "graph", "statistics",
    ],
}

# Pre-compile patterns for performance
_COMPILED_RULES: dict[TaskType, list[re.Pattern]] = {
    task_type: [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]
    for task_type, keywords in RULES.items()
}


def classify_by_rules(prompt: str) -> ClassificationResult:
    """Classify a prompt using keyword/pattern matching.

    Returns the task type with the most keyword matches.
    Falls back to CONVERSATION if nothing matches.
    """
    scores: dict[TaskType, list[str]] = {t: [] for t in TaskType}

    for task_type, patterns in _COMPILED_RULES.items():
        for pattern in patterns:
            match = pattern.search(prompt)
            if match:
                scores[task_type].append(match.group().lower())

    # Find the task type with the most matches
    best_type = max(scores, key=lambda t: len(scores[t]))
    matched = scores[best_type]

    if not matched:
        return ClassificationResult(
            task_type=TaskType.CONVERSATION,
            confidence=0.3,
            keywords_matched=[],
        )

    # Confidence scales with number of matches, capped at 0.9
    # (rules alone can never be fully confident)
    total_keywords = len(RULES[best_type])
    confidence = min(0.9, 0.4 + (len(matched) / total_keywords) * 0.5)

    return ClassificationResult(
        task_type=best_type,
        confidence=round(confidence, 2),
        keywords_matched=matched,
    )
