"""Core routing logic — classify prompt, select model, call provider."""

from __future__ import annotations

from .classifier import classify
from .models import get_best_model_for_task, get_ranked_models
from .providers import OpenRouterProvider
from .types import RouterResponse, RoutingDecision


class Router:
    """Routes prompts to the best model for each task."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        budget: str = "best",
    ):
        self.provider = OpenRouterProvider(api_key=api_key)
        self.budget = budget

    def route(self, prompt: str, *, dry_run: bool = False) -> RouterResponse | RoutingDecision:
        """Classify a prompt, pick the best model, and (optionally) call it.

        Args:
            prompt: The user's input prompt.
            dry_run: If True, return only the routing decision without calling the model.

        Returns:
            RouterResponse with the model's output and metadata,
            or RoutingDecision if dry_run=True.
        """
        # Step 1: Classify the prompt
        classification = classify(prompt)

        # Step 2: Select the best model
        model = get_best_model_for_task(classification.task_type, budget=self.budget)
        ranked = get_ranked_models(classification.task_type)
        score = model.scores.get(
            classification.task_type.value,
            model.scores.get(classification.task_type, 0),
        )

        # Build routing decision
        alternatives = [(m, s) for m, s in ranked if m.id != model.id][:3]
        decision = RoutingDecision(
            model=model,
            task_type=classification.task_type,
            score=score,
            reasoning=(
                f"Classified as '{classification.task_type.value}' "
                f"(confidence: {classification.confidence}, "
                f"keywords: {classification.keywords_matched}). "
                f"Selected {model.name} with score {score}/10 "
                f"(budget: {self.budget})."
            ),
            alternatives=alternatives,
        )

        if dry_run:
            return decision

        # Step 3: Call the model
        content, latency_ms = self.provider.call(model, prompt)

        # Estimate cost (rough: assume ~prompt_len/4 input tokens, ~response_len/4 output tokens)
        est_input_tokens = len(prompt) / 4
        est_output_tokens = len(content) / 4
        estimated_cost = (
            (est_input_tokens / 1_000_000) * model.cost_per_million_input
            + (est_output_tokens / 1_000_000) * model.cost_per_million_output
        )

        return RouterResponse(
            content=content,
            decision=decision,
            latency_ms=round(latency_ms, 1),
            estimated_cost=round(estimated_cost, 6),
        )
