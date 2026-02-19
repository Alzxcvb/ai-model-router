"""CLI entry point: python -m router "Your prompt here" """

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .router import Router
from .types import RouterResponse, RoutingDecision


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Model Router — routes prompts to the best model",
    )
    parser.add_argument("prompt", help="The prompt to route and send")
    parser.add_argument(
        "--budget",
        choices=["best", "balanced", "cheap"],
        default="best",
        help="Budget mode: best (highest quality), balanced (score/cost), cheap (lowest cost)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show routing decision, don't call the model",
    )
    args = parser.parse_args()

    try:
        router = Router(budget=args.budget)
    except ValueError as e:
        if args.dry_run:
            # Dry run doesn't need an API key — create a minimal router
            router = None
        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        # For dry run without API key, classify and select manually
        from .classifier import classify
        from .models import get_best_model_for_task, get_ranked_models

        classification = classify(args.prompt)
        model = get_best_model_for_task(classification.task_type, budget=args.budget)
        ranked = get_ranked_models(classification.task_type)
        score = model.scores.get(
            classification.task_type.value,
            model.scores.get(classification.task_type, 0),
        )

        print("=== Routing Decision (dry run) ===")
        print(f"Prompt:     {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
        print(f"Task type:  {classification.task_type.value}")
        print(f"Confidence: {classification.confidence}")
        print(f"Keywords:   {', '.join(classification.keywords_matched) or '(none)'}")
        print(f"Model:      {model.name} ({model.id})")
        print(f"Score:      {score}/10")
        print(f"Budget:     {args.budget}")
        print(f"Cost:       ${model.cost_per_million_input}/M in, ${model.cost_per_million_output}/M out")
        print()
        print("--- Alternatives ---")
        for alt_model, alt_score in ranked:
            if alt_model.id != model.id:
                print(f"  {alt_model.name:<25} score: {alt_score}/10  cost: ${alt_model.cost_per_million_input}/M in")
        return

    result = router.route(args.prompt)
    assert isinstance(result, RouterResponse)

    print("=== AI Model Router ===")
    print(f"Task type:  {result.decision.task_type.value}")
    print(f"Model:      {result.decision.model.name}")
    print(f"Score:      {result.decision.score}/10")
    print(f"Latency:    {result.latency_ms}ms")
    if result.estimated_cost is not None:
        print(f"Est. cost:  ${result.estimated_cost:.6f}")
    print(f"Reasoning:  {result.decision.reasoning}")
    print()
    print("--- Response ---")
    print(result.content)


if __name__ == "__main__":
    main()
