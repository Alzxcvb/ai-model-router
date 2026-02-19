"""Evaluator — scores model responses using an LLM judge.

Uses a strong model (Claude Sonnet) to evaluate benchmark responses
against the eval criteria defined in the prompt dataset.

Usage:
    python -m benchmarks.evaluator benchmarks/data/results/run_XXXXXXXX_XXXXXX.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.providers import OpenRouterProvider

# Use a strong model as the judge
JUDGE_MODEL = "anthropic/claude-sonnet-4-5"

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator scoring AI model responses. Given:
1. The original prompt
2. The model's response
3. Evaluation criteria

Score the response on a scale of 1-10 and explain your reasoning briefly.

Return ONLY this JSON (no markdown, no backticks):
{
  "score": <1-10>,
  "reasoning": "<1-2 sentence explanation>"
}"""


def load_prompts_map() -> dict[str, dict]:
    """Load prompts indexed by ID."""
    prompts_file = Path(__file__).resolve().parent / "data" / "prompts.json"
    with open(prompts_file) as f:
        data = json.load(f)
    return {p["id"]: p for p in data["prompts"]}


def evaluate_result(
    provider: OpenRouterProvider,
    prompt_data: dict,
    model_response: str,
) -> tuple[float, str]:
    """Score a single model response.

    Returns (score, reasoning).
    """
    eval_prompt = (
        f"## Original Prompt\n{prompt_data['text']}\n\n"
        f"## Evaluation Criteria\n{prompt_data['eval_criteria']}\n\n"
        f"## Model Response\n{model_response}\n\n"
        f"Score this response (1-10) based on the evaluation criteria."
    )

    response_text, _ = provider.call_raw(
        JUDGE_MODEL,
        eval_prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        max_tokens=200,
    )

    return _parse_eval(response_text)


def _parse_eval(response_text: str) -> tuple[float, str]:
    """Parse the judge's JSON response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        score = float(data.get("score", 5))
        score = max(1.0, min(10.0, score))
        reasoning = data.get("reasoning", "No reasoning provided")
        return score, reasoning
    except (json.JSONDecodeError, ValueError):
        return 5.0, f"Failed to parse judge response: {text[:100]}"


def evaluate_run(results_file: str) -> None:
    """Evaluate all results in a benchmark run file."""
    results_path = Path(results_file)
    with open(results_path) as f:
        run_data = json.load(f)

    prompts_map = load_prompts_map()
    provider = OpenRouterProvider()

    results = run_data["results"]
    evaluated = 0
    skipped = 0

    print(f"Evaluating {len(results)} results from run {run_data['run_id']}...")

    for i, result in enumerate(results):
        # Skip already-evaluated or errored results
        if result.get("eval_score") is not None:
            skipped += 1
            continue
        if result["response"].startswith("ERROR:"):
            result["eval_score"] = 0
            result["eval_reasoning"] = "Model returned an error"
            skipped += 1
            continue

        prompt_data = prompts_map.get(result["prompt_id"])
        if not prompt_data:
            print(f"  [SKIP] Unknown prompt: {result['prompt_id']}")
            skipped += 1
            continue

        print(
            f"  [{i+1}/{len(results)}] {result['prompt_id']} × {result['model_name']}...",
            end=" ",
            flush=True,
        )

        try:
            score, reasoning = evaluate_result(provider, prompt_data, result["response"])
            result["eval_score"] = score
            result["eval_reasoning"] = reasoning
            evaluated += 1
            print(f"score: {score}/10")
        except Exception as e:
            print(f"ERROR: {e}")
            result["eval_score"] = None
            result["eval_reasoning"] = f"Evaluation error: {e}"

        time.sleep(0.5)

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"\nDone: {evaluated} evaluated, {skipped} skipped")
    print(f"Updated: {results_path}")

    # Print summary
    _print_summary(run_data)


def _print_summary(run_data: dict) -> None:
    """Print a summary table of scores by model and task type."""
    from collections import defaultdict

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for result in run_data["results"]:
        if result.get("eval_score") is not None and result["eval_score"] > 0:
            scores[result["model_name"]][result["expected_type"]].append(result["eval_score"])

    if not scores:
        return

    print("\n=== Score Summary ===")
    print(f"{'Model':<25} {'Task Type':<18} {'Avg Score':<10} {'Count'}")
    print("-" * 65)

    for model_name in sorted(scores):
        for task_type in sorted(scores[model_name]):
            task_scores = scores[model_name][task_type]
            avg = sum(task_scores) / len(task_scores)
            print(f"{model_name:<25} {task_type:<18} {avg:>6.1f}/10   {len(task_scores)}")

    # Overall averages per model
    print("\n--- Overall Averages ---")
    for model_name in sorted(scores):
        all_scores = [s for task_scores in scores[model_name].values() for s in task_scores]
        avg = sum(all_scores) / len(all_scores)
        print(f"  {model_name:<25} {avg:.1f}/10  ({len(all_scores)} evals)")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate benchmark results using LLM judge")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    args = parser.parse_args()

    evaluate_run(args.results_file)


if __name__ == "__main__":
    main()
