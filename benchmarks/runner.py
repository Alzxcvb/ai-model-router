"""Benchmark runner — sends prompts to multiple models and records results.

Usage:
    python -m benchmarks.runner                    # Run all prompts against all models
    python -m benchmarks.runner --models 2         # Only test top 2 models per task
    python -m benchmarks.runner --prompt code-001  # Run a single prompt
    python -m benchmarks.runner --dry-run          # Show what would run without calling APIs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.models import MODELS, get_ranked_models
from router.providers import OpenRouterProvider
from router.types import TaskType

BENCHMARKS_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARKS_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"


@dataclass
class BenchmarkResult:
    """Result of running a single prompt against a single model."""
    prompt_id: str
    model_id: str
    model_name: str
    expected_type: str
    response: str
    latency_ms: float
    timestamp: str
    eval_score: float | None = None
    eval_reasoning: str | None = None


@dataclass
class BenchmarkRun:
    """A complete benchmark run across prompts and models."""
    run_id: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_prompts: int = 0
    total_calls: int = 0
    total_latency_ms: float = 0.0


def load_prompts(prompt_id: str | None = None) -> list[dict]:
    """Load benchmark prompts from the data file."""
    prompts_file = DATA_DIR / "prompts.json"
    with open(prompts_file) as f:
        data = json.load(f)

    prompts = data["prompts"]
    if prompt_id:
        prompts = [p for p in prompts if p["id"] == prompt_id]
        if not prompts:
            raise ValueError(f"Prompt '{prompt_id}' not found in dataset")

    return prompts


def get_models_for_task(task_type_str: str, top_n: int | None = None) -> list:
    """Get models ranked for a task type, optionally limited to top N."""
    task_type = TaskType(task_type_str)
    ranked = get_ranked_models(task_type)
    if top_n:
        ranked = ranked[:top_n]
    return [model for model, score in ranked]


def run_benchmark(
    *,
    prompt_id: str | None = None,
    top_n_models: int | None = None,
    dry_run: bool = False,
) -> BenchmarkRun:
    """Run the benchmark suite.

    Args:
        prompt_id: If set, only run this specific prompt.
        top_n_models: If set, only test the top N models per task type.
        dry_run: If True, don't call APIs — just show what would run.
    """
    prompts = load_prompts(prompt_id)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = BenchmarkRun(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_prompts=len(prompts),
    )

    if dry_run:
        for prompt in prompts:
            models = get_models_for_task(prompt["expected_type"], top_n_models)
            for model in models:
                print(f"  [DRY RUN] {prompt['id']:<20} → {model.name}")
                run.total_calls += 1
        print(f"\nWould make {run.total_calls} API calls across {run.total_prompts} prompts.")
        return run

    provider = OpenRouterProvider()

    for i, prompt in enumerate(prompts):
        models = get_models_for_task(prompt["expected_type"], top_n_models)
        print(f"\n[{i+1}/{len(prompts)}] {prompt['id']} ({prompt['expected_type']}, {prompt['difficulty']})")

        for model in models:
            print(f"  → {model.name}...", end=" ", flush=True)

            try:
                response, latency_ms = provider.call_raw(
                    model.id,
                    prompt["text"],
                    max_tokens=1024,
                )

                result = BenchmarkResult(
                    prompt_id=prompt["id"],
                    model_id=model.id,
                    model_name=model.name,
                    expected_type=prompt["expected_type"],
                    response=response,
                    latency_ms=round(latency_ms, 1),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                run.results.append(result)
                run.total_calls += 1
                run.total_latency_ms += latency_ms

                print(f"{latency_ms:.0f}ms, {len(response)} chars")

            except Exception as e:
                print(f"ERROR: {e}")
                run.results.append(BenchmarkResult(
                    prompt_id=prompt["id"],
                    model_id=model.id,
                    model_name=model.name,
                    expected_type=prompt["expected_type"],
                    response=f"ERROR: {e}",
                    latency_ms=0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

            # Brief pause between calls to avoid rate limits
            time.sleep(0.5)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"run_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "run_id": run.run_id,
                "timestamp": run.timestamp,
                "total_prompts": run.total_prompts,
                "total_calls": run.total_calls,
                "total_latency_ms": round(run.total_latency_ms, 1),
                "results": [asdict(r) for r in run.results],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_file}")
    print(f"Total: {run.total_calls} calls, {run.total_latency_ms:.0f}ms total latency")

    return run


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run AI Model Router benchmarks")
    parser.add_argument("--prompt", help="Run a specific prompt by ID")
    parser.add_argument("--models", type=int, help="Only test top N models per task type")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    run_benchmark(
        prompt_id=args.prompt,
        top_n_models=args.models,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
