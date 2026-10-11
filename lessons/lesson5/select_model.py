"""Accuracy-first model selector with a *hypothetical* execution-time budget.

The p95-based proxy is not a bound. This only filters model candidates by a
measured-latency heuristic; it cannot establish end-to-end schedulability.
"""
import argparse
import json
from pathlib import Path
from microsecond_scheduler import model_execution_us


def select_model(models: dict, budget_us: int):
    """Choose highest reported test accuracy among budget-compatible models."""
    if budget_us < 1:
        raise ValueError("budget_us must be positive")
    candidates = [(name, stats, model_execution_us(stats))
                  for name, stats in models.items()
                  if model_execution_us(stats) <= budget_us]
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x[1]["test_accuracy"], -x[2], x[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=Path, default=Path("outputs/benchmarks.json"))
    parser.add_argument("--budget-us", type=int, default=80)
    args = parser.parse_args()
    data = json.loads(args.benchmarks.read_text(encoding="utf-8"))
    winner = select_model(data["models"], args.budget_us)
    if winner is None:
        print(f"No model compatible with assumed budget of {args.budget_us} us.")
    else:
        name, stats, cost = winner
        print(f"Chosen model: {name} | reported test accuracy: {stats['test_accuracy']:.2%} "
              f"| empirical p95 proxy rounded: {cost} us | budget: {args.budget_us} us")
    print("Heuristic only: empirical p95 is NOT a WCET; this does not prove real-time feasibility.")


if __name__ == "__main__":
    main()
