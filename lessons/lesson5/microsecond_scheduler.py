"""Microsecond-resolution, event-driven EDF/RM simulation.

This is a WHAT-IF model, not a hard real-time certification:
* A sample p95 is used as an *assumed deterministic job cost* after rounding
  up to the nearest whole microsecond. It is not a valid WCET bound.
* Jobs execute on one fully preemptive processor with no context switch cost.
* All jobs of a periodic task share the same deterministic execution budget.
* Preprocessing, input transfers, deployment and contention are excluded.

Typical commands (from repository root):
  python lessons/lesson5/microsecond_scheduler.py --benchmarks outputs/benchmarks.json
  python lessons/lesson5/microsecond_scheduler.py --benchmarks outputs/benchmarks.json --deadline-us 80
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    execution_us: int
    period_us: int
    deadline_us: int
    offset_us: int = 0

    def __post_init__(self):
        if self.execution_us <= 0 or self.period_us <= 0 or self.deadline_us <= 0 or self.offset_us < 0:
            raise ValueError("Costs, periods and deadlines must be positive; offset nonnegative")


@dataclass
class Job:
    task: Task
    release_us: int
    remaining_us: int
    finish_us: int | None = None

    @property
    def deadline_us(self):
        return self.release_us + self.task.deadline_us


def simulate(tasks: list[Task], policy: str, horizon_us: int) -> dict:
    """Preempt at job releases or completions, without 1-us iteration loops.

    Counts as due only jobs whose absolute deadline <= horizon_us.
    Correct when a job finishes exactly at its deadline (on time).
    """
    if policy not in {"EDF", "RM"}:
        raise ValueError("policy must be EDF or RM")
    if horizon_us <= 0:
        raise ValueError("horizon_us must be > 0")
    all_jobs = [
        Job(task, release, task.execution_us)
        for task in tasks
        for release in range(task.offset_us, horizon_us, task.period_us)
    ]
    all_jobs.sort(key=lambda job: (job.release_us, job.task.name))
    now = 0
    next_job_idx = 0
    timeline = []
    while now < horizon_us:
        next_release = (all_jobs[next_job_idx].release_us
                        if next_job_idx < len(all_jobs) else horizon_us)
        ready = [j for j in all_jobs[:next_job_idx] if j.remaining_us > 0]
        if not ready:
            next_t = min(next_release, horizon_us)
            if next_t <= now:
                # Pending arrivals at this instant.
                while next_job_idx < len(all_jobs) and all_jobs[next_job_idx].release_us <= now:
                    next_job_idx += 1
                continue
            timeline.append((now, next_t, "IDLE"))
            now = next_t
            continue
        if policy == "EDF":
            selected = min(ready, key=lambda j: (j.deadline_us, j.release_us, j.task.name))
        else:
            selected = min(ready, key=lambda j: (j.task.period_us, j.release_us, j.task.name))
        run_us = min(selected.remaining_us, horizon_us - now,
                     next_release - now if next_job_idx < len(all_jobs) else horizon_us-now)
        if run_us <= 0:
            while next_job_idx < len(all_jobs) and all_jobs[next_job_idx].release_us <= now:
                next_job_idx += 1
            continue
        timeline.append((now, now + run_us, selected.task.name))
        selected.remaining_us -= run_us
        now += run_us
        if selected.remaining_us == 0:
            selected.finish_us = now
        while next_job_idx < len(all_jobs) and all_jobs[next_job_idx].release_us <= now:
            next_job_idx += 1

    due = [j for j in all_jobs if j.deadline_us <= horizon_us]
    misses = [j for j in due if j.finish_us is None or j.finish_us > j.deadline_us]
    return {
        "policy": policy,
        "horizon_us": horizon_us,
        "due": len(due),
        "missed": len(misses),
        "missed_by_task": {task.name: sum(j.task == task for j in misses) for task in tasks},
        "first_misses": [{"task": j.task.name, "release_us": j.release_us,
                          "deadline_us": j.deadline_us, "finish_us": j.finish_us}
                         for j in misses[:5]],
        "cpu_busy_percent": 100 * sum(b-a for a,b,name in timeline if name != "IDLE") / horizon_us,
        "utilization_percent": 100 * sum(t.execution_us/t.period_us for t in tasks),
        "timeline": timeline,
    }


def model_execution_us(stats: dict) -> int:
    """Ceil observed p95 milliseconds to microseconds (NOT a WCET)."""
    p95 = stats.get("p95_ms")
    if not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 <= 0:
        raise ValueError(f"invalid p95_ms value: {p95!r}")
    return max(1, math.ceil(p95 * 1000))


def tasks_for_model(model_name: str, cost_us: int, deadline_us: int) -> list[Task]:
    # T1/T2 costs remain synthetic classroom examples, not measured workloads.
    return [
        Task("T1_synthetic", execution_us=2000, period_us=5000, deadline_us=5000),
        Task("T2_synthetic", execution_us=2000, period_us=7000, deadline_us=7000),
        Task(f"T3_{model_name}", execution_us=cost_us, period_us=12000,
             deadline_us=deadline_us),
    ]


def main():
    parser = argparse.ArgumentParser(description="EDF/RM microsecond what-if simulator")
    parser.add_argument("--benchmarks", type=Path, default=Path("outputs/benchmarks.json"))
    parser.add_argument("--deadline-us", type=int, default=12000,
                        help="Relative deadline for inference job T3 in microseconds")
    parser.add_argument("--horizon-ms", type=int, default=420)
    parser.add_argument("--output", type=Path, default=Path("outputs/lesson5_microsecond_results.json"))
    args = parser.parse_args()
    if args.deadline_us <= 0 or args.horizon_ms <= 0:
        parser.error("deadline and horizon must be positive")
    measurements = json.loads(args.benchmarks.read_text(encoding="utf-8"))
    if "models" not in measurements or not measurements["models"]:
        parser.error("JSON needs a nonempty 'models' dictionary")
    results = {
        "assumptions": "empirical p95 rounded up to 1us; NOT a WCET; ideal preemptive single CPU",
        "inference_deadline_us": args.deadline_us,
        "models": {},
    }
    print("WARNING: p95-based deterministic costs are exploratory and provide NO hard real-time guarantees.")
    for model_name, stats in measurements["models"].items():
        cost_us = model_execution_us(stats)
        tasks = tasks_for_model(model_name, cost_us, args.deadline_us)
        print(f"\n{model_name}: p95={stats['p95_ms']:.4f} ms -> assumed cost={cost_us} us"
              f" | reported test accuracy={stats.get('test_accuracy', float('nan')):.2%}")
        model_result = {"assumed_cost_us": cost_us,
                        "reported_accuracy": stats.get("test_accuracy"),
                        "policies": {}}
        for policy in ("EDF", "RM"):
            r = simulate(tasks, policy=policy, horizon_us=args.horizon_ms * 1000)
            print(f"  {policy}: utilization={r['utilization_percent']:.2f}% "
                  f"missed={r['missed']}/{r['due']} "
                  f"missed by task={r['missed_by_task']}")
            # Keep output JSON compact: omit the full timeline by default.
            model_result["policies"][policy] = {key: val for key, val in r.items() if key != "timeline"}
        results["models"][model_name] = model_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
