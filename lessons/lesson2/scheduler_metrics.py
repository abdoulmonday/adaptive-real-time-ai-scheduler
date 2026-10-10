"""TP 2 - Analyse de performances d'un ordonnanceur EDF/RM.

Réutilise le simulateur pédagogique dans scheduler.py.
Exemple : python scheduler_metrics.py --scenario overload --horizon 40

Modèle : un processeur, pas de temps d'une ms, tâches périodiques synchrones,
coûts déterministes, préemption et commutation sans surcoût. Les C sont des coûts
hypothétiques et NE sont PAS des WCET prouvés sur une machine physique.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean

from scheduler import Task, Job, simulate, plot_gantt


SCENARIOS = {
    "baseline": [
        Task("T1", C=1, T=4, D=4),
        Task("T2", C=2, T=5, D=5),
        Task("T3", C=2, T=10, D=10),
    ],
    "overload": [
        Task("T1", C=1, T=4, D=4),
        Task("T2", C=2, T=5, D=5),
        Task("T3", C=4, T=10, D=10),
    ],
}


def percentile(values: list[int], q: float) -> float | None:
    """Percentile par interpolation linéaire sur les données triées."""
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q doit être compris entre 0 et 100")
    data = sorted(values)
    index = (len(data) - 1) * q / 100
    lo = int(index)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (index - lo) * (data[hi] - data[lo])


def analyze(timeline: list[str], jobs: list[Job], horizon: int) -> dict:
    """N'évalue les deadlines que si elles sont atteintes avant l'horizon.

    Un job dont l'échéance est après horizon n'est NI réussi NI manqué.
    Un job dont l'échéance est atteinte mais non achevé est compté manqué.
    Les temps de réponse portent seulement sur les jobs terminés.
    """
    due = [j for j in jobs if j.absolute_deadline <= horizon]
    missed = [j for j in due if j.finish is None or j.finish > j.absolute_deadline]
    completed = [j for j in jobs if j.finish is not None]
    response_times = [j.finish - j.release for j in completed]
    waiting_times = [j.finish - j.release - j.task.C for j in completed]
    names = sorted({j.task.name for j in jobs})
    misses_per_task = {
        name: sum(j.task.name == name for j in missed) for name in names
    }
    due_per_task = {
        name: sum(j.task.name == name for j in due) for name in names
    }
    return {
        "due": len(due),
        "missed": len(missed),
        "met": len(due) - len(missed),
        "miss_rate_pct": 100 * len(missed) / len(due) if due else None,
        "completed": len(completed),
        "unfinished": sum(j.finish is None for j in jobs),
        "cpu_busy_pct": 100 * sum(t != "IDLE" for t in timeline) / len(timeline),
        "mean_response_ms": mean(response_times) if response_times else None,
        "p50_response_ms": percentile(response_times, 50),
        "p95_response_ms": percentile(response_times, 95),
        "p99_response_ms": percentile(response_times, 99),
        "mean_waiting_ms": mean(waiting_times) if waiting_times else None,
        "missed_by_task": misses_per_task,
        "due_by_task": due_per_task,
    }


def export_jobs_csv(jobs: list[Job], horizon: int, path: Path) -> None:
    """Enregistre le sort de chaque job, pour examen sur tableur."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task", "release_ms", "deadline_ms", "finish_ms", "response_ms", "status"
        ])
        for job in jobs:
            if job.absolute_deadline > horizon:
                status = "not_due_yet"
            elif job.finish is None:
                status = "missed_unfinished"
            elif job.finish > job.absolute_deadline:
                status = "missed_late"
            else:
                status = "on_time"
            writer.writerow([
                job.task.name, job.release, job.absolute_deadline,
                "" if job.finish is None else job.finish,
                "" if job.finish is None else job.finish - job.release,
                status,
            ])


def report(metrics: dict, policy: str) -> None:
    def fmt(value):
        return "n/d" if value is None else f"{value:.2f}"

    print(f"\n=== {policy} ===")
    print(f"Deadlines atteintes : {metrics['due']}")
    print(f"Respectées          : {metrics['met']}")
    print(f"Manquées            : {metrics['missed']}")
    print(f"Taux de dépassement : {fmt(metrics['miss_rate_pct'])} %")
    print(f"CPU réellement occupé : {fmt(metrics['cpu_busy_pct'])} %")
    print(f"Temps de réponse moyen (terminées) : {fmt(metrics['mean_response_ms'])} ms")
    print(f"Temps de réponse p95 (terminées)   : {fmt(metrics['p95_response_ms'])} ms")
    print(f"Temps d'attente moyen (terminées)  : {fmt(metrics['mean_waiting_ms'])} ms")
    print(f"Instances inachevées à l'horizon  : {metrics['unfinished']}")
    print("Deadlines manquées par tâche :", metrics["missed_by_task"])


def plot_missed_comparison(results: dict, path: Path) -> None:
    """Graphique simple : quels jobs sont pénalisés par chaque politique ?"""
    import matplotlib.pyplot as plt

    task_names = sorted(results["EDF"]["missed_by_task"])
    x = list(range(len(task_names)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x],
           [results["EDF"]["missed_by_task"][name] for name in task_names],
           width=0.4, label="EDF")
    ax.bar([i + 0.2 for i in x],
           [results["RM"]["missed_by_task"][name] for name in task_names],
           width=0.4, label="RM")
    ax.set_xticks(x, task_names)
    ax.set_ylabel("Échéances manquées (nombre)")
    ax.set_title("Répartition des échéances manquées par tâche")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP2 - métriques EDF/RM")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="overload")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results_tp2"))
    args = parser.parse_args()

    tasks = SCENARIOS[args.scenario]
    utilization = sum(task.C / task.T for task in tasks)
    print(f"Scénario : {args.scenario} ; charge théorique : {utilization:.0%}")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for policy in ("EDF", "RM"):
        timeline, jobs = simulate(tasks, policy=policy, horizon=args.horizon)
        metrics = analyze(timeline, jobs, horizon=args.horizon)
        report(metrics, policy)
        export_jobs_csv(jobs, args.horizon, args.output / f"jobs_{policy.lower()}.csv")
        plot_gantt(timeline, str(args.output / f"gantt_{policy.lower()}.png"), limit=40)
        results[policy] = metrics

    plot_missed_comparison(results, args.output / "missed_comparison.png")
    print(f"\nFichiers générés dans : {args.output.resolve()}")


if __name__ == "__main__":
    main()
