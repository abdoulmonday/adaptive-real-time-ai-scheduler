"""Cours 1 — mini-simulateur discret EDF/RM (une unité = 1 ms).

Hypothèses : un CPU, préemption à la milliseconde, tâches indépendantes,
arrivées périodiques synchrones, durées d'exécution déterministes et
coût de commutation nul. Ce simulateur ne démontre pas un WCET réel.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Task:
    name: str
    C: int   # Coût d'exécution en millisecondes
    T: int   # Période en millisecondes
    D: int   # Deadline relative en millisecondes


@dataclass
class Job:
    task: Task
    release: int
    remaining: int
    finish: Optional[int] = None

    @property
    def absolute_deadline(self) -> int:
        return self.release + self.task.D


def simulate(tasks: list[Task], policy: str, horizon: int):
    """Retourne la chronologie (1 case/ms) et les instances créées."""
    if policy not in {"EDF", "RM"}:
        raise ValueError("La politique doit être EDF ou RM.")
    if horizon <= 0 or any(t.C <= 0 or t.T <= 0 or t.D <= 0 for t in tasks):
        raise ValueError("Horizon et paramètres C, T, D doivent être positifs.")

    jobs: list[Job] = []
    timeline: list[str] = []

    for now in range(horizon):
        # 1. Libérer les nouvelles instances périodiques.
        for task in tasks:
            if now % task.T == 0:
                jobs.append(Job(task, release=now, remaining=task.C))

        # 2. Chercher les instances encore prêtes.
        ready = [job for job in jobs if job.remaining > 0]
        if not ready:
            timeline.append("IDLE")
            continue

        # 3. Choisir UNE instance pour la prochaine milliseconde.
        if policy == "EDF":
            chosen = min(
                ready,
                key=lambda j: (j.absolute_deadline, j.release, j.task.name),
            )
        else:  # RM : priorité fixe, période plus courte d'abord.
            chosen = min(
                ready,
                key=lambda j: (j.task.T, j.release, j.task.name),
            )

        # 4. Exécuter exactement 1 ms, puis réévaluer au prochain pas.
        chosen.remaining -= 1
        timeline.append(chosen.task.name)
        if chosen.remaining == 0:
            chosen.finish = now + 1

    return timeline, jobs


def summarize(jobs: list[Job], horizon: int):
    """Ne compte que les échéances effectivement atteintes dans l'horizon."""
    due = [j for j in jobs if j.absolute_deadline <= horizon]
    missed = [
        j for j in due
        if j.finish is None or j.finish > j.absolute_deadline
    ]
    return len(due), len(missed)


def display(tasks: list[Task], policy: str, horizon: int, preview: int = 30):
    timeline, jobs = simulate(tasks, policy, horizon)
    due, missed = summarize(jobs, horizon)
    print(f"\n{policy} — horizon : {horizon} ms")
    print("Occupation CPU :", " ".join(timeline[:preview]))
    print(f"Échéances évaluées : {due} ; manquées : {missed}")
    if due:
        print(f"Taux de dépassement : {100 * missed / due:.2f} %")
    return timeline, jobs


def plot_gantt(timeline: list[str], path: str, limit: int = 40):
    """Sauvegarde un diagramme simple ; nécessite matplotlib."""
    import matplotlib.pyplot as plt

    visible = timeline[:limit]
    names = sorted(set(visible) - {"IDLE"})
    fig, ax = plt.subplots(figsize=(11, 2.2 + 0.55 * len(names)))
    for row, name in enumerate(names):
        start = None
        for t in range(len(visible) + 1):
            active = t < len(visible) and visible[t] == name
            if active and start is None:
                start = t
            if not active and start is not None:
                ax.broken_barh([(start, t - start)], (row - 0.35, 0.7))
                start = None
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, len(visible))
    ax.set_xticks(range(0, len(visible) + 1, 2))
    ax.set_xlabel("Temps (ms)")
    ax.set_title("Diagramme de Gantt des tâches")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    # Premier scénario : 85 % de charge CPU.
    baseline = [
        Task("T1", C=1, T=4, D=4),
        Task("T2", C=2, T=5, D=5),
        Task("T3", C=2, T=10, D=10),
    ]
    utilization = sum(t.C / t.T for t in baseline)
    print(f"Charge CPU du premier scénario : {utilization:.0%}")

    for policy in ("EDF", "RM"):
        timeline, _ = display(baseline, policy, horizon=40, preview=20)
        plot_gantt(timeline, f"gantt_{policy.lower()}.png")

    # Scénario où EDF réussit et RM manque des deadlines.
    contrast = [
        Task("T1", C=2, T=5, D=5),
        Task("T2", C=2, T=7, D=7),
        Task("T3", C=3, T=12, D=12),
    ]
    print(f"\nCharge CPU du second scénario : {sum(t.C/t.T for t in contrast):.2%}")
    for policy in ("EDF", "RM"):
        display(contrast, policy, horizon=420, preview=30)
