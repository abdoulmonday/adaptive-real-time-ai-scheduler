"""TP EDF avancé : instances périodiques à offsets et deadlines contraintes.

Hypothèses : 1 CPU, préemption à chaque ms, aucun coût de changement de contexte,
coûts de calcul entiers et fixes (pas des WCET prouvés sur une machine réelle).
"""
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class Task:
    name: str
    C: int       # Temps de calcul total (ms)
    T: int       # Période entre deux activations (ms)
    D: int       # Deadline relative (ms)
    offset: int  # Première activation (ms)


@dataclass
class Job:
    task: Task
    release: int
    remaining: int
    finish: int | None = None

    @property
    def deadline(self):
        # Calculée à partir de l'arrivée de cette instance ; ne varie pas.
        return self.release + self.task.D


def simulate_edf(tasks: list[Task], horizon: int):
    jobs = []
    timeline = []
    for now in range(horizon):
        # De nouvelles instances sont libérées aux instants offset + k*T.
        for task in tasks:
            if now >= task.offset and (now - task.offset) % task.T == 0:
                jobs.append(Job(task, release=now, remaining=task.C))

        ready = [j for j in jobs if j.remaining > 0]
        if not ready:
            timeline.append("IDLE")
            continue

        # EDF préemptif : plus petite deadline absolue en premier.
        chosen = min(ready, key=lambda j: (j.deadline, j.release, j.task.name))
        chosen.remaining -= 1   # Une ms de calcul simulée.
        timeline.append(chosen.task.name)
        if chosen.remaining == 0:
            chosen.finish = now + 1
    return timeline, jobs


def print_report(timeline: list[str], jobs: list[Job], horizon: int):
    print(f"\nCHRONOLOGIE EDF, de 0 à {horizon} ms")
    for task_name, part in groupby(enumerate(timeline), key=lambda v: v[1]):
        slots = list(part)
        print(f"[{slots[0][0]:2d}, {slots[-1][0] + 1:2d}[  {task_name}")

    print("\nINSTANCES")
    print("Job   arrivée  deadline  fin  réponse  résultat")
    due = 0
    missed = 0
    for j in jobs:
        if j.deadline <= horizon:
            due += 1
            late = j.finish is None or j.finish > j.deadline
            missed += int(late)
            result = "MANQUÉE" if late else "respectée"
        else:
            result = "pas encore échue"
        response = j.finish - j.release if j.finish is not None else None
        print(f"{j.task.name:3s}   {j.release:5d}    {j.deadline:5d}   {str(j.finish):>4s}   {str(response):>5s}   {result}")
    print(f"\nDeadlines échues dans l'horizon : {due} ; manquées : {missed}")


if __name__ == "__main__":
    tasks = [
        Task("T1", C=5, T=12, D=12, offset=0),
        Task("T2", C=2, T=10, D=4, offset=3),
        Task("T3", C=2, T=14, D=5, offset=6),
    ]
    horizon = 30
    timeline, jobs = simulate_edf(tasks, horizon)
    print_report(timeline, jobs, horizon)
