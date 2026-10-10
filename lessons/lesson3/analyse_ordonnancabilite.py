"""Cours 3 -- analyse d'ordonnancabilite sur un seul CPU simule.

Ce module explique l'EDF, le RM, la RTA et la demande processeur.
Hypotheses : taches independantes, periodiques, synchrones, preemptives ;
couts d'execution deterministes (et non WCET certifies) ; pas de surcout.
Unite de temps : 1 milliseconde ; C, T, D sont entiers positifs.
"""
from dataclasses import dataclass
from math import ceil, floor


@dataclass(frozen=True)
class Task:
    name: str
    C: int
    T: int
    D: int

    def __post_init__(self):
        if min(self.C, self.T, self.D) <= 0:
            raise ValueError("C, T, D doivent etre positifs")


@dataclass
class Job:
    task: Task
    arrival: int
    remaining: int
    end: int | None = None

    @property
    def absolute_deadline(self) -> int:
        return self.arrival + self.task.D


SCENARIOS = {
    "deadlines_courtes": [Task("T1", 3, 10, 4), Task("T2", 2, 12, 4)],
    "initial": [Task("T1", 2, 5, 5), Task("T2", 2, 7, 7), Task("T3", 3, 12, 12)],
    "modele_leger": [Task("T1", 2, 5, 5), Task("T2", 2, 7, 7), Task("T3", 2, 12, 12)],
    "periode_allongee": [Task("T1", 2, 6, 5), Task("T2", 2, 7, 7), Task("T3", 3, 12, 12)],
}


def utilization(tasks: list[Task]) -> float:
    return sum(t.C / t.T for t in tasks)


def response_time_analysis(task: Task, tasks: list[Task], limit: int = 100):
    """RTA classique pour D<=T ; seules les taches a priorites strictes plus hautes interferent.

    En cas de periodes identiques, un departage fixe explicite par nom est utilise.
    Retourne la suite w0, w1, ... jusqu'a convergence ou depassement de la deadline.
    """
    if any(t.D > t.T for t in tasks):
        raise ValueError("Cette RTA pedagogique exige D <= T pour toutes les taches")
    higher = [t for t in tasks if (t.T, t.name) < (task.T, task.name)]
    w = task.C
    steps = [w]
    for _ in range(limit):
        w_new = task.C + sum(ceil(w / high.T) * high.C for high in higher)
        steps.append(w_new)
        if w_new == w or w_new > task.D:
            break
        w = w_new
    return steps, steps[-1] <= task.D and steps[-1] == steps[-2]


def dbf(task: Task, interval: int) -> int:
    """Demande processeur (deadline-bound demand), taches periodiques D<=T."""
    if task.D > task.T:
        raise ValueError("Le cours traite les deadlines contraintes D<=T")
    return max(0, floor((interval - task.D) / task.T) + 1) * task.C


def simulate(tasks: list[Task], policy: str, horizon: int):
    if policy not in ("EDF", "RM"):
        raise ValueError("Politique EDF ou RM attendue")
    if horizon <= 0:
        raise ValueError("Horizon positif requis")
    jobs: list[Job] = []
    timeline: list[str] = []
    for now in range(horizon):
        for task in tasks:
            if now % task.T == 0:
                jobs.append(Job(task, arrival=now, remaining=task.C))
        ready = [j for j in jobs if j.remaining > 0]
        if not ready:
            timeline.append("IDLE")
            continue
        if policy == "EDF":
            selected = min(ready, key=lambda j: (j.absolute_deadline, j.arrival, j.task.name))
        else:
            selected = min(ready, key=lambda j: (j.task.T, j.task.name, j.arrival))
        selected.remaining -= 1
        timeline.append(selected.task.name)
        if selected.remaining == 0:
            selected.end = now + 1
    due = [j for j in jobs if j.absolute_deadline <= horizon]
    missed = [j for j in due if j.end is None or j.end > j.absolute_deadline]
    return timeline, jobs, due, missed


def run_case(name: str, tasks: list[Task], horizon: int = 420) -> None:
    print("\n" + "=" * 65)
    print(f"SCENARIO : {name} ; U = {utilization(tasks):.2%}")
    for t in tasks:
        print(f"  {t.name}: (C={t.C}, T={t.T}, D={t.D})")
    print("\nRTA (RM) :")
    for t in sorted(tasks, key=lambda t: (t.T, t.name)):
        steps, feasible = response_time_analysis(t, tasks)
        print(f"  {t.name}: {steps}, {'OK' if feasible else 'ECHEC'} (D={t.D})")
    print("\nDemande EDF dbf(t) pour quelques intervalles :")
    for span in (4, 5, 7, 12):
        demand = sum(dbf(t, span) for t in tasks)
        print(f"  t={span:2} ms : {demand:2} ms de calcul requis ; capacite {span:2} ms")
    print("\nSimulation :")
    for policy in ("EDF", "RM"):
        timeline, jobs, due, missed = simulate(tasks, policy, horizon)
        by_task = {t.name: sum(j.task.name == t.name for j in missed) for t in tasks}
        print(f"  {policy}: {len(missed)}/{len(due)} deadlines manquees, par tache {by_task}")
        print(f"  Chronologie 0..20 ms : {' '.join(timeline[:20])}")
    print("  Note : une simulation d'horizon fini n'est pas une preuve pour tout l'avenir.")


if __name__ == "__main__":
    for scenario_name, scenario in SCENARIOS.items():
        run_case(scenario_name, scenario)
