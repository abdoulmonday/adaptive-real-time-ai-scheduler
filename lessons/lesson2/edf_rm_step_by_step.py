"""Cours EDF vs Rate Monotonic : simulations pas à pas (sans dépendances).

Modèle pédagogique : 1 processeur, tâches périodiques synchrones,
préemption aux instants entiers, C/T/D en ms, aucun surcoût de commutation,
aucun WCET vérifié matériellement. On continue les jobs même en retard.

Exécution : python cours_edf_rm_avance.py
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    name: str
    C: int  # demande de calcul par instance
    T: int  # période
    D: int  # deadline relative
    O: int = 0  # premier instant d'arrivée

@dataclass
class Job:
    task: Task
    arrival: int
    deadline: int
    remaining: int
    completed: int | None = None


def simulate(tasks: list[Task], policy: str, horizon: int):
    if policy not in ('EDF', 'RM'):
        raise ValueError('Politique inconnue')
    if horizon < 1 or any(min(t.C, t.T, t.D) <= 0 or t.O < 0 for t in tasks):
        raise ValueError('Paramètres invalides')
    jobs: list[Job] = []
    timeline: list[str] = []
    for now in range(horizon):
        # Une deadline ABSOLUE est fixée à CHAQUE création d'instance.
        for task in tasks:
            if now >= task.O and (now - task.O) % task.T == 0:
                jobs.append(Job(task, now, now + task.D, task.C))

        ready = [j for j in jobs if j.remaining > 0]
        if not ready:
            timeline.append('-')
            continue

        # EDF : deadline absolue la plus proche ; RM : période la plus courte.
        # En cas d'égalité : instance la plus ancienne, puis nom de tâche.
        if policy == 'EDF':
            selected = min(ready, key=lambda j: (j.deadline, j.arrival, j.task.name))
        else:
            selected = min(ready, key=lambda j: (j.task.T, j.arrival, j.task.name))

        selected.remaining -= 1  # un pas de 1 ms
        timeline.append(selected.task.name)
        if selected.remaining == 0:
            selected.completed = now + 1
    return timeline, jobs


def print_report(tasks: list[Task], policy: str, horizon: int = 30, preview: int = 15):
    timeline, jobs = simulate(tasks, policy, horizon)
    due = [j for j in jobs if j.deadline <= horizon]
    misses = [j for j in due if j.completed is None or j.completed > j.deadline]
    U = sum(t.C/t.T for t in tasks)
    print(f'\n{policy}: U={U:.2%}, horizon={horizon} ms, manquées={len(misses)}/{len(due)}')
    print('t:    ' + ' '.join(f'{i:>2}' for i in range(min(preview, horizon))))
    print('CPU:  ' + ' '.join(f'{n:>2}' for n in timeline[:preview]))
    print('instance | arrivée | deadline absolue | fin | réponse | verdict')
    for job in jobs:
        if job.deadline > horizon:
            continue
        finish = job.completed
        response = None if finish is None else finish - job.arrival
        status = 'MANQUÉE' if finish is None or finish > job.deadline else 'ok'
        print(f'{job.task.name:>8} | {job.arrival:>7} | {job.deadline:>16} | {str(finish):>3} | {str(response):>7} | {status}')
    return timeline, jobs


if __name__ == '__main__':
    edf_failure = [Task('T1', 3, 10, 4), Task('T2', 2, 12, 4)]
    comparison = [Task('T1', 2, 5, 5), Task('T2', 2, 7, 7), Task('T3', 3, 12, 12)]
    lower_c = [Task('T1', 2, 5, 5), Task('T2', 2, 7, 7), Task('T3', 2, 12, 12)]
    longer_t = [Task('T1', 2, 6, 5), Task('T2', 2, 7, 7), Task('T3', 3, 12, 12)]
    scenarios = [
        ('EDF peut échouer avec U < 100% si D < T', edf_failure, 16),
        ('EDF réussit mais RM rate une échéance', comparison, 24),
        ('Correction A : diminuer C3 de 3 à 2', lower_c, 24),
        ('Correction B : allonger T1 de 5 à 6', longer_t, 24),
    ]
    for name, tasks, horizon in scenarios:
        print('\n' + '=' * 72 + '\n' + name)
        for policy in ('EDF', 'RM'):
            print_report(tasks, policy, horizon, preview=min(horizon, 15))
