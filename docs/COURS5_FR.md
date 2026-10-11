# Cours 5 (début) — Précision temporelle et sélection de modèles d'IA

## Résultats du cours 4 (communiqués par l'utilisateur)

Les deux modèles ont été entraînés sur Fashion-MNIST pendant 3 époques puis évalués sur le jeu de test de 10 000 images. L'entraînement a été effectué avec CUDA, tandis que les durées d'inférence ont été chronométrées sur **CPU, batch=1, un thread, 50 échauffements et 500 mesures**.

| Indicateur | Small MLP | Large MLP |
|---|---:|---:|
| Paramètres | 50 890 | 235 146 |
| Précision test | 84,69 % | 86,66 % |
| p50 CPU | 0,0318 ms | 0,0660 ms |
| p95 CPU | 0,0504 ms | 0,1038 ms |
| Max observé CPU | 0,1176 ms | 0,2566 ms |

Le grand modèle gagne **1,97 point de pourcentage** de précision et son p95 est environ **2,06 fois** celui du petit modèle. Une seule exécution ne permet pas encore de conclure à la significativité statistique de cette différence.

## Pourquoi l'ancienne simulation ne distingue pas les modèles

Le TP 4 imposait `--min-budget-ms 2` et manipulait des coûts d'exécution entiers en millisecondes. Les p95 de 0,0504 et 0,1038 ms sont **tous deux remplacés par C3=2 ms**, ce qui produit le même ensemble de tâches et les mêmes statistiques de deadlines.

## Nouvelle approche : modèle à événements, unité = microseconde

`lessons/lesson5/microsecond_scheduler.py` prend comme entrée `outputs/benchmarks.json`, convertit les p95 en microsecondes et les **arrondit vers le haut** à l'entier. Les coûts *supposés* deviennent :

- Small : 0,0504 ms = 50,4 µs → **51 µs**
- Large : 0,1038 ms = 103,8 µs → **104 µs**

Ce nouvel outil ne chronomètre pas les réseaux pendant la simulation. C'est toujours un modèle déterministe, **sans WCET justifié**, sur **un seul processeur préemptif**, où le coût de commutation est nul.

### Expérience A : deadline T3 = 12 ms

```bash
python lessons/lesson5/microsecond_scheduler.py --benchmarks outputs/benchmarks.json --deadline-us 12000
```

Sur l'horizon de 420 ms, les deux modèles respectent 179/179 deadlines avec EDF et RM dans ce modèle. Les charges estimées sont **69,00 %** et **69,44 %**. Les autres tâches restent entièrement synthétiques : T1=(2000,5000,5000) et T2=(2000,7000,7000), en µs.

**Interprétation :** aucun modèle n'est en difficulté temporelle dans ce scénario. Selon la métrique de test, le grand modèle est un candidat naturel si la priorité est la précision et si les contraintes matérielles réelles le permettent.

### Expérience B : deadline T3 = 80 µs

```bash
python lessons/lesson5/microsecond_scheduler.py --benchmarks outputs/benchmarks.json --deadline-us 80
```

**80 µs est une échéance fictive volontairement très serrée**, non imposée par le sujet de thèse ou Fashion-MNIST.

| Politique | Small (51 µs) | Large (104 µs) |
|---|---:|---:|
| EDF, dépassements / 179 | 0 | 35 |
| RM, dépassements / 179 | 24 | 35 |

Le Large MLP ne peut pas satisfaire une deadline de 80 µs **même sans concurrence**, si son coût est de 104 µs. Pour le Small MLP, EDF peut préempter les tâches synthétiques lorsque la nouvelle instance T3 a une échéance absolue très proche. Avec RM, T3 garde une priorité inférieure aux tâches de plus courte période et certaines de ses instances sont retardées.

Ce résultat constitue une *comparaison conditionnelle*, pas une garantie temporelle sur CPU réel.

### Expérience C : heuristique de sélection

```bash
python lessons/lesson5/select_model.py --benchmarks outputs/benchmarks.json --budget-us 80
python lessons/lesson5/select_model.py --benchmarks outputs/benchmarks.json --budget-us 12000
```

Cette heuristique filtre les modèles par rapport au p95 arrondi **puis choisit celui dont la précision de test est la plus élevée** : petit modèle pour 80 µs, grand pour 12 ms. Le choix n'est PAS une analyse globale d'ordonnançabilité et n'utilise aucune garantie WCET.

## Exercices

1. Pourquoi l'intervalle de 80 µs conduit-il à 35 dépassements pour le grand modèle, même avec EDF ?
2. Pourquoi RM peut-il rater des deadlines pour le petit modèle alors qu'une exécution isolée de 51 µs respecte la limite de 80 µs ?
3. Pourquoi le p95 arrondi de 51 µs n'est-il toujours pas un WCET ?
4. Proposer une règle de sélection plus robuste : tenir compte de la latence maximale observée, des tâches en concurrence et d'une marge de sécurité. Expliquer pourquoi cela reste différent d'une preuve temps réel.

## Suite de la formation

Le prochain chapitre introduira la **quantification**, puis le **pruning structuré**, avec mesures comparatives de précision, de taille mémoire, de p50/p95/p99 et des deadlines simulées. Le véritable système adaptatif devra aussi étudier les interférences, l'ordonnancement non préemptif et l'hétérogénéité CPU/GPU/TPU.
