# Notes de cours — systèmes temps réel et IA embarquée

## Cours 1 : fondamentaux

Une tâche périodique est décrite par `τ = (C, T, D)` : coût CPU, période et deadline relative. Une **instance** créée à `r` possède une deadline absolue `d = r + D`, qui reste fixe. À chaque événement, EDF choisit l'instance prête avec la plus petite deadline absolue ; RM donne la priorité fixe à la tâche de période la plus courte.

Notre modèle : un CPU, préemption aux millisecondes entières, coûts fixes, tâches indépendantes, pas de coût de commutation.

- Référence : `T1=(1,4,4), T2=(2,5,5), T3=(2,10,10)` ; `U=85 %`.
- Surcharge : remplacer `C3=2` par `C3=4` ; `U=105 %`, donc impossible de respecter toutes les échéances indéfiniment.

## Cours 2 : temps de réponse et priorités

Temps de réponse d'une instance : `R = finish - release`. Temps d'attente dans notre modèle : `W = R - C`. Une échéance est manquée si `finish > release + D`, ou si elle est atteinte avant la fin.

EDF avec offsets : arrivée de la k-ième instance (k commençant à 0) : `r = O + k*T`. Exemple dans `edf_simulation_offsets.py`.

## Cours 3 : ordonnançabilité

- EDF pour tâches indépendantes préemptives avec `D=T` sur un seul processeur : `U <= 1` est nécessaire et suffisant sous les hypothèses usuelles.
- Pour `D<=T`, la demande processeur dans un intervalle de longueur t s'écrit `dbf_i(t)=max(0, floor((t-D_i)/T_i)+1)*C_i`. La vérifier pour *quelques* intervalles ne constitue pas une preuve exhaustive.
- RM : borne suffisante de Liu–Layland : `U <= n*(2^(1/n)-1)` pour deadlines implicites. Un échec de ce test ne prouve pas que RM est impossible.
- RTA : `R_i(next)=C_i+sum_{j dans hp(i)} ceil(R_i(current)/T_j)*C_j`. Pour `T1=(2,5,5), T2=(2,7,7), T3=(3,12,12)`, on obtient `3 -> 7 -> 9 -> 11 -> 13 -> 13` pour T3, qui rate `D3=12` sous RM.

## Cours 4 : lien avec les réseaux neuronaux

On entraîne deux modèles sur Fashion-MNIST, on mesure *séparément* la qualité prédictive et les latences sur CPU, puis on introduit des budgets **hypothétiques** dans le simulateur pour explorer la sensibilité. Les latences p50/p95/max observées **ne sont pas des bornes WCET**. Un simulateur à un CPU ne représente pas automatiquement la préemption d'un GPU/TPU.

Voir `lessons/lesson4/README.md`.
