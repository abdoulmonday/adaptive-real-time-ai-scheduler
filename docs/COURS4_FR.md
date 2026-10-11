# Cours 4 — PyTorch : entraîner deux réseaux et mesurer la latence d'inférence

Ce cours prolonge les TP EDF / Rate Monotonic / RTA du projet **Adaptive Real-Time AI Scheduler**.

## Objectifs

- Construire deux perceptrons multicouches (MLP) pour Fashion-MNIST.
- Comprendre la boucle `forward → loss → backward → optimizer.step`.
- Entraîner deux modèles sur **le même jeu de données**, puis mesurer leur précision sur le jeu de test.
- Comparer leurs latences d'inférence CPU, à **batch size = 1**, via p50, p95, p99 et maximum observé.
- Introduire des budgets de calcul *hypothétiques* dans EDF et RM sans les confondre avec un WCET justifié.

## Architecture

| Modèle | Architecture | Paramètres |
|:--|:--|--:|
| SmallMLP | `784 -> 64 -> 10` | 50 890 |
| LargeMLP | `784 -> 256 -> 128 -> 10` | 235 146 |

Images Fashion-MNIST : 28 × 28 pixels en niveaux de gris, 10 catégories ; 60 000 exemples d'entraînement et 10 000 exemples de test.

Le *forward* calcule des **logits** ; `CrossEntropyLoss` s'utilise directement sur ces logits (ne pas ajouter de Softmax avant la perte).

## 1. Installation

Depuis la racine du dépôt :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
python lessons/lesson4/smoke_test.py
```

Sur PowerShell, l'activation est `.venv\Scripts\Activate.ps1`.

Le smoke-test n'utilise que des tenseurs aléatoires et ne prouve **ni précision ni généralisation**.

## 2. Vérifier d'abord le pipeline, sur un petit sous-ensemble

```bash
python lessons/lesson4/train_models.py \
  --epochs 1 --batch-size 128 --limit-train 1000 --limit-test 200
```

C'est seulement une vérification fonctionnelle. Les scores obtenus sur cet échantillon limité ne constituent **pas** des résultats de recherche finaux.

## 3. Entraîner les deux modèles

```bash
python lessons/lesson4/train_models.py --epochs 2 --batch-size 128
```

Au premier lancement, TorchVision doit télécharger Fashion-MNIST et nécessite une connexion réseau.

Dans la boucle principale :

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

- `zero_grad` remet à zéro les gradients accumulés.
- `model(images)` effectue la propagation avant.
- `criterion` mesure l'erreur sur les labels du lot.
- `backward` calcule les gradients par différentiation automatique.
- `step` actualise les paramètres.

Le script produit `outputs/models/small.pt`, `outputs/models/large.pt` et `outputs/training_summary.json`. Les poids sont sauvegardés en `state_dict`, puis rechargés avec `torch.load(..., weights_only=True)`.

## 4. Mesurer la latence d'inférence des modèles entraînés

```bash
python lessons/lesson4/benchmark_trained.py \
  --model-dir outputs/models \
  --samples 200 --warmup 30 --threads 1
```

Le script fixe l'entrée à `(1,1,28,28)`, passe le modèle en `eval()` et active `torch.inference_mode()`. Il chronomètre **le forward CPU seulement**, et non le téléchargement, le chargement du modèle, le prétraitement, l'attente d'un ordonnanceur ou le post-traitement.

Les **30 passages de chauffe** permettent de limiter l'impact du démarrage. Les **200 répétitions** forment la distribution expérimentale de latence.

| Indicateur | SmallMLP | LargeMLP |
|:--|--:|--:|
| Test accuracy | à mesurer | à mesurer |
| p50 (ms) | à mesurer | à mesurer |
| p95 (ms) | à mesurer | à mesurer |
| p99 (ms) | à mesurer | à mesurer |
| Maximum observé (ms) | à mesurer | à mesurer |

Des données différentes, la charge machine, la fréquence, les versions de PyTorch et le nombre de threads peuvent modifier les mesures.

**Limite scientifique :** p95 et maximum observé **ne sont pas des WCET certifiés**. À l'opposé, une preuve de temps réel exige une borne valide selon des hypothèses d'exécution justifiées.

## 5. Première intégration dans le simulateur

```bash
python lessons/lesson4/compare_schedulers.py \
  --benchmarks outputs/benchmarks.json --min-budget-ms 2

python lessons/lesson4/compare_schedulers.py \
  --benchmarks outputs/benchmarks.json --min-budget-ms 3
```

Ce code lit les latences empiriques et crée un **coût d'exécution fictif en millisecondes entières** pour une tâche T3, afin de comparer EDF/RM. L'arrondi et le budget minimal servent à faire des expériences de sensibilité ; ils ne donnent pas une preuve de WCET.

**Attention :** avec des latences de l'ordre d'une fraction de milliseconde, le pas de simulation d'une milliseconde peut masquer toutes les différences entre SmallMLP et LargeMLP. Une future version devrait adopter un pas plus fin et une vraie description des interruptions/transferts CPU/accélérateur.

## 6. Exercices

1. Pourquoi séparer entraînement, évaluation de la précision, puis benchmark d'inférence ?
2. Un modèle a p95 = 1,8 ms mais une exécution isolée dure 4 ms. Une deadline de 3 ms est-elle garantie ? Pourquoi ?
3. Si une tâche de classification prend 2 ms toutes les 12 ms et passe à 3 ms toutes les 12 ms, quelle variation de charge CPU introduit-elle ?
4. Mesurer les deux modèles avec trois graines (`--seed 1`, `--seed 2`, `--seed 3`) en sauvegardant chaque essai dans un `--output-dir` distinct.
5. Pourquoi la précision du LargeMLP n'est-elle pas automatiquement supérieure ?

## Références officielles

- PyTorch Quickstart : https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- Fashion-MNIST et DataLoader : https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Optimizing Model Parameters : https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- `torch.inference_mode()` : https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
- `torch.load` / poids : https://docs.pytorch.org/docs/stable/generated/torch.load.html
