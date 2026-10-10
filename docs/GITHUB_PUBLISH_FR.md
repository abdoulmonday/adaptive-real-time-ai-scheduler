# Publication GitHub — pas à pas

## Cas A : nouveau dépôt (recommandé)

1. Créer un dépôt **vide** appelé `adaptive-real-time-ai-scheduler` sur https://github.com/new. Ne pas ajouter de README ou de `.gitignore` dans le formulaire.
2. Décompresser l'archive localement. Sous Windows PowerShell :

```powershell
Expand-Archive -Path .\adaptive-real-time-ai-scheduler.zip -DestinationPath .
cd .\adaptive-real-time-ai-scheduler
```

Si l'archive est ailleurs, positionner d'abord le terminal dans son dossier (`cd` ou Explorateur > Ouvrir dans Terminal).

3. Initialiser Git et consulter les fichiers :

```bash
git init -b main
git status
git add .
git commit -m "Initial research prototype: EDF, RM, RTA and PyTorch lessons"
```

Si Git réclame l'identité locale, la configurer avec les informations que l'on souhaite associer à ses commits :

```bash
git config --global user.name "Your Name"
git config --global user.email "your-public-or-noreply-email@example.com"
```

Refaire ensuite le commit.

4. **Remplacer `TON-IDENTIFIANT` par son véritable nom d'utilisateur GitHub**, puis ajouter le dépôt distant :

```bash
git remote add origin https://github.com/TON-IDENTIFIANT/adaptive-real-time-ai-scheduler.git
git push -u origin main
```

GitHub peut ouvrir une fenêtre d'authentification ; utiliser le flux sécurisé du Git Credential Manager / GitHub CLI / clés SSH. Ne jamais publier un token GitHub dans un dépôt ou une conversation. Le mot de passe du compte ne permet plus l'authentification Git HTTPS de cette façon.

5. Vérifier sur `https://github.com/TON-IDENTIFIANT/adaptive-real-time-ai-scheduler` que `README.md`, `docs/`, `lessons/` et `.github/workflows/tests.yml` sont présents et que les tests GitHub Actions passent.

## Cas B : dépôt GitHub déjà créé (et non vide)

Ne pas exécuter `git init` dans le dépôt distant ; préférer le cloner :

```bash
git clone https://github.com/TON-IDENTIFIANT/NOM-DEPOT.git
cd NOM-DEPOT
```

Copier le contenu du dossier extrait `adaptive-real-time-ai-scheduler` dans la copie locale clonée, **sans supprimer ni écraser aveuglément les fichiers préexistants** ; comparer d'abord les versions si des noms sont identiques.

```bash
git status
git add .
git commit -m "Add real-time AI scheduling courses and reproducible simulations"
git push origin main
```

Si la branche existante n'est pas `main`, vérifier `git branch --show-current` et pousser la branche affichée. Ne pas utiliser `--force` pour contourner un conflit.

## Mettre à jour plus tard

Dans le dossier du dépôt local :

```bash
git status
git add .
git commit -m "Add Fashion-MNIST training and inference benchmarks"
git push
```

Le dossier `data/`, les poids `*.pt`, `outputs/` et `.venv/` sont exclus via `.gitignore` pour ne pas publier de gros fichiers ou des données accidentellement.

## Mini-checklist de qualité avant publication

- `python -m unittest discover -s lessons/lesson2 -p 'test_*.py' -v`
- `python -m unittest discover -s lessons/lesson3 -p 'test_*.py' -v`
- `python lessons/lesson4/smoke_test.py`
- `git status` avant chaque commit pour détecter d'éventuels fichiers non destinés à la publication.
