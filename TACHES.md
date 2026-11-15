# Répartition des tâches (3 personnes)

Rendu : dépôt envoyé à elsa.dupraz@imt-atlantique.fr avant le **21/11 à 23h** (rôle *Developer*).

## Qui fait quoi

### A — Données & pré-traitement (`feature/data-preprocessing`)
- [ ] `load_spambase`, `load_diabetes`, `split_features_target`, `describe_dataset`
- [ ] `inject_missing_values` (valeurs manquantes simulées, MCAR)
- [ ] `build_pipeline` : imputation (moyenne/médiane) → centrage-réduction → sélection (appelle `make_selector` de B) → ACP → modèle
- [ ] `split_data` (stratifié), `subsample`
- [ ] `test_data.py`
- [ ] Notebooks, sections 1-3 (Spambase) et 1-2 (Diabetes) ; README, `requirements.txt`

### B — Modèles & sélection de variables (`feature/models-selection`)
- [ ] `get_models` (LogReg, KNN, RandomForest, HistGB, SVM-RBF ; option `class_weight`)
- [ ] `make_selector` (anova, mutual_info, l1, rf → garde les k meilleures variables)
- [ ] `tune_model` (GridSearchCV), `feature_scores`, `selected_features`
- [ ] `test_models.py`
- [ ] Notebooks, sections 4-6 (comparaison, sélection de variables, réglage)

### C — Évaluation, déséquilibre & qualité (`feature/evaluation-imbalance`)
- [ ] `get_scores`, `evaluate_model`, `compare_models`, `cross_validate_models`
- [ ] `find_best_threshold`, `oof_scores`
- [ ] `plot_roc_pr`, `plot_confusion`, `plot_pca_variance`, `plot_feature_scores`
- [ ] `test_evaluation.py`
- [ ] Notebooks : Diabetes section 3 (déséquilibre) ; sections 7-9 des deux notebooks (évaluation finale, conclusions, usage de l'IA) ; relecture finale

## Contrat entre nous
Les **signatures** dans `ml_workflow.py` sont fixées : ne pas les changer sans prévenir les autres. Chacun travaille uniquement dans la section de sa personne (évite les conflits).

Ordre conseillé : **A pousse en premier** (chargement, split, `build_pipeline`) car B et C en dépendent. Pendant ce temps B et C écrivent leurs fonctions et testent avec de petites données synthétiques (`conftest.py`).

## Règles Git
1. Jamais de push direct sur `main` : une branche par personne, puis *Merge Request* relue par un autre membre.
2. Commits petits, message clair (`feat: ...`, `test: ...`, `fix: ...`, `docs: ...`).
3. `git pull --rebase origin main` avant chaque `push`.
4. Notebooks : **vider les sorties avant de commit** (`jupyter nbconvert --clear-output --inplace 01_spambase.ipynb`) et ne les éditer qu'une personne à la fois (chacun ne touche que ses sections). Les notebooks sont exécutés et sauvegardés avec leurs résultats **une seule fois, à la fin**.

## Avant le rendu
- [ ] `pytest -q` vert
- [ ] Les deux notebooks s'exécutent de haut en bas (« Restart & Run All »)
- [ ] Conclusions rédigées à partir des résultats réels (quelle méthode choisir ?)
- [ ] `AI_USAGE.md` complété honnêtement
- [ ] Elsa Dupraz ajoutée au dépôt (rôle Developer) et mail envoyé
