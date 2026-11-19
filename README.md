# Projet ML — Classification binaire (Spambase & Diabetes)

TAF MCE — IMT Atlantique — UE Machine Learning.
Groupe : **BINI Kouassi, KOUASSI Emmanuel, NGUENGOUE WATCHO Ginger**.

Ce projet met en œuvre, de façon collaborative (Git), un même workflow de
Machine Learning sur deux jeux de données de classification binaire, puis compare
plusieurs méthodes.

## Jeux de données

| Jeu | Source | Taille | Cible | Particularité |
|---|---|---|---|---|
| **Spambase** | [UCI](https://archive.ics.uci.edu/dataset/94/spambase) | 4 601 × 57 variables | `is_spam` (≈ 39 % de spams) | fréquences de mots/caractères, très asymétriques |
| **Diabetes Health Indicators** (BRFSS 2015) | [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | 253 680 × 21 variables | `Diabetes_binary` (≈ 14 % de positifs) | **fort déséquilibre**, doublons |

Les fichiers sont dans `data/`. Aucun des deux jeux ne contient de valeurs
manquantes : elles sont simulées (MCAR) par `inject_missing_values` pour comparer
des stratégies d'imputation.

## Workflow

1. Import des données
2. Pré-traitement : imputation (moyenne/médiane), centrage-réduction, ACP
3. Split train/test stratifié
4. Entraînement, avec sélection de variables
5. Validation et comparaison des modèles

Principe anti-fuite : tout le pré-traitement est dans un `sklearn.Pipeline`
ajusté sur le **train uniquement**. Le choix du modèle et du seuil de décision se
fait par validation croisée sur le train ; le jeu de test ne sert qu'à la mesure
finale.

Les étapes du pipeline portent des noms fixes : `imputer`, `scaler`, `selector`,
`pca`, `model`.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `ml_workflow.py` | Fonctions du workflow (sections A : données, B : modèles, C : évaluation) |
| `01_spambase.ipynb` | Application du workflow à Spambase |
| `02_diabetes.ipynb` | Application du workflow à Diabetes (avec gestion du déséquilibre) |
| `test_data.py`, `test_models.py`, `test_evaluation.py` | Tests unitaires (`pytest`) |
| `conftest.py` | Fixture partagée par les tests (petit jeu synthétique déséquilibré) |
| `data/` | Jeux de données |
| `requirements.txt` | Dépendances Python |
| `TACHES.md` | Répartition des tâches et règles de travail |
| `AI_USAGE.md` | Où / comment / pourquoi l'IA a été utilisée |

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
git clone <url-du-depot>
cd ml-project

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

**Notebooks** (à exécuter de haut en bas, « Restart & Run All ») :

```bash
jupyter notebook 01_spambase.ipynb
jupyter notebook 02_diabetes.ipynb
```

Le notebook Diabetes travaille sur un sous-échantillon pour le réglage des
hyperparamètres (jeu volumineux) ; l'entraînement final utilise le train complet.

**Tests unitaires** :

```bash
pytest -q
```

Les graines aléatoires sont fixées (`RANDOM_STATE = 42`) : les résultats sont
reproductibles.

## Métriques

Les modèles sont comparés avec : accuracy, balanced accuracy,
précision, rappel, F1, MCC, ROC-AUC et PR-AUC (average precision). 

## Organisation du travail

Une branche par personne, puis Pull Request relue par un autre membre ;
voir `TACHES.md` pour la répartition et les règles Git.

## Utilisation de l'IA

Voir `AI_USAGE.md`.
