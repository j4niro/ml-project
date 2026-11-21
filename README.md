# Projet ML : classification binaire (Spambase et Diabetes)

TAF MCE, IMT Atlantique, UE Machine Learning.
Groupe : BINI Kouassi, KOUASSI Emmanuel, NGUENGOUE WATCHO Ginger.

Ce projet met en œuvre, de façon collaborative (Git), un même workflow de
Machine Learning sur deux jeux de données de classification binaire, puis compare
plusieurs méthodes.

## Jeux de données

| Jeu | Source | Taille | Cible | Particularité |
|---|---|---|---|---|
| Spambase | [UCI](https://archive.ics.uci.edu/dataset/94/spambase) | 4 601 × 57 variables | `is_spam` (≈ 39 % de spams) | fréquences de mots/caractères, très asymétriques |
| Diabetes Health Indicators (BRFSS 2015) | [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | 253 680 × 21 variables | `Diabetes_binary` (≈ 14 % de positifs) | fort déséquilibre, doublons |

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
ajusté sur le train uniquement. Le choix du modèle et du seuil de décision se
fait par validation croisée sur le train ; le jeu de test ne sert qu'à la mesure
finale.

Les étapes du pipeline portent des noms fixes : `imputer`, `scaler`, `selector`,
`pca`, `model`.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `ml_workflow.py` | Toutes les fonctions du workflow (A : données, B : modèles et sélection, C : évaluation) |
| `01_spambase.ipynb` | Application du workflow à Spambase |
| `02_diabetes.ipynb` | Application du workflow à Diabetes (avec gestion du déséquilibre) |
| `test_data.py` | Tests unitaires des parties A et B (`pytest`) |
| `test_evaluation.py` | Tests unitaires de la partie C |
| `conftest.py` | Fixture partagée par les tests (petit jeu synthétique déséquilibré) |
| `data/` | Jeux de données |
| `requirements.txt` | Dépendances Python |
| `TACHES.md` | Répartition des tâches et règles de travail |
| `AI_USAGE.md` | Où / comment / pourquoi l'IA a été utilisée |

## Installation

Python 3.10 ou plus récent. Les notebooks fournis ont été exécutés avec Python 3.13 et les versions figées dans `requirements.txt`.

```bash
git clone <url-du-depot>
cd ml-project

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

Notebooks (à exécuter de haut en bas, « Restart & Run All ») :

```bash
jupyter notebook 01_spambase.ipynb
jupyter notebook 02_diabetes.ipynb
```

Le notebook Diabetes travaille sur un sous-échantillon pour le réglage des
hyperparamètres (jeu volumineux) ; l'entraînement final utilise le train complet.

Tests unitaires :

```bash
python -m pytest -q
```

Les graines aléatoires sont fixées (`RANDOM_STATE = 42`) : les résultats sont
reproductibles avec les versions de `requirements.txt`. Les chemins vers `data/` sont calculés à partir de `ml_workflow.py`, donc le code fonctionne quel que soit le dossier de lancement.

Durée d'exécution indicative (2 cœurs) : environ 2 min pour Spambase et environ 3 min pour Diabetes.

## Métriques

Les modèles sont comparés avec : accuracy, balanced accuracy,
précision, rappel, F1, MCC, ROC-AUC et PR-AUC (average precision). Sur Diabetes, l'accuracy est trompeuse (un modèle qui répond toujours « non » obtient 0,861) : on décide avec la PR-AUC, le F1, le rappel et la précision.

## Organisation du travail

Une branche par personne, puis Merge Request relue par un autre membre ;
voir `TACHES.md` pour la répartition et les règles Git.

## Utilisation de l'IA

Voir `AI_USAGE.md`.
