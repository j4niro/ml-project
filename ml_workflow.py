"""Workflow ML pour la classification binaire (Spambase & Diabetes).

Workflow : import -> pré-traitement (imputation, centrage-réduction, ACP) -> split
-> entraînement avec sélection de variables -> validation.

Principe anti-fuite : tout le pré-traitement est dans un sklearn.Pipeline ajusté
sur le TRAIN uniquement."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,  
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold,  
    cross_val_predict, cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

SPAMBASE_COLUMNS: List[str] = (
    [f"word_freq_{w}" for w in (
        "make address all 3d our over remove internet order mail receive will "
        "people report addresses free business email you credit your font 000 "
        "money hp hpl george 650 lab labs telnet 857 data 415 85 technology "
        "1999 parts pm direct cs meeting original project re edu table "
        "conference").split()]
    + [f"char_freq_{c}" for c in (";", "(", "[", "!", "$", "#")]
    + ["capital_run_length_average", "capital_run_length_longest",
       "capital_run_length_total", "is_spam"]
)


# ============================================================================
#  A — Données & pré-traitement
# ============================================================================
def load_spambase(path: str='data/spambase.data') -> pd.DataFrame:
    """Load the UCI Spambase dataset (57 features + binary target ``is_spam``)."""
    raise NotImplementedError('TODO (personne A)')


def load_diabetes(path: str='data/diabetes_binary_health_indicators_BRFSS2015.csv') -> pd.DataFrame:
    """Load the Diabetes Health Indicators dataset (21 features + ``Diabetes_binary``)."""
    raise NotImplementedError('TODO (personne A)')


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from a dataframe and the name of its target column."""
    raise NotImplementedError('TODO (personne A)')


def describe_dataset(df: pd.DataFrame, target: str) -> pd.Series:
    """Quick summary: size, missing values, duplicates, class balance."""
    raise NotImplementedError('TODO (personne A)')


def inject_missing_values(X: pd.DataFrame, frac: float=0.05, seed: int=RANDOM_STATE) -> pd.DataFrame:
    """Randomly (MCAR) replace a fraction of the cells by NaN.

Neither dataset contains missing values; this function lets us test and
compare imputation strategies in a controlled way."""
    raise NotImplementedError('TODO (personne A)')


def build_pipeline(model: BaseEstimator, imputer_strategy: str='median', scale: bool=True, selector: Optional[str]=None, k: int=10, pca_components: Optional[float | int]=None, random_state: int=RANDOM_STATE) -> Pipeline:
    """Build ``imputer -> scaler -> [feature selection] -> [PCA] -> model``.

Parameters
----------
imputer_strategy : 'mean' or 'median' (missing values are replaced by it).
scale : center and reduce the data (StandardScaler).
selector : None, 'anova', 'mutual_info', 'l1' (Lasso-like logistic
    regression) or 'rf' (random-forest importances). Keeps the ``k`` best.
pca_components : None (no PCA), an int (number of axes) or a float in
    (0, 1) (fraction of explained variance to keep)."""
    raise NotImplementedError('TODO (personne A)')


def split_data(X, y, test_size: float=0.2, seed: int=RANDOM_STATE):
    """Stratified train/test split (keeps the class proportions in both parts)."""
    raise NotImplementedError('TODO (personne A)')


def subsample(X, y, n: int, seed: int=RANDOM_STATE):
    """Stratified subsample of ``n`` rows (to speed up tuning on large data)."""
    raise NotImplementedError('TODO (personne A)')



# ============================================================================
# B — Modèles & sélection de variables
# ============================================================================
def make_selector(method: str, k: int=10, random_state: int=RANDOM_STATE):
    """Feature selector keeping the ``k`` best features."""
    raise NotImplementedError('TODO (personne B)')


def get_models(class_weight: Optional[str]=None, include_svm: bool=True, random_state: int=RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """Candidate classifiers. ``class_weight='balanced'`` handles imbalance."""
    raise NotImplementedError('TODO (personne B)')


def tune_model(pipeline: Pipeline, param_grid: dict, X, y, scoring: str='f1', cv: int=3, n_samples: Optional[int]=None) -> GridSearchCV:
    """Grid search with cross-validation on the training data only.

``n_samples`` limits the number of rows used for the search (large data);
the best model is refit on all the rows given."""
    raise NotImplementedError('TODO (personne B)')


def feature_scores(X: pd.DataFrame, y, method: str='anova') -> pd.Series:
    """Univariate score of each feature (higher = more related to the target)."""
    raise NotImplementedError('TODO (personne B)')


def selected_features(fitted_pipeline: Pipeline, feature_names: Iterable[str]) -> List[str]:
    """Names of the features kept by the 'selector' step of a fitted pipeline."""
    raise NotImplementedError('TODO (personne B)')



# ============================================================================
# C — Évaluation, déséquilibre & graphiques
# ============================================================================

# Ajout d'un helper _predict_labels() pour éviter de réecrire la même logique chez evaluate_model et plot_confusion
def _predict_labels(fitted_pipeline: Pipeline, X, threshold: Optional[float] = None) -> np.ndarray:
    """Labels 0/1. Sans seuil : ``predict`` ; avec seuil : ``score >= threshold``."""
    if threshold is None:
        return np.asarray(fitted_pipeline.predict(X))
    return (get_scores(fitted_pipeline, X) >= threshold).astype(int)


def get_scores(fitted_pipeline: Pipeline, X) -> np.ndarray:
    """Continuous score of the positive class (probability or decision value)."""
    if hasattr(fitted_pipeline, "predict_proba"):
        return fitted_pipeline.predict_proba(X)[:, 1]
    if hasattr(fitted_pipeline, "decision_function"):
        return np.asarray(fitted_pipeline.decision_function(X))
    raise AttributeError("Le modèle fourni ne supporte ni 'predict_proba' ni 'decision_function'.")


def evaluate_model(fitted_pipeline: Pipeline, X_test, y_test, threshold: Optional[float] = None) -> Dict[str, float]:
    """Metrics on the test set. A custom decision ``threshold`` can be given.
 
    Clés retournées : accuracy, balanced_accuracy, precision, recall, f1, mcc
    (dépendent du seuil) et roc_auc, pr_auc (indépendantes du seuil).
    ``pr_auc`` = average precision. Le seuil est exprimé dans l'échelle de
    ``get_scores`` (probabilité, ou valeur de décision pour un SVM sans proba)."""
    y_true = np.asarray(y_test)
    y_pred = _predict_labels(fitted_pipeline, X_test, threshold)
    scores = get_scores(fitted_pipeline, X_test)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }

def compare_models(models: Dict[str, BaseEstimator], X_train, y_train, X_test, y_test, **pipeline_kwargs) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Fit every model in the same pipeline and return a metrics table.
 
    ``pipeline_kwargs`` est transmis à ``build_pipeline`` (imputer_strategy,
    selector, k, pca_components...). Renvoie ``(tableau, {nom: pipeline ajusté})``."""
    rows, fitted = {}, {}
    for name, model in models.items():
        pipe = build_pipeline(clone(model), **pipeline_kwargs)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows[name] = evaluate_model(pipe, X_test, y_test)
    return pd.DataFrame(rows).T, fitted
 
 
def cross_validate_models(models: Dict[str, BaseEstimator], X, y, cv: int = 5, scoring=("f1", "roc_auc", "average_precision"), **pipeline_kwargs) -> pd.DataFrame:
    """Mean +- std of the CV scores (on the TRAIN set) for each model.
 
    Colonnes : ``<score>_mean`` et ``<score>_std``. CV stratifiée (mélangée)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = list(scoring)
    rows = {}
    for name, model in models.items():
        pipe = build_pipeline(clone(model), **pipeline_kwargs)
        res = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
        row = {}
        for s in scoring:
            vals = res[f"test_{s}"]
            row[f"{s}_mean"], row[f"{s}_std"] = vals.mean(), vals.std()
        row["fit_time_s"] = res["fit_time"].mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def find_best_threshold(y_true, scores, metric: str = "f1") -> float:
    """Decision threshold maximising F1 (``metric='f1'``) on given scores.
 
    Use scores that are NOT computed on the test set (see ``oof_scores``).
    ``metric='balanced_accuracy'`` est aussi accepté."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes.")
    if metric == "f1":
        prec, rec, thr = precision_recall_curve(y_true, scores)
        prec, rec = prec[:-1], rec[:-1]  # un point de plus que de seuils
        denom = prec + rec
        f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
        return float(thr[np.argmax(f1)])
    if metric == "balanced_accuracy":
        fpr, tpr, thr = roc_curve(y_true, scores)
        return float(thr[np.argmax(tpr - fpr)])  # maximise (TPR + TNR) / 2
    raise ValueError("metric must be 'f1' or 'balanced_accuracy'.")
 
 
def oof_scores(pipeline: Pipeline, X, y, cv: int = 3) -> np.ndarray:
    """Out-of-fold scores on the training set (used to tune the threshold)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    if hasattr(pipeline, "predict_proba"):
        return cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    return np.asarray(cross_val_predict(pipeline, X, y, cv=skf, method="decision_function"))


def plot_roc_pr(fitted: Dict[str, Pipeline], X_test, y_test, figsize=(12, 4.5)):
    """ROC and precision-recall curves of several fitted models."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    for name, pipe in fitted.items():
        s = get_scores(pipe, X_test)
        RocCurveDisplay.from_predictions(y_test, s, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_test, s, name=name, ax=ax_pr)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="hasard")
    ax_roc.set_title("Courbes ROC")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.axhline(np.mean(y_test), color="k", ls="--", lw=1, label="hasard (prévalence)")
    ax_pr.set_title("Courbes précision-rappel")
    ax_pr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
 
 
def plot_confusion(fitted_pipeline: Pipeline, X_test, y_test, threshold: Optional[float] = None, title: str = "", ax=None):
    """Confusion matrix (counts) of a fitted model. Returns the ConfusionMatrixDisplay."""
    y_pred = _predict_labels(fitted_pipeline, X_test, threshold)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False, values_format="d")
    disp.ax_.set_title(title)
    return disp

def plot_pca_variance(X_train, figsize=(6, 4)):
    """Cumulative explained variance of a PCA fitted on the pre-processed train data.
 
    Le pré-traitement (imputation médiane + centrage-réduction) est appliqué ici,
    sur ``X_train`` uniquement. Renvoie la figure."""
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_train))
    cum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=3)
    for level in (0.90, 0.95):
        ax.axhline(level, color="gray", ls="--", lw=1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    ax.set(xlabel="Nombre de composantes", ylabel="Variance expliquée cumulée",
           title=f"ACP : {n95} composantes pour 95 % de variance")
    fig.tight_layout()
    return fig
 
 
def plot_feature_scores(scores: pd.Series, top: int = 15, title: str = "Feature scores"):
    """Horizontal bar plot of the ``top`` best feature scores."""
    best = scores.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(best) + 1.5))
    ax.barh(best.index.astype(str), best.values)
    ax.set_title(title)
    ax.set_xlabel("Score")
    fig.tight_layout()
    return fig
