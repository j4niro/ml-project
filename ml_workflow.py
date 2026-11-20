"""Workflow ML pour la classification binaire (Spambase & Diabetes).

Workflow : import -> pré-traitement (imputation, centrage-réduction, ACP) -> split
-> entraînement avec sélection de variables -> validation.

Principe anti-fuite : tout le pré-traitement est dans un sklearn.Pipeline ajusté
sur le TRAIN uniquement."""

from __future__ import annotations

from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import  HistGradientBoostingClassifier, RandomForestClassifier
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

    df = pd.read_csv(path, header=None, names=SPAMBASE_COLUMNS)
    return df

def load_diabetes(path: str='data/diabetes_binary_health_indicators_BRFSS2015.csv') -> pd.DataFrame:
    """Load the Diabetes Health Indicators dataset (21 features + ``Diabetes_binary``)."""

    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from a dataframe and the name of its target column."""

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    x = df.drop(columns=[target])
    y = df[target]
    return x, y 


def describe_dataset(df: pd.DataFrame, target: str) -> pd.Series:
    """Quick summary: size, missing values, duplicates, class balance."""

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    class_counts = df[target].value_counts(dropna=False).to_dict()

    class_proportions = (df[target].value_counts(normalize=True, dropna=False).to_dict())

    summary = pd.Series({
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_features": len(df.columns) - 1,
        "n_missing_values": int(df.isna().sum().sum()),
        "n_duplicated_rows": int(df.duplicated().sum()),
        "target": target,
        "class_counts": class_counts,
        "class_proportions": class_proportions
    })
    return summary

def inject_missing_values(X: pd.DataFrame, frac: float=0.05, seed: int=RANDOM_STATE) -> pd.DataFrame:
    """Randomly (MCAR) replace a fraction of the cells by NaN.

        Neither dataset contains missing values; this function lets us test and
        compare imputation strategies in a controlled way."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame.")

    if not 0 <= frac < 1:
        raise ValueError("Fraction must be between 0 and 1(prohibited).")
    
    X_missing = X.copy()

    n_cells = X_missing.size

    n_missing = int(round(frac * n_cells))

    rng = np.random.default_rng(seed)

    # Création d'un masque de départ : toutes les cellules sont False
    mask = np.zeros(X.shape, dtype=bool)

    # Mélange des positions possibles et sélection des n_missing premières
    selected_positions = rng.permutation(n_cells)[:n_missing]

    # On marque uniquement ces positions comme manquantes
    mask.ravel()[selected_positions] = True

    # Application du masque : True devient NaN
    X_missing = X_missing.mask(mask)

    return X_missing
    

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

    if imputer_strategy not in ['mean', 'median']:
        raise ValueError("imputer_strategy must be 'mean' or 'median'.")

    if selector not in (None, "anova", "mutual_info", "l1", "rf"):
        raise ValueError("selector must be None, 'anova', 'mutual_info', 'l1', or 'rf'.")

    if k < 1:
        raise ValueError("k must be a greater or equal to 1.")

    steps = []

    steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))

    if scale: 
        steps.append(('scaler', StandardScaler()))

    else:
        steps.append(('scaler', 'passthrough'))

    if selector is not None:
        steps.append(('selector', make_selector(selector, k=k, random_state=random_state)))

    else:
        steps.append(('selector', 'passthrough'))

    if pca_components is not None:

        if isinstance(pca_components, float):
            if not (0 < pca_components <= 1):
                raise ValueError("If pca_components is a float, it must be in ]0, 1).")

        elif isinstance(pca_components, int):
            if pca_components < 1:
                raise ValueError("If pca_components is an int, it must be greater or equal to 1.")

        else:
            raise TypeError("pca_components must be None, a float in (0, 1), or an int >= 1.")

        steps.append(('pca', PCA(n_components=pca_components, random_state=random_state)))

    else:
        steps.append(('pca', 'passthrough'))

    steps.append(('model', model))

    return Pipeline(steps)



def split_data(X, y, test_size: float=0.2, seed: int=RANDOM_STATE):
    """Stratified train/test split (keeps the class proportions in both parts)."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    return X_train, X_test, y_train, y_test

def subsample(X, y, n: int, seed: int=RANDOM_STATE):
    """Stratified subsample of ``n`` rows (to speed up tuning on large data)."""

    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError("n must be an integer.")

    if n < 1:
        raise ValueError("n must be positive.")

    if n > len(X):
        raise ValueError("n can't be greater than the number of rows in X.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if n == len(X):
        return X.copy(), y.copy()

    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)

    return X_sub, y_sub
    


# ============================================================================
# B — Modèles & sélection de variables
# ============================================================================

def make_selector(method: str, k: int=10, random_state: int=RANDOM_STATE):
    
    """Feature selector keeping the ``k`` best features."""
    if method == 'anova':
        return SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        return SelectKBest(partial(mutual_info_classif, random_state=random_state), k=k)
    elif method == 'l1':
        return SelectFromModel(
            LogisticRegression(
                l1_ratio=1.0,
                solver='liblinear',
                random_state=random_state,
                C=1.0,
                max_iter=500,
            ),
            max_features=k,
            threshold=-np.inf,
        )
    elif method == 'rf':
        return SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=random_state), max_features=k, threshold=-np.inf)
    else:
        raise ValueError("method must be one of: 'anova', 'mutual_info', 'l1', 'rf'.")


def get_models(class_weight: Optional[str]=None, include_svm: bool=True, random_state: int=RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """Candidate classifiers. ``class_weight='balanced'`` handles imbalance."""
    models = {
        'lr': LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=500),
        'rf': RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=random_state),
        'hgb': HistGradientBoostingClassifier(class_weight=class_weight, random_state=random_state),
    }
    if include_svm:
        models['SVM-RBF'] = SVC(class_weight=class_weight, probability=False, random_state=random_state)
    return models


def tune_model(pipeline: Pipeline, param_grid: dict, X, y, scoring: str='average_precision', cv: int=3, n_samples: Optional[int]=None) -> GridSearchCV:
    """Grid search with cross-validation on the training data only.

    ``n_samples`` limits the number of rows used for the search (large data);
    The best hyperparameters are then refit on the complete X and y."""

    X_full, y_full = X, y
    if n_samples is not None:
        X_search, y_search = subsample(X_full, y_full, min(n_samples, len(X_full)))
    else:
        X_search, y_search = X_full, y_full

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=(n_samples is None),
    )
    grid_search.fit(X_search, y_search)

    if n_samples is not None:
        best_pipeline = clone(pipeline).set_params(**grid_search.best_params_)
        best_pipeline.fit(X_full, y_full)
        grid_search._best_estimator_ = best_pipeline
        grid_search.best_estimator_ = best_pipeline

    return grid_search


def feature_scores(X: pd.DataFrame, y, method: str='anova') -> pd.Series:
    """Univariate score of each feature (higher = more related to the target)."""
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    if method == 'anova':
        scores = f_classif(X_imp, y)[0]
    elif method == 'mutual_info':
        scores = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE)
    elif method == 'l1':
        Z = StandardScaler().fit_transform(X_imp)
        model = LogisticRegression(
            l1_ratio=1.0,
            solver='liblinear',
            random_state=RANDOM_STATE,
            C=1.0,
            max_iter=500,
        ).fit(Z, y)
        scores = np.abs(model.coef_[0])
    elif method == 'rf':
        scores = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_imp, y).feature_importances_
    else:
        raise ValueError("method must be one of: 'anova', 'mutual_info', 'l1', 'rf'.")

    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def selected_features(fitted_pipeline: Pipeline, feature_names: Iterable[str]) -> List[str]:
    """Names of the features kept by the 'selector' step of a fitted pipeline."""

    selector = fitted_pipeline.named_steps.get('selector')
    if selector is None or selector == 'passthrough':
        return list(feature_names)

    if hasattr(selector, 'get_support'):
        mask = selector.get_support()
        return [name for name, keep in zip(feature_names, mask) if keep]

    raise ValueError("The pipeline does not have a valid 'selector' step.")



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
        res = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        row = {}
        for s in scoring:
            vals = res[f"test_{s}"]
            row[f"{s}_mean"], row[f"{s}_std"] = vals.mean(), vals.std()
        row["fit_time_s"] = res["fit_time"].mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def compare_feature_selectors(
    model: BaseEstimator,
    X,
    y,
    methods: Iterable[str] = ("anova", "mutual_info", "l1", "rf"),
    k_values: Iterable[int] = (5, 10, 15),
    cv: int = 5,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Compare feature-selection methods and k values with stratified CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rows = {}

    for method in methods:
        for k in k_values:
            if k < 1:
                raise ValueError("k_values must contain positive integers.")
            pipe = build_pipeline(clone(model), selector=method, k=k)
            res = cross_validate(
                pipe,
                X,
                y,
                cv=skf,
                scoring=(scoring,),
                n_jobs=-1,
            )
            values = res[f"test_{scoring}"]
            rows[(method, k)] = {
                f"{scoring}_mean": values.mean(),
                f"{scoring}_std": values.std(),
                "fit_time_s": res["fit_time"].mean(),
            }

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["method", "k"])
    return table


def compare_pca(
    model: BaseEstimator,
    X,
    y,
    pca_components: Iterable[Optional[float | int]] = (None, 0.95),
    cv: int = 5,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Compare pipelines with and without PCA using stratified CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rows = {}

    for components in pca_components:
        pipe = build_pipeline(clone(model), pca_components=components)
        res = cross_validate(
            pipe,
            X,
            y,
            cv=skf,
            scoring=(scoring,),
            n_jobs=-1,
        )
        values = res[f"test_{scoring}"]
        label = "none" if components is None else str(components)
        rows[label] = {
            f"{scoring}_mean": values.mean(),
            f"{scoring}_std": values.std(),
            "fit_time_s": res["fit_time"].mean(),
        }

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
