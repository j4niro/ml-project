"""Tests unitaires de la partie C (évaluation, déséquilibre, graphiques).

Les tests n'utilisent pas ``build_pipeline`` (partie A) : on le remplace par un
pipeline simple grâce à ``monkeypatch``. Ils vérifient des propriétés (formes,
bornes, cas limites), pas des valeurs de performance."""
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre pendant les tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ml_workflow as ml


def simple_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


@pytest.fixture
def fake_build(monkeypatch):
    """Remplace build_pipeline (partie A) par un pipeline minimal."""
    monkeypatch.setattr(ml, "build_pipeline", lambda model, **kw: simple_pipeline(model))


@pytest.fixture
def split(data):
    X, y = data
    return ml.train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.fixture
def fitted_logreg(split):
    X_tr, X_te, y_tr, y_te = split
    return simple_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- get_scores -------------------------------------------------------------
def test_get_scores_probabilities(fitted_logreg, split):
    _, X_te, _, _ = split
    s = ml.get_scores(fitted_logreg, X_te)
    assert s.shape == (len(X_te),)
    assert ((s >= 0) & (s <= 1)).all()


def test_get_scores_decision_function(split):
    X_tr, X_te, y_tr, _ = split
    svm = simple_pipeline(SVC()).fit(X_tr, y_tr)  # pas de predict_proba
    s = ml.get_scores(svm, X_te)
    assert s.shape == (len(X_te),)
    assert s.min() < 0  # valeurs de décision, pas des probabilités


# --- evaluate_model ---------------------------------------------------------
def test_evaluate_model_keys_and_bounds(fitted_logreg, split):
    _, X_te, _, y_te = split
    m = ml.evaluate_model(fitted_logreg, X_te, y_te)
    assert set(m) == {"accuracy", "balanced_accuracy", "precision", "recall",
                      "f1", "mcc", "roc_auc", "pr_auc"}
    for k, v in m.items():
        assert -1 <= v <= 1, k
    assert m["roc_auc"] > 0.5


def test_evaluate_model_extreme_thresholds(fitted_logreg, split):
    _, X_te, _, y_te = split
    all_pos = ml.evaluate_model(fitted_logreg, X_te, y_te, threshold=0.0)
    all_neg = ml.evaluate_model(fitted_logreg, X_te, y_te, threshold=1.1)
    assert all_pos["recall"] == 1.0
    assert all_neg["recall"] == 0.0 and all_neg["precision"] == 0.0  # zero_division géré
    assert all_neg["balanced_accuracy"] == pytest.approx(0.5)
    # les métriques indépendantes du seuil ne bougent pas
    assert all_pos["roc_auc"] == pytest.approx(all_neg["roc_auc"])


# --- seuil ------------------------------------------------------------------
def test_find_best_threshold_perfect_scores():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    scores = y * 0.8 + 0.1  # séparation parfaite
    t = ml.find_best_threshold(y, scores)
    assert ((scores >= t).astype(int) == y).all()


def test_find_best_threshold_in_score_range(data):
    X, y = data
    s = ml.oof_scores(simple_pipeline(LogisticRegression(max_iter=1000)), X, y)
    for metric in ("f1", "balanced_accuracy"):
        t = ml.find_best_threshold(y, s, metric=metric)
        assert s.min() <= t <= s.max()


def test_find_best_threshold_errors():
    with pytest.raises(ValueError):
        ml.find_best_threshold([0, 0, 0], [0.1, 0.2, 0.3])  # une seule classe
    with pytest.raises(ValueError):
        ml.find_best_threshold([0, 1], [0.1, 0.9], metric="inconnue")


def test_oof_scores(data):
    X, y = data
    s = ml.oof_scores(simple_pipeline(LogisticRegression(max_iter=1000)), X, y, cv=3)
    assert s.shape == (len(y),)
    assert not np.isnan(s).any()
    assert ml.roc_auc_score(y, s) > 0.5


# --- comparaison de modèles -------------------------------------------------
def test_compare_models(fake_build, split):
    X_tr, X_te, y_tr, y_te = split
    models = {"logreg": LogisticRegression(max_iter=1000), "svm": SVC()}
    table, fitted = ml.compare_models(models, X_tr, y_tr, X_te, y_te)
    assert list(table.index) == ["logreg", "svm"]
    assert set(fitted) == {"logreg", "svm"}
    assert "pr_auc" in table.columns
    assert table.notna().all().all()


def test_cross_validate_models(fake_build, data):
    X, y = data
    out = ml.cross_validate_models({"logreg": LogisticRegression(max_iter=1000)}, X, y, cv=3)
    assert list(out.index) == ["logreg"]
    for s in ("f1", "roc_auc", "average_precision"):
        assert f"{s}_mean" in out.columns and f"{s}_std" in out.columns
    assert 0 <= out.loc["logreg", "roc_auc_mean"] <= 1


# --- graphiques -------------------------------------------------------------
def test_plot_roc_pr(fitted_logreg, split):
    _, X_te, _, y_te = split
    fig = ml.plot_roc_pr({"logreg": fitted_logreg}, X_te, y_te)
    assert len(fig.axes) == 2


def test_plot_confusion_counts(fitted_logreg, split):
    _, X_te, _, y_te = split
    disp = ml.plot_confusion(fitted_logreg, X_te, y_te, threshold=0.3, title="test")
    assert disp.confusion_matrix.sum() == len(y_te)
    assert disp.ax_.get_title() == "test"


def test_plot_pca_variance_handles_nan(data):
    X, _ = data
    X = X.copy()
    X.iloc[::17, 3] = np.nan
    fig = ml.plot_pca_variance(X)
    y_cum = fig.axes[0].lines[0].get_ydata()
    assert y_cum[-1] == pytest.approx(1.0)
    assert (np.diff(y_cum) >= 0).all()


def test_plot_feature_scores_top():
    scores = pd.Series(np.arange(20, dtype=float), index=[f"f{i}" for i in range(20)])
    fig = ml.plot_feature_scores(scores, top=5)
    assert len(fig.axes[0].patches) == 5