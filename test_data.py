import numpy as np
import pandas as pd
import pytest

import ml_workflow as ml

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class FitCounterClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, value=0):
        self.value = value
        self.fit_rows_seen = None

    def fit(self, X, y):
        self.fit_rows_seen = len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_load_spambase():

    df = ml.load_spambase()

    assert df.shape == (4601, 58)
    assert "is_spam" in df.columns
    assert df["is_spam"].isin([0, 1]).all()


def test_load_diabetes():

    df = ml.load_diabetes()

    assert df.shape == (253680, 22)
    assert "Diabetes_binary" in df.columns
    assert df["Diabetes_binary"].isin([0, 1]).all()


def test_split_features_target():

    df = ml.load_spambase()

    X, y = ml.split_features_target(df, "is_spam")

    assert X.shape == (4601, 57)
    assert len(y) == 4601
    assert "is_spam" not in X.columns


def test_split_features_target_invalid_column():

    df = ml.load_spambase()

    with pytest.raises(ValueError):
        ml.split_features_target(df, "unknown_target")


def test_describe_dataset():

    df = ml.load_spambase()

    summary = ml.describe_dataset(df, "is_spam")

    assert summary["n_rows"] == 4601
    assert summary["n_columns"] == 58
    assert summary["n_features"] == 57
    assert summary["n_missing_values"] == 0
    assert summary["n_duplicated_rows"] == 391


def test_inject_missing_values():

    X = pd.DataFrame({
        "A": range(100),
        "B": range(100),
        "C": range(100)
    })

    X_missing = ml.inject_missing_values(X, frac=0.05, seed=42)

    assert X.isna().sum().sum() == 0
    assert X_missing.isna().sum().sum() == 15


def test_inject_missing_values_reproducibility():

    X = pd.DataFrame(np.arange(100).reshape(20, 5))

    X1 = ml.inject_missing_values(X, frac=0.1, seed=42)
    X2 = ml.inject_missing_values(X, frac=0.1, seed=42)

    pd.testing.assert_frame_equal(X1, X2)


def test_inject_missing_values_invalid_fraction():

    X = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        ml.inject_missing_values(X, frac=1.5)


def test_split_data():

    X = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 4)))
    y = pd.Series([0] * 80 + [1] * 20)

    X_train, X_test, y_train, y_test = ml.split_data(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20

    assert abs(y_train.mean() - y.mean()) < 0.02
    assert abs(y_test.mean() - y.mean()) < 0.02


def test_subsample():

    X = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 4)))
    y = pd.Series([0] * 80 + [1] * 20)

    X_sub, y_sub = ml.subsample(X, y, n=30)

    assert len(X_sub) == 30
    assert len(y_sub) == 30

    assert abs(y_sub.mean() - y.mean()) < 0.05


def test_subsample_invalid_size():

    X = pd.DataFrame({"A": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError):
        ml.subsample(X, y, n=10)


def test_build_pipeline():

    X = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "B": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    })

    y = pd.Series([0, 0, 1, 1, 0, 1])

    pipeline = ml.build_pipeline(
        model=LogisticRegression(),
        imputer_strategy="median",
        scale=True
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)

    assert list(pipeline.named_steps.keys()) == [
        "imputer",
        "scaler",
        "selector",
        "pca",
        "model"
    ]
    
@pytest.mark.parametrize("method", ["anova", "mutual_info", "l1", "rf"])
def test_selector_keeps_k_features(data, method):
    X, y = data
    pipe = ml.build_pipeline(LogisticRegression(), selector=method, k=5).fit(X, y)
    assert len(ml.selected_features(pipe, X.columns)) == 5


def test_make_selector_unknown_method_raises():
    with pytest.raises(ValueError):
        ml.make_selector("unknown")


def test_selected_features_without_selector_returns_all(data):
    X, y = data
    pipe = ml.build_pipeline(LogisticRegression()).fit(X, y)
    assert ml.selected_features(pipe, X.columns) == list(X.columns)


def test_get_models_keys_and_svm_option():
    without_svm = ml.get_models(include_svm=False)
    with_svm = ml.get_models(include_svm=True)
    assert {"lr", "rf", "hgb"} <= set(without_svm)
    assert "SVM-RBF" not in without_svm
    assert "SVM-RBF" in with_svm
    assert all(hasattr(m, "fit") for m in with_svm.values())


def test_get_models_class_weight_is_applied():
    models = ml.get_models(class_weight="balanced")
    assert all(m.class_weight == "balanced" for m in models.values())


@pytest.mark.parametrize("method", ["anova", "mutual_info", "l1", "rf"])
def test_feature_scores_sorted_and_complete(data, method):
    X, y = data
    scores = ml.feature_scores(X, y, method)
    assert set(scores.index) == set(X.columns)
    assert list(scores.values) == sorted(scores.values, reverse=True)


def test_feature_scores_handles_missing_values(data):
    X, y = data
    X_nan = ml.inject_missing_values(X, frac=0.1)
    scores = ml.feature_scores(X_nan, y, "anova")
    assert not scores.isna().any()


def test_tune_model_returns_best_params(data):
    X, y = data
    pipe = ml.build_pipeline(LogisticRegression(max_iter=1000))
    gs = ml.tune_model(pipe, {"model__C": [0.1, 1.0]}, X, y, n_samples=10_000)
    assert gs.best_params_["model__C"] in (0.1, 1.0)


def test_tune_model_refits_best_estimator_on_full_data(data):
    X, y = data
    pipe = ml.build_pipeline(FitCounterClassifier())
    gs = ml.tune_model(
        pipe,
        {"model__value": [0, 1]},
        X,
        y,
        scoring="accuracy",
        cv=2,
        n_samples=100,
    )

    assert gs.best_estimator_.named_steps["model"].fit_rows_seen == len(X)