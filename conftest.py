"""Fixtures partagées par tous les tests."""
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def data():
    """Petit jeu synthétique déséquilibré (400 lignes, 12 variables)."""
    X, y = make_classification(n_samples=400, n_features=12, n_informative=4,
                               weights=[0.8, 0.2], random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), pd.Series(y)
