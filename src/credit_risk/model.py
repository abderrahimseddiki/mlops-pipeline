"""Construction du pipeline scikit-learn (prétraitement + modèle) en un seul objet.

Un seul objet sérialisé = le même prétraitement à l'entraînement et en production,
ce qui supprime toute une classe de bugs (« training/serving skew »).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

MODELS = {
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def build_pipeline(model_type: str, params: dict, random_state: int = 42) -> Pipeline:
    if model_type not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_type}. Choix : {sorted(MODELS)}")
    cls = MODELS[model_type]
    kwargs = dict(params or {})
    if model_type == "logistic_regression":
        kwargs.setdefault("max_iter", 1000)
    if "random_state" in cls().get_params():
        kwargs.setdefault("random_state", random_state)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", cls(**kwargs))])
