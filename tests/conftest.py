"""Jeu de données synthétique au format German Credit : tests rapides, sans réseau."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import CATEGORICAL

CATEGORIES = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": ["existing paid", "critical/other existing credit", "delayed previously"],
    "purpose": ["radio/tv", "new car", "used car", "furniture/equipment", "business"],
    "savings_status": ["<100", "100<=X<500", "no known savings"],
    "employment": ["<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": ["male single", "female div/dep/mar", "male mar/wid"],
    "other_parties": ["none", "guarantor"],
    "property_magnitude": ["real estate", "car", "life insurance", "no known property"],
    "other_payment_plans": ["none", "bank"],
    "housing": ["own", "rent", "for free"],
    "job": ["skilled", "unskilled resident", "high qualif/self emp/mgmt"],
    "own_telephone": ["yes", "none"],
    "foreign_worker": ["yes", "no"],
}


def make_credit_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "duration": rng.integers(6, 60, n),
            "credit_amount": rng.lognormal(8, 0.6, n).round(),
            "installment_commitment": rng.integers(1, 5, n),
            "residence_since": rng.integers(1, 5, n),
            "age": rng.integers(19, 75, n),
            "existing_credits": rng.integers(1, 4, n),
            "num_dependents": rng.integers(1, 3, n),
        }
    )
    for col in CATEGORICAL:
        df[col] = rng.choice(CATEGORIES[col], n)
    # signal appris : crédits longs et compte négatif -> plus de risque
    logit = 0.05 * (df["duration"] - 30) + 1.2 * (df["checking_status"] == "<0") - 0.8
    df["class"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "bad", "good")
    return df


@pytest.fixture
def credit_df():
    return make_credit_df()


@pytest.fixture
def small_params():
    return {
        "data": {"openml_name": "credit-g", "openml_version": 1, "test_size": 0.25, "random_state": 0},
        "model": {
            "type": "gradient_boosting",
            "params": {"n_estimators": 50, "max_depth": 2},
            "decision_threshold": 0.5,
            "cv_folds": 3,
        },
        "quality_gate": {"min_roc_auc": 0.6},
    }
