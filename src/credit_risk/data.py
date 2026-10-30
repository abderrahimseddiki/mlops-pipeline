"""Récupération, validation et découpage des données."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import POSITIVE_LABEL, RAW_DATA, TARGET

log = logging.getLogger(__name__)

NUMERIC = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
]
CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status", "employment",
    "personal_status", "other_parties", "property_magnitude", "other_payment_plans",
    "housing", "job", "own_telephone", "foreign_worker",
]
FEATURES = NUMERIC + CATEGORICAL


def download(name: str, version: int) -> pd.DataFrame:
    """Télécharge le jeu OpenML une fois, puis le garde en cache dans data/credit.csv."""
    if RAW_DATA.exists():
        return pd.read_csv(RAW_DATA)
    from sklearn.datasets import fetch_openml

    log.info("Téléchargement OpenML %s v%s", name, version)
    df = fetch_openml(name=name, version=version, as_frame=True).frame
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA, index=False)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Contrôles de schéma : on échoue tôt plutôt que d'entraîner sur des données cassées."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    if df[NUMERIC].isna().any().any():
        raise ValueError("Valeurs numériques manquantes")
    if not set(df[TARGET].unique()) <= {"good", "bad"}:
        raise ValueError("La cible doit valoir 'good' ou 'bad'")
    return df


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], (df[TARGET] == POSITIVE_LABEL).astype(int)


def split(df: pd.DataFrame, test_size: float, random_state: int):
    X, y = to_xy(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
