"""Détection de dérive des données (data drift) entre la référence d'entraînement et la production.

Implémentation maison du PSI (Population Stability Index), métrique standard en scoring crédit :
    PSI = Σ (p_cur - p_ref) * ln(p_cur / p_ref)
    < 0.1  : stable · 0.1–0.2 : dérive modérée · > 0.2 : dérive significative

Pourquoi pas uniquement Evidently ? La logique d'alerte reste testable, sans dépendance lourde
et indépendante des changements d'API d'une librairie. Evidently sert en plus pour un rapport HTML visuel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _psi(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, None)
    p_cur = np.clip(p_cur, EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variable quasi constante : on retombe sur des catégories
        return psi_categorical(ref.astype(str), cur.astype(str))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, edges)[0] / len(ref)
    p_cur = np.histogram(cur, edges)[0] / len(cur)
    return _psi(p_ref, p_cur)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    p_ref = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    p_cur = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(p_ref, p_cur)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    psi_threshold: float = 0.2,
    drift_share_threshold: float = 0.3,
    critical_features: list[str] | None = None,
) -> dict:
    """Alerte si trop de variables dérivent OU si une variable critique pour le modèle dérive."""
    features = {}
    for col in numeric:
        features[col] = psi_numeric(reference[col].astype(float), current[col].astype(float))
    for col in categorical:
        features[col] = psi_categorical(reference[col].astype(str), current[col].astype(str))

    drifted = sorted(c for c, v in features.items() if v > psi_threshold)
    share = len(drifted) / len(features) if features else 0.0
    critical_drifted = sorted(set(drifted) & set(critical_features or []))
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "psi": {k: round(v, 4) for k, v in sorted(features.items(), key=lambda kv: -kv[1])},
        "drifted_features": drifted,
        "drift_share": round(share, 4),
        "critical_drifted": critical_drifted,
        "dataset_drift": share > drift_share_threshold or bool(critical_drifted),
        "thresholds": {"psi": psi_threshold, "drift_share": drift_share_threshold},
    }
