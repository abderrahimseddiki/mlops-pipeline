"""Entraînement : données -> validation croisée -> test -> artefacts -> MLflow -> quality gate.

Usage : python -m credit_risk.train
Code de sortie 1 si le modèle ne passe pas le quality gate (utilisé par la CI pour bloquer le déploiement).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config
from .data import download, split, validate
from .model import build_pipeline

log = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate(pipeline, X, y, threshold: float) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
    }


def train(
    params: dict,
    df: pd.DataFrame | None = None,
    models_dir: Path = config.MODELS_DIR,
    reference_path: Path = config.REFERENCE_DATA,
    use_mlflow: bool = True,
) -> dict:
    d, m = params["data"], params["model"]
    if df is None:
        df = download(d["openml_name"], d["openml_version"])
    df = validate(df)
    X_train, X_test, y_train, y_test = split(df, d["test_size"], d["random_state"])

    pipeline = build_pipeline(m["type"], m.get("params", {}), d["random_state"])
    cv = StratifiedKFold(n_splits=m.get("cv_folds", 5), shuffle=True, random_state=d["random_state"])
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    metrics = evaluate(pipeline, X_test, y_test, m["decision_threshold"])
    metrics.update(cv_roc_auc_mean=round(cv_auc.mean(), 4), cv_roc_auc_std=round(cv_auc.std(), 4))

    version = time.strftime("%Y%m%d-%H%M%S")
    metadata = {
        "model_version": version,
        "git_sha": _git_sha(),
        "model_type": m["type"],
        "params": m.get("params", {}),
        "decision_threshold": m["decision_threshold"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_seconds": round(train_seconds, 2),
        "metrics": metrics,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "model.joblib")
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Données de référence pour la détection de dérive en production
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(reference_path, index=False)

    if use_mlflow:
        _log_mlflow(pipeline, X_train, metadata)
    return metadata


def _log_mlflow(pipeline, X_train: pd.DataFrame, metadata: dict) -> None:
    import mlflow
    from mlflow.models import infer_signature

    # Par défaut : base SQLite locale (le backend fichier ./mlruns est déprécié depuis MLflow 3).
    # En docker compose, MLFLOW_TRACKING_URI pointe vers le serveur MLflow.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run(run_name=f"{metadata['model_type']}-{metadata['model_version']}"):
        mlflow.log_params({"model_type": metadata["model_type"], **metadata["params"]})
        mlflow.log_param("decision_threshold", metadata["decision_threshold"])
        mlflow.log_metrics(metadata["metrics"])
        mlflow.set_tag("git_sha", metadata["git_sha"])
        sample = X_train.head(5)
        kwargs = {
            "name": "model",
            "signature": infer_signature(sample, pipeline.predict_proba(sample)[:, 1]),
            "input_example": sample,
        }
        # MLflow récent sérialise via skops et exige de déclarer les types autorisés.
        # Le modèle est produit ici même : on autorise uniquement la structure d'arbre de scikit-learn.
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        mlflow.sklearn.log_model(pipeline, **kwargs)


def quality_gate(metadata: dict, min_roc_auc: float) -> bool:
    return metadata["metrics"]["roc_auc"] >= min_roc_auc


CANDIDATE_DIR = config.MODELS_DIR / "candidate"


def promote(candidate_dir: Path, models_dir: Path, reference_path: Path) -> None:
    """Le candidat remplace le modèle de production uniquement après validation."""
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_dir / "model.joblib", models_dir / "model.joblib")
    shutil.copy2(candidate_dir / "metadata.json", models_dir / "metadata.json")
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_dir / "reference.csv", reference_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = config.load_params()
    use_mlflow = os.getenv("DISABLE_MLFLOW", "0") != "1"
    metadata = train(
        params, models_dir=CANDIDATE_DIR, reference_path=CANDIDATE_DIR / "reference.csv", use_mlflow=use_mlflow
    )
    print(json.dumps(metadata["metrics"], indent=2))

    threshold = params["quality_gate"]["min_roc_auc"]
    if not quality_gate(metadata, threshold):
        print(f"QUALITY GATE FAILED: roc_auc {metadata['metrics']['roc_auc']} < {threshold}")
        print("Le modèle de production n'a pas été modifié.")
        sys.exit(1)
    promote(CANDIDATE_DIR, config.MODELS_DIR, config.REFERENCE_DATA)
    print(f"Quality gate OK (roc_auc >= {threshold}) — modèle {metadata['model_version']} promu en production")


if __name__ == "__main__":
    main()
