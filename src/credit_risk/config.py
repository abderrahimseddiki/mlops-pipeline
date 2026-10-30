"""Chargement de params.yaml et chemins du projet."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_DATA = DATA_DIR / "credit.csv"
REFERENCE_DATA = DATA_DIR / "reference.csv"
PREDICTIONS_LOG = DATA_DIR / "predictions.jsonl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
METADATA_PATH = MODELS_DIR / "metadata.json"
DRIFT_REPORT = REPORTS_DIR / "drift_latest.json"

TARGET = "class"
POSITIVE_LABEL = "bad"  # on prédit la probabilité qu'un crédit tourne mal


def load_params(path: str | Path | None = None) -> dict:
    path = Path(path or ROOT / "params.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
