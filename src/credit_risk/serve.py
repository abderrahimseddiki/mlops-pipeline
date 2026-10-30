"""Service de prédiction FastAPI, instrumenté pour Prometheus.

Endpoints : POST /predict · GET /health · GET /model · GET /metrics
Chaque requête est journalisée (data/predictions.jsonl) pour alimenter le monitoring de dérive.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from . import config
from .data import FEATURES


class CreditApplication(BaseModel):
    checking_status: str = Field(examples=["<0"])
    duration: float = Field(ge=0, examples=[24])
    credit_history: str = Field(examples=["existing paid"])
    purpose: str = Field(examples=["radio/tv"])
    credit_amount: float = Field(ge=0, examples=[3500])
    savings_status: str = Field(examples=["<100"])
    employment: str = Field(examples=["1<=X<4"])
    installment_commitment: float = Field(ge=0, examples=[3])
    personal_status: str = Field(examples=["male single"])
    other_parties: str = Field(examples=["none"])
    residence_since: float = Field(ge=0, examples=[2])
    property_magnitude: str = Field(examples=["car"])
    age: float = Field(ge=18, le=120, examples=[35])
    other_payment_plans: str = Field(examples=["none"])
    housing: str = Field(examples=["own"])
    existing_credits: float = Field(ge=0, examples=[1])
    job: str = Field(examples=["skilled"])
    num_dependents: float = Field(ge=0, examples=[1])
    own_telephone: str = Field(examples=["yes"])
    foreign_worker: str = Field(examples=["yes"])


class Prediction(BaseModel):
    probability_bad: float
    decision: str
    model_version: str


def create_app(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
    log_path: Path = config.PREDICTIONS_LOG,
    drift_report: Path = config.DRIFT_REPORT,
) -> FastAPI:
    registry = CollectorRegistry()
    requests_total = Counter("predictions_total", "Prédictions servies", ["decision"], registry=registry)
    errors_total = Counter("prediction_errors_total", "Erreurs de prédiction", registry=registry)
    latency = Histogram(
        "prediction_latency_seconds", "Latence de /predict", registry=registry,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    )
    score = Histogram(
        "prediction_score", "Distribution des probabilités prédites", registry=registry,
        buckets=tuple(i / 10 for i in range(1, 11)),
    )
    drift_share = Gauge("data_drift_share", "Part des variables en dérive", registry=registry)
    drift_flag = Gauge("data_drift_detected", "1 si dérive détectée", registry=registry)

    state: dict = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not Path(model_path).exists():
            raise RuntimeError(f"Modèle introuvable : {model_path}. Lancer d'abord l'entraînement.")
        state["model"] = joblib.load(model_path)
        meta = json.loads(Path(metadata_path).read_text("utf-8")) if Path(metadata_path).exists() else {}
        state["meta"] = meta
        state["threshold"] = meta.get("decision_threshold", 0.5)
        yield
        state.clear()

    app = FastAPI(title="Credit Risk API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": "model" in state}

    @app.get("/model")
    def model_info() -> dict:
        return state.get("meta", {})

    @app.post("/predict", response_model=Prediction)
    def predict(application: CreditApplication) -> Prediction:
        t0 = time.perf_counter()
        features = application.model_dump()
        try:
            X = pd.DataFrame([features], columns=FEATURES)
            proba = float(state["model"].predict_proba(X)[0, 1])
        except Exception as exc:
            errors_total.inc()
            raise HTTPException(status_code=500, detail="Erreur de prédiction") from exc

        decision = "reject" if proba >= state["threshold"] else "accept"
        version = state["meta"].get("model_version", "unknown")
        requests_total.labels(decision=decision).inc()
        score.observe(proba)
        latency.observe(time.perf_counter() - t0)

        record = {"ts": time.time(), "features": features, "probability_bad": proba, "model_version": version}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return Prediction(probability_bad=round(proba, 4), decision=decision, model_version=version)

    @app.get("/metrics")
    def metrics() -> Response:
        if Path(drift_report).exists():
            rep = json.loads(Path(drift_report).read_text("utf-8"))
            drift_share.set(rep.get("drift_share", 0))
            drift_flag.set(1 if rep.get("dataset_drift") else 0)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


# Le modèle n'est chargé qu'au démarrage du serveur (lifespan), pas à l'import.
app = create_app()
