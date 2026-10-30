import json

import pytest
from fastapi.testclient import TestClient

from credit_risk.data import FEATURES
from credit_risk.serve import create_app
from credit_risk.train import train


@pytest.fixture
def client(credit_df, small_params, tmp_path):
    train(small_params, df=credit_df, models_dir=tmp_path / "models",
          reference_path=tmp_path / "ref.csv", use_mlflow=False)
    app = create_app(
        model_path=tmp_path / "models" / "model.joblib",
        metadata_path=tmp_path / "models" / "metadata.json",
        log_path=tmp_path / "predictions.jsonl",
        drift_report=tmp_path / "drift.json",
    )
    with TestClient(app) as c:
        c.tmp = tmp_path
        yield c


@pytest.fixture
def application(credit_df):
    row = credit_df[FEATURES].iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_and_log(client, application):
    r = client.post("/predict", json=application)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_bad"] <= 1
    assert body["decision"] in {"accept", "reject"}
    logged = [json.loads(x) for x in (client.tmp / "predictions.jsonl").read_text().splitlines()]
    assert len(logged) == 1 and logged[0]["features"]["age"] == application["age"]


def test_invalid_input_is_rejected(client, application):
    application["age"] = 5
    assert client.post("/predict", json=application).status_code == 422
    del application["purpose"]
    assert client.post("/predict", json=application).status_code == 422


def test_metrics_exposes_counters_and_drift(client, application):
    client.post("/predict", json=application)
    (client.tmp / "drift.json").write_text(json.dumps({"drift_share": 0.4, "dataset_drift": True}))
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert "prediction_latency_seconds_bucket" in text
    assert "data_drift_share 0.4" in text
    assert "data_drift_detected 1.0" in text


def test_missing_model_fails_fast(tmp_path):
    app = create_app(model_path=tmp_path / "nope.joblib")
    with pytest.raises(RuntimeError, match="Modèle introuvable"), TestClient(app):
        pass
