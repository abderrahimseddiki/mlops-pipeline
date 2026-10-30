import json

import joblib
import pytest

from credit_risk.data import FEATURES, validate
from credit_risk.model import build_pipeline
from credit_risk.train import quality_gate, train


def test_validate_rejects_missing_columns(credit_df):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate(credit_df.drop(columns=["age"]))


def test_validate_rejects_bad_target(credit_df):
    credit_df.loc[0, "class"] = "maybe"
    with pytest.raises(ValueError):
        validate(credit_df)


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest", "logistic_regression"])
def test_pipeline_outputs_probabilities(credit_df, model_type):
    df = validate(credit_df)
    pipe = build_pipeline(model_type, {}).fit(df[FEATURES], df["class"] == "bad")
    proba = pipe.predict_proba(df[FEATURES].head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_category_does_not_crash(credit_df):
    df = validate(credit_df)
    pipe = build_pipeline("logistic_regression", {}).fit(df[FEATURES], df["class"] == "bad")
    row = df[FEATURES].head(1).copy()
    row["purpose"] = "catégorie jamais vue"
    assert pipe.predict_proba(row).shape == (1, 2)


def test_train_writes_artifacts_and_learns_signal(credit_df, small_params, tmp_path):
    meta = train(
        small_params, df=credit_df, models_dir=tmp_path / "models",
        reference_path=tmp_path / "reference.csv", use_mlflow=False,
    )
    assert (tmp_path / "models" / "model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert saved["metrics"] == meta["metrics"]
    assert meta["metrics"]["roc_auc"] > 0.6  # le signal synthétique doit être appris
    model = joblib.load(tmp_path / "models" / "model.joblib")
    assert model.predict_proba(credit_df[FEATURES].head(3)).shape == (3, 2)


def test_quality_gate():
    assert quality_gate({"metrics": {"roc_auc": 0.75}}, 0.7)
    assert not quality_gate({"metrics": {"roc_auc": 0.65}}, 0.7)


def test_promote_copies_candidate_to_production(credit_df, small_params, tmp_path):
    from credit_risk.train import promote

    cand = tmp_path / "candidate"
    train(small_params, df=credit_df, models_dir=cand, reference_path=cand / "reference.csv", use_mlflow=False)
    prod, ref = tmp_path / "models", tmp_path / "data" / "reference.csv"
    promote(cand, prod, ref)
    assert (prod / "model.joblib").exists() and (prod / "metadata.json").exists() and ref.exists()


def test_failed_gate_keeps_production_model(monkeypatch, credit_df, small_params, tmp_path):
    import credit_risk.train as tr

    prod = tmp_path / "models"
    prod.mkdir()
    (prod / "model.joblib").write_text("modèle de production existant")
    small_params["quality_gate"]["min_roc_auc"] = 0.99  # impossible à atteindre
    monkeypatch.setattr(tr.config, "load_params", lambda: small_params)
    monkeypatch.setattr(tr.config, "MODELS_DIR", prod)
    monkeypatch.setattr(tr.config, "REFERENCE_DATA", tmp_path / "ref.csv")
    monkeypatch.setattr(tr, "CANDIDATE_DIR", tmp_path / "candidate")
    monkeypatch.setattr(tr, "download", lambda *a: credit_df)
    monkeypatch.setenv("DISABLE_MLFLOW", "1")
    with pytest.raises(SystemExit) as exc:
        tr.main()
    assert exc.value.code == 1
    assert (prod / "model.joblib").read_text() == "modèle de production existant"
