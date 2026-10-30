import numpy as np
import pandas as pd
from conftest import make_credit_df

from credit_risk.data import CATEGORICAL, NUMERIC, validate
from credit_risk.drift import detect_drift, psi_categorical, psi_numeric


def test_psi_is_zero_for_identical_distributions():
    s = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert psi_numeric(s, s) < 1e-9


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.1
    assert psi_numeric(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.2


def test_psi_categorical_handles_new_category():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 30 + ["c"] * 70)
    assert psi_categorical(ref, cur) > 0.2


def test_no_drift_on_same_population():
    ref = validate(make_credit_df(1000, seed=1))
    cur = validate(make_credit_df(500, seed=2))
    report = detect_drift(ref, cur, NUMERIC, CATEGORICAL)
    assert not report["dataset_drift"]


def test_drift_on_shifted_population():
    ref = validate(make_credit_df(1000, seed=1))
    cur = validate(make_credit_df(500, seed=2))
    cur["credit_amount"] *= 2.5
    cur["duration"] = cur["duration"] * 1.7
    cur["age"] += 15
    cur["purpose"] = "business"
    cur["checking_status"] = "no checking"
    report = detect_drift(ref, cur, NUMERIC, CATEGORICAL, drift_share_threshold=0.2)
    assert {"credit_amount", "age", "purpose", "checking_status"} <= set(report["drifted_features"])
    assert report["dataset_drift"]


def test_critical_feature_drift_triggers_alert_even_below_share_threshold():
    ref = validate(make_credit_df(1000, seed=1))
    cur = validate(make_credit_df(500, seed=2))
    cur["credit_amount"] *= 2.5  # une seule variable dérive : 1/20 = 5 %
    report = detect_drift(ref, cur, NUMERIC, CATEGORICAL, drift_share_threshold=0.3)
    assert not report["dataset_drift"]
    report = detect_drift(ref, cur, NUMERIC, CATEGORICAL, drift_share_threshold=0.3,
                          critical_features=["credit_amount"])
    assert report["dataset_drift"] and report["critical_drifted"] == ["credit_amount"]
