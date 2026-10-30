"""Job de monitoring : compare les requêtes de production à la référence et produit un rapport.

Usage : python -m credit_risk.monitor [--html]
Code de sortie 2 en cas de dérive (pratique pour un cron ou une GitHub Action planifiée).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from . import config
from .data import CATEGORICAL, FEATURES, NUMERIC
from .drift import detect_drift


def load_current(path=config.PREDICTIONS_LOG) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FEATURES)
    rows = [json.loads(line)["features"] for line in path.read_text("utf-8").splitlines() if line]
    return pd.DataFrame(rows, columns=FEATURES)


def evidently_html(reference: pd.DataFrame, current: pd.DataFrame, out) -> bool:
    """Rapport visuel Evidently, optionnel (pip install evidently)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        try:  # ancienne API (< 0.7)
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError:
            return False
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    (result or report).save_html(str(out))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="Générer aussi un rapport HTML Evidently")
    args = ap.parse_args()

    params = config.load_params()["monitoring"]
    reference = pd.read_csv(config.REFERENCE_DATA)
    current = load_current()
    if len(current) < params["min_samples"]:
        print(f"Pas assez de requêtes ({len(current)} < {params['min_samples']}) : rien à analyser.")
        return

    report = detect_drift(
        reference, current, NUMERIC, CATEGORICAL,
        params["psi_threshold"], params["drift_share_threshold"],
        params.get("critical_features", []),
    )
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DRIFT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_keys = ("drift_share", "dataset_drift", "drifted_features", "critical_drifted")
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))

    if args.html:
        out = config.REPORTS_DIR / "drift_report.html"
        print("Rapport HTML :", out if evidently_html(reference, current, out) else "evidently non installé")

    if report["dataset_drift"]:
        print("ALERTE : dérive des données détectée — envisager un réentraînement.")
        sys.exit(2)


if __name__ == "__main__":
    main()
