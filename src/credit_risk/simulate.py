"""Simule du trafic de production vers l'API, avec ou sans dérive.

python -m credit_risk.simulate --n 200            # trafic « normal » (échantillons du jeu réel)
python -m credit_risk.simulate --n 200 --drift    # population qui change (montants, durées, âges)
"""

from __future__ import annotations

import argparse

import httpx
import numpy as np
import pandas as pd

from . import config
from .data import FEATURES, validate


def shift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["credit_amount"] = df["credit_amount"] * rng.uniform(1.8, 2.6, len(df))
    df["duration"] = (df["duration"] * 1.7).clip(upper=72)
    df["age"] = (df["age"] + 15).clip(upper=90)
    df["purpose"] = rng.choice(["new car", "business"], len(df))
    df["checking_status"] = "no checking"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = validate(pd.read_csv(config.RAW_DATA))[FEATURES].sample(args.n, replace=True, random_state=args.seed)
    if args.drift:
        df = shift(df, rng)

    decisions: dict[str, int] = {}
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for row in df.to_dict(orient="records"):
            r = client.post("/predict", json=row)
            r.raise_for_status()
            d = r.json()["decision"]
            decisions[d] = decisions.get(d, 0) + 1
    print(f"{args.n} requêtes envoyées ({'avec' if args.drift else 'sans'} dérive) : {decisions}")


if __name__ == "__main__":
    main()
