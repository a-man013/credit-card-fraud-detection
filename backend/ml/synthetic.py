"""
Synthetic dataset generator for FraudShield AI demo mode.
Produces a realistic-looking fraud dataset when the Kaggle CSV is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_dataset(
    n_samples: int = 2000,
    fraud_ratio: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic credit-card transaction dataset with:
    - Time  : seconds elapsed from first transaction
    - V1–V28: PCA-like anonymised features
    - Amount: transaction amount (USD)
    - Class : 0 = genuine, 1 = fraud
    """
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_genuine = n_samples - n_fraud

    # ── Genuine transactions ──────────────────────────────────────────────────
    gen_amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n_genuine)
    gen_amounts = np.clip(gen_amounts, 0.5, 5000.0)

    gen_time = np.sort(rng.uniform(0, 172800, size=n_genuine))

    gen_v = rng.normal(0, 1, size=(n_genuine, 28))

    # ── Fraudulent transactions ───────────────────────────────────────────────
    fraud_amounts = rng.lognormal(mean=4.5, sigma=0.9, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 1.0, 8000.0)

    fraud_time = rng.uniform(0, 172800, size=n_fraud)

    fraud_v = rng.normal(0, 1.5, size=(n_fraud, 28))
    # Introduce separation: V1, V3, V14 tend to be more negative in fraud
    fraud_v[:, 0] -= rng.uniform(1.5, 3.0, size=n_fraud)
    fraud_v[:, 2] -= rng.uniform(1.0, 2.5, size=n_fraud)
    fraud_v[:, 13] -= rng.uniform(2.0, 4.0, size=n_fraud)
    # V17 more negative in fraud
    fraud_v[:, 16] -= rng.uniform(1.0, 2.0, size=n_fraud)

    # ── Combine ───────────────────────────────────────────────────────────────
    times = np.concatenate([gen_time, fraud_time])
    amounts = np.concatenate([gen_amounts, fraud_amounts])
    v_features = np.vstack([gen_v, fraud_v])
    labels = np.array([0] * n_genuine + [1] * n_fraud)

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([times, v_features, amounts, labels])
    df = pd.DataFrame(data, columns=cols)

    # Round for display
    df["Time"] = df["Time"].round(0).astype(int)
    df["Amount"] = df["Amount"].round(2)
    for v in [f"V{i}" for i in range(1, 29)]:
        df[v] = df[v].round(6)
    df["Class"] = df["Class"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_time_series(df: pd.DataFrame) -> list[dict]:
    """
    Bin transactions into ~10 daily buckets and return
    [{date, genuine, fraud}, ...] for the dashboard chart.
    """
    df = df.copy()
    max_time = df["Time"].max() if df["Time"].max() > 0 else 172800
    df["day"] = (df["Time"] / (max_time / 10)).astype(int).clip(0, 9)

    base_date = datetime(2024, 1, 1)
    rows = []
    for day in range(10):
        subset = df[df["day"] == day]
        rows.append(
            {
                "date": (base_date + timedelta(days=day)).strftime("%b %d"),
                "genuine": int((subset["Class"] == 0).sum()),
                "fraud": int((subset["Class"] == 1).sum()),
            }
        )
    return rows
