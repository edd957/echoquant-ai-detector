from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "customer_age",
    "account_age_days",
    "merchant_risk_score",
    "country_risk_score",
    "hour_of_day",
    "device_trust_score",
    "failed_login_count",
    "is_cross_border",
    "is_new_device",
]

TARGET_COLUMN = "is_high_risk"


def generate_transactions(rows: int = 5000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    """Generate realistic synthetic transaction data with optional distribution drift."""

    rng = np.random.default_rng(seed)
    amount_scale = 115 if not drift else 155
    amount = rng.lognormal(mean=4.2, sigma=0.9, size=rows) * (amount_scale / 100)
    customer_age = rng.integers(18, 82, rows)
    account_age_days = rng.gamma(shape=2.5, scale=180, size=rows).astype(int)
    merchant_risk_score = rng.beta(2.0 if not drift else 2.8, 5.0, rows)
    country_risk_score = rng.beta(1.8 if not drift else 2.4, 5.5, rows)
    hour_of_day = rng.integers(0, 24, rows)
    device_trust_score = rng.beta(6.0 if not drift else 4.2, 2.4, rows)
    failed_login_count = rng.poisson(0.7 if not drift else 1.1, rows)
    is_cross_border = rng.binomial(1, 0.16 if not drift else 0.24, rows)
    is_new_device = rng.binomial(1, 0.19 if not drift else 0.28, rows)

    night_activity = ((hour_of_day <= 5) | (hour_of_day >= 22)).astype(int)
    logit = (
        -4.4
        + 0.006 * amount
        + 2.5 * merchant_risk_score
        + 2.1 * country_risk_score
        - 2.2 * device_trust_score
        + 0.34 * failed_login_count
        + 0.85 * is_cross_border
        + 0.9 * is_new_device
        + 0.42 * night_activity
        - 0.0012 * account_age_days
    )
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, np.clip(probability, 0.01, 0.98), rows)

    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "customer_age": customer_age,
            "account_age_days": account_age_days,
            "merchant_risk_score": merchant_risk_score.round(4),
            "country_risk_score": country_risk_score.round(4),
            "hour_of_day": hour_of_day,
            "device_trust_score": device_trust_score.round(4),
            "failed_login_count": failed_login_count,
            "is_cross_border": is_cross_border.astype(bool),
            "is_new_device": is_new_device.astype(bool),
            TARGET_COLUMN: target.astype(int),
        }
    )


def write_default_datasets(
    reference_path: Path,
    current_path: Path,
    rows: int = 5000,
    seed: int = 42,
) -> tuple[Path, Path]:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    generate_transactions(rows=rows, seed=seed, drift=False).to_csv(reference_path, index=False)
    generate_transactions(rows=max(rows // 2, 100), seed=seed + 7, drift=True).to_csv(
        current_path, index=False
    )
    return reference_path, current_path
