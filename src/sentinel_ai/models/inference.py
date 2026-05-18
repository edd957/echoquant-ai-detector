from __future__ import annotations

from typing import Any

import pandas as pd

from sentinel_ai.data.synthetic import FEATURE_COLUMNS
from sentinel_ai.schemas import PredictionResponse, Transaction


def score_transaction(model: Any, transaction: Transaction) -> PredictionResponse:
    frame = pd.DataFrame([transaction.model_dump()])[FEATURE_COLUMNS]
    probability = float(model.predict_proba(frame)[0, 1])
    band = risk_band(probability)
    return PredictionResponse(
        risk_probability=round(probability, 4),
        risk_band=band,
        recommended_action=recommended_action(band),
        top_signals=top_signals(transaction),
    )


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "critical"
    if probability >= 0.5:
        return "high"
    if probability >= 0.25:
        return "medium"
    return "low"


def recommended_action(band: str) -> str:
    return {
        "critical": "Block transaction and escalate to manual review.",
        "high": "Step-up authentication and queue for analyst review.",
        "medium": "Allow with enhanced monitoring.",
        "low": "Allow transaction.",
    }[band]


def top_signals(transaction: Transaction) -> list[str]:
    signals: list[str] = []
    if transaction.amount > 500:
        signals.append("Large transaction amount")
    if transaction.device_trust_score < 0.35:
        signals.append("Low device trust")
    if transaction.failed_login_count >= 3:
        signals.append("Recent failed login activity")
    if transaction.is_cross_border:
        signals.append("Cross-border transaction")
    if transaction.is_new_device:
        signals.append("New device")
    if transaction.merchant_risk_score >= 0.65:
        signals.append("High merchant risk")
    return signals or ["No dominant risk signal"]
