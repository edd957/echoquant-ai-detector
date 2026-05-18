from pathlib import Path

from sentinel_ai.data.synthetic import generate_transactions
from sentinel_ai.models.inference import score_transaction
from sentinel_ai.models.train import load_model, train_model
from sentinel_ai.schemas import Transaction


def test_train_and_score_transaction(tmp_path: Path) -> None:
    data_path = tmp_path / "transactions.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    generate_transactions(rows=600, seed=21).to_csv(data_path, index=False)

    metrics = train_model(data_path, model_path, metrics_path, seed=21)
    model = load_model(model_path)
    response = score_transaction(
        model,
        Transaction(
            amount=900,
            customer_age=33,
            account_age_days=35,
            merchant_risk_score=0.8,
            country_risk_score=0.7,
            hour_of_day=2,
            device_trust_score=0.2,
            failed_login_count=5,
            is_cross_border=True,
            is_new_device=True,
        ),
    )

    assert metrics["roc_auc"] > 0.7
    assert 0 <= response.risk_probability <= 1
    assert response.risk_band in {"low", "medium", "high", "critical"}
