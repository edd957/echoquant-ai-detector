from pathlib import Path

from fastapi.testclient import TestClient

from sentinel_ai.api.main import app
from sentinel_ai.config import get_settings
from sentinel_ai.data.synthetic import generate_transactions
from sentinel_ai.models.train import train_model


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    data_path = tmp_path / "transactions.csv"
    generate_transactions(rows=600, seed=31).to_csv(data_path, index=False)
    train_model(data_path, model_path, metrics_path, seed=31)

    settings = get_settings()
    monkeypatch.setattr(settings, "model_path", model_path)
    from sentinel_ai.api import main

    main.get_model.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/v1/predict",
        json={
            "amount": 250,
            "customer_age": 45,
            "account_age_days": 500,
            "merchant_risk_score": 0.35,
            "country_risk_score": 0.25,
            "hour_of_day": 14,
            "device_trust_score": 0.85,
            "failed_login_count": 0,
            "is_cross_border": False,
            "is_new_device": False,
        },
    )

    assert response.status_code == 200
    assert "risk_probability" in response.json()
