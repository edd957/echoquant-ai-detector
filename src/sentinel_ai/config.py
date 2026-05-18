from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for local development and deployment."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    env: str = "local"
    random_seed: int = 42
    model_path: Path = Field(default=Path("artifacts/models/risk_model.joblib"))
    metrics_path: Path = Field(default=Path("artifacts/reports/metrics.json"))
    drift_report_path: Path = Field(default=Path("artifacts/reports/drift_report.json"))
    reference_data_path: Path = Field(default=Path("data/processed/reference_transactions.csv"))
    current_data_path: Path = Field(default=Path("data/processed/current_transactions.csv"))
    policy_corpus_path: Path = Field(default=Path("docs/policy_corpus.md"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
