from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException

from sentinel_ai import __version__
from sentinel_ai.agents.workflow import analyze_transaction
from sentinel_ai.config import get_settings
from sentinel_ai.models.inference import score_transaction
from sentinel_ai.models.train import load_model
from sentinel_ai.rag.vector_store import LocalPolicyRetriever
from sentinel_ai.schemas import AnalysisResponse, HealthResponse, PredictionResponse, Transaction

app = FastAPI(
    title="Sentinel AI Risk Platform",
    version=__version__,
    description=(
        "ML and AI service for risk scoring, retrieval-grounded analysis, "
        "and drift-aware operations."
    ),
)


@lru_cache
def get_model() -> Any:
    settings = get_settings()
    return load_model(settings.model_path)


@lru_cache
def get_retriever() -> LocalPolicyRetriever:
    settings = get_settings()
    return LocalPolicyRetriever.from_markdown(settings.policy_corpus_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model_loaded=settings.model_path.exists(),
        version=__version__,
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    try:
        return score_transaction(get_model(), transaction)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing. Run `sentinel train`.",
        ) from exc


@app.post("/v1/analyze", response_model=AnalysisResponse)
def analyze(transaction: Transaction) -> AnalysisResponse:
    try:
        return analyze_transaction(get_model(), get_retriever(), transaction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
