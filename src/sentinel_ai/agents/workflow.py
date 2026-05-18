from __future__ import annotations

from typing import Any

from sentinel_ai.models.inference import score_transaction
from sentinel_ai.rag.vector_store import LocalPolicyRetriever
from sentinel_ai.schemas import AnalysisResponse, Transaction


def analyze_transaction(
    model: Any,
    retriever: LocalPolicyRetriever,
    transaction: Transaction,
) -> AnalysisResponse:
    prediction = score_transaction(model, transaction)
    query = " ".join(prediction.top_signals + [prediction.risk_band, prediction.recommended_action])
    context = retriever.search(query, top_k=3)
    actions = _actions_for_band(prediction.risk_band)
    summary = (
        f"The transaction is classified as {prediction.risk_band} risk "
        f"with probability {prediction.risk_probability}. "
        f"Primary signals: {', '.join(prediction.top_signals)}. "
        f"Recommended action: {prediction.recommended_action}"
    )
    return AnalysisResponse(
        prediction=prediction,
        retrieved_context=context,
        analyst_summary=summary,
        actions=actions,
    )


def _actions_for_band(band: str) -> list[str]:
    if band == "critical":
        return [
            "Block payment authorization.",
            "Create manual review case.",
            "Request strong customer authentication.",
            "Attach retrieved policy context to the case.",
        ]
    if band == "high":
        return [
            "Trigger step-up authentication.",
            "Queue transaction for same-day analyst review.",
            "Monitor related account activity for 24 hours.",
        ]
    if band == "medium":
        return ["Approve with enhanced monitoring.", "Sample for post-transaction QA."]
    return ["Approve transaction.", "Log decision for monitoring."]
