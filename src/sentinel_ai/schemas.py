from pydantic import BaseModel, Field


class Transaction(BaseModel):
    amount: float = Field(ge=0, description="Transaction amount in USD.")
    customer_age: int = Field(ge=18, le=100)
    account_age_days: int = Field(ge=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    country_risk_score: float = Field(ge=0, le=1)
    hour_of_day: int = Field(ge=0, le=23)
    device_trust_score: float = Field(ge=0, le=1)
    failed_login_count: int = Field(ge=0, le=25)
    is_cross_border: bool
    is_new_device: bool


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_band: str
    recommended_action: str
    top_signals: list[str]


class RetrievalResult(BaseModel):
    title: str
    score: float
    text: str


class AnalysisResponse(BaseModel):
    prediction: PredictionResponse
    retrieved_context: list[RetrievalResult]
    analyst_summary: str
    actions: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
