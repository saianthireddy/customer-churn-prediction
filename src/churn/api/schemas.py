from pydantic import BaseModel, Field


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=1200)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    avg_call_minutes: float = Field(..., ge=0)
    data_usage_gb: float = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    contract_monthly: int = Field(..., ge=0, le=1)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
