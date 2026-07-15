"""FastAPI service for real-time churn prediction.

Run locally:  uvicorn churn.api.main:app --reload

Loads the persisted model artifact if present; otherwise trains a
fresh pipeline on synthetic data so the service always boots.
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI

from .. import __version__
from ..config import get_settings
from ..data.synth import generate_customers
from ..features.engineering import ALL_FEATURES, engineer_features, split_xy
from ..models.trainer import load_model, train
from .schemas import Customer, HealthResponse, Prediction

app = FastAPI(title="Churn Prediction API", version=__version__)


@lru_cache(maxsize=1)
def get_pipeline():
    settings = get_settings()
    try:
        return load_model(settings.model_path)
    except FileNotFoundError:
        frame = engineer_features(generate_customers(2000, settings.random_state))
        X, y = split_xy(frame)
        return train(X, y, random_state=settings.random_state)


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok", version=__version__, model=type(pipeline[-1]).__name__
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    settings = get_settings()
    frame = engineer_features(pd.DataFrame([customer.model_dump()]))
    probability = float(get_pipeline().predict_proba(frame[ALL_FEATURES])[0, 1])
    return Prediction(
        churn_probability=round(probability, 4),
        will_churn=probability >= settings.churn_threshold,
        risk_band=risk_band(probability),
    )
