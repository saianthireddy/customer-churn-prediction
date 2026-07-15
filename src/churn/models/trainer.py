"""Train, persist and load the churn classification pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.engineering import ALL_FEATURES

MODELS = {
    "logistic": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
}


def build_pipeline(model: str = "logistic", random_state: int = 42) -> Pipeline:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {sorted(MODELS)}")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MODELS[model](random_state)),
    ])


def train(
    X: pd.DataFrame, y: pd.Series, model: str = "logistic", random_state: int = 42
) -> Pipeline:
    pipeline = build_pipeline(model, random_state)
    pipeline.fit(X[ALL_FEATURES], y)
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
