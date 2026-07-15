"""Central configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "artifacts/churn_model.joblib")
    )
    random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    test_size: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2")))
    cv_folds: int = field(default_factory=lambda: int(os.getenv("CV_FOLDS", "5")))
    churn_threshold: float = field(
        default_factory=lambda: float(os.getenv("CHURN_THRESHOLD", "0.5"))
    )


def get_settings() -> Settings:
    return Settings()
