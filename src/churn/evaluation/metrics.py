"""Model evaluation: holdout metrics, confusion matrix and cross-validation."""

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


@dataclass
class Evaluation:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(pipeline, X: pd.DataFrame, y: pd.Series) -> Evaluation:
    predictions = pipeline.predict(X)
    scores = pipeline.predict_proba(X)[:, 1]
    return Evaluation(
        accuracy=round(accuracy_score(y, predictions), 4),
        precision=round(precision_score(y, predictions), 4),
        recall=round(recall_score(y, predictions), 4),
        f1=round(f1_score(y, predictions), 4),
        roc_auc=round(roc_auc_score(y, scores), 4),
        confusion=confusion_matrix(y, predictions).tolist(),
    )


def cross_validate_f1(pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    scores = cross_val_score(pipeline, X, y, cv=folds, scoring="f1")
    return {
        "folds": folds,
        "f1_mean": round(float(scores.mean()), 4),
        "f1_std": round(float(scores.std()), 4),
    }
