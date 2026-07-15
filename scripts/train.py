"""CLI: train the churn model and print evaluation metrics.

Usage:
    python scripts/train.py                     # synthetic data, logistic
    python scripts/train.py --model gradient_boosting
    python scripts/train.py --data data/customers.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import train_test_split

from churn.config import get_settings
from churn.data.loader import load_customers
from churn.data.synth import generate_customers
from churn.evaluation.metrics import cross_validate_f1, evaluate
from churn.features.engineering import engineer_features, split_xy
from churn.models.trainer import save_model, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="CSV of labelled customers (default: synthetic)")
    parser.add_argument("--model", default="logistic", choices=["logistic", "gradient_boosting"])
    parser.add_argument("--rows", type=int, default=5000, help="synthetic rows if no --data")
    args = parser.parse_args()

    settings = get_settings()
    if args.data:
        df = load_customers(args.data)
    else:
        df = generate_customers(args.rows, settings.random_state)
    frame = engineer_features(df)
    X, y = split_xy(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )

    pipeline = train(X_train, y_train, model=args.model, random_state=settings.random_state)
    holdout = evaluate(pipeline, X_test, y_test)
    cv = cross_validate_f1(pipeline, X_train, y_train, settings.cv_folds)
    path = save_model(pipeline, settings.model_path)

    report = {"holdout": holdout.as_dict(), "cross_validation": cv, "artifact": str(path)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
