"""Feature engineering: derive signals the raw extract does not carry."""

import pandas as pd

BASE_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "avg_call_minutes", "data_usage_gb", "late_payments", "contract_monthly",
]
DERIVED_FEATURES = ["charges_per_month_of_tenure", "tickets_per_year", "is_new_customer"]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with derived features appended."""
    out = df.copy()
    tenure = out["tenure_months"].clip(lower=1)
    out["charges_per_month_of_tenure"] = (out["total_charges"] / tenure).round(4)
    out["tickets_per_year"] = (out["support_tickets"] * 12 / tenure).round(4)
    out["is_new_customer"] = (out["tenure_months"] <= 6).astype(int)
    return out


def split_xy(df: pd.DataFrame):
    """Split an engineered frame into (X, y)."""
    return df[ALL_FEATURES], df["churned"]
