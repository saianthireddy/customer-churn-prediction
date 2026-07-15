"""Load and validate customer records from CSV extracts."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "avg_call_minutes", "data_usage_gb", "late_payments", "contract_monthly",
}


def load_customers(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    """Read a customer CSV and validate the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Customer file not found: {path}")
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if require_label and "churned" not in df.columns:
        raise ValueError("Missing label column 'churned'")

    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fill numeric gaps with medians, clip negatives."""
    df = df.drop_duplicates().copy()
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].fillna(df[numeric].median()).clip(lower=0)
    return df
