"""Deterministic synthetic customer dataset for demos and tests.

Mirrors the shape of a telecom retention dataset: usage, billing and
support interactions with a churn label whose signal is learnable.
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "avg_call_minutes", "data_usage_gb", "late_payments", "contract_monthly",
]


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate *n* synthetic customers with a realistic churn signal."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    tickets = rng.poisson(1.2, n)
    call_minutes = rng.normal(300, 90, n).clip(0).round(1)
    data_gb = rng.gamma(2.0, 4.0, n).round(2)
    late = rng.poisson(0.4, n)
    contract_monthly = rng.integers(0, 2, n)

    # churn probability: short tenure, many tickets, late payments,
    # month-to-month contracts and high bills push churn up
    logit = (
        -1.2
        - 0.045 * tenure
        + 0.55 * tickets
        + 0.65 * late
        + 1.1 * contract_monthly
        + 0.012 * monthly
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": (tenure * monthly).round(2),
        "support_tickets": tickets,
        "avg_call_minutes": call_minutes,
        "data_usage_gb": data_gb,
        "late_payments": late,
        "contract_monthly": contract_monthly,
        "churned": churned,
    })
