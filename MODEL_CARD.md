# Model Card — Customer Churn Classifier

## Overview

Binary classifier that predicts whether a customer will churn, given usage,
billing, and support-interaction features. Served via the `/predict` endpoint
in this repo, returning a churn probability and a `low` / `medium` / `high`
risk band.

## Intended use

Prioritization signal for retention teams — flag high-risk accounts for
proactive outreach. Not intended as the sole basis for automated actions
(e.g., auto-cancelling service, auto-applying discounts) without human review.

## Training data

- **Source**: `src/churn/data/synth.py` — a deterministic synthetic generator
  (seed 42) that mirrors the shape of a telecom retention dataset. There is
  no real customer data in this repo; `scripts/train.py --data <csv>` swaps
  in a real extract with the same schema.
- **Size**: 5,000 rows, 80/20 train/test split, stratified on the label.
- **Base rate**: 40.4% churned in the synthetic set.
- **Features (8 raw + 3 derived)**: `tenure_months`, `monthly_charges`,
  `total_charges`, `support_tickets`, `avg_call_minutes`, `data_usage_gb`,
  `late_payments`, `contract_monthly`, plus engineered
  `charges_per_month_of_tenure`, `tickets_per_year`, `is_new_customer`.

## Models

Two interchangeable backends behind one `sklearn.Pipeline` (`StandardScaler`
+ classifier), selected via `--model`:

| Model | Holdout accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV F1 (mean ± std) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.736 | 0.696 | 0.616 | 0.654 | 0.804 | 0.637 ± 0.016 |
| Gradient Boosting | 0.724 | 0.681 | 0.597 | 0.636 | 0.794 | 0.625 ± 0.023 |

Numbers reproduced by running `python scripts/train.py --model <name>` on the
default synthetic dataset (seed 42). Logistic Regression is the default: it's
marginally stronger on this dataset and its coefficients are directly
interpretable, which matters more than a few points of accuracy for a signal
that's handed to a retention team, not auto-acted on.

## Feature importance (Logistic Regression coefficients, standardized)

| Feature | Coefficient | Direction |
|---|---|---|
| `tenure_months` | -0.941 | longer tenure → lower churn risk |
| `support_tickets` | +0.655 | more tickets → higher churn risk |
| `contract_monthly` | +0.551 | month-to-month → higher churn risk |
| `late_payments` | +0.372 | more late payments → higher churn risk |
| `monthly_charges` | +0.150 | higher bill → higher churn risk |
| `charges_per_month_of_tenure` | +0.150 | higher spend-per-tenure → higher churn risk |
| `tickets_per_year` | -0.133 | mixed signal once tenure is controlled for |
| `avg_call_minutes` | +0.086 | weak positive |
| `is_new_customer` | +0.048 | weak positive |
| `data_usage_gb` | -0.023 | negligible |
| `total_charges` | +0.016 | negligible |

Tenure, support-ticket volume, contract type, and late payments dominate —
consistent with how the synthetic label is constructed, and directionally
consistent with published telecom churn literature.

## Limitations

- **Synthetic data**: coefficients and metrics reflect the synthetic
  generator's assumptions, not a real customer base. Retrain on real data
  before using in production; expect metrics to shift.
- **No temporal validation**: the holdout split is random, not time-based, so
  it doesn't test for concept drift (e.g., pricing changes, new product
  lines). Add a time-based split before deploying against live traffic.
- **No fairness/subgroup analysis**: the model has not been evaluated for
  disparate impact across customer segments, since the synthetic data has no
  demographic attributes.
- **Threshold**: the default 0.5 decision threshold (`CHURN_THRESHOLD`) is
  not tuned to a business cost function (cost of a missed churn vs. cost of
  an unnecessary retention offer) — tune it against real outcomes.

## Reproducing these numbers

```bash
pip install -r requirements-dev.txt
python scripts/train.py --model logistic
python scripts/train.py --model gradient_boosting
```
