# Customer Churn Prediction

**Customer Retention Intelligence & Churn Prediction Platform** — an end-to-end machine learning system that identifies at-risk customers and serves real-time churn predictions through a REST API.

Built to support data-driven retention: customer usage, billing and support data flows through a cleaning and feature-engineering pipeline into scikit-learn classifiers, evaluated with cross-validation and served with FastAPI.

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  CSV extract │──▶│  Clean +     │──▶│  Train +      │──▶│  Persist     │
│  (usage,     │   │  Feature     │   │  Evaluate     │   │  artifact    │
│  billing,    │   │  Engineering │   │  (CV, F1,     │   │  (joblib)    │
│  support)    │   │              │   │  ROC-AUC)     │   │              │
└──────────────┘   └──────────────┘   └───────────────┘   └──────┬───────┘
                                                                 │
                                                    ┌────────────▼────────────┐
                                                    │   FastAPI  /predict     │
                                                    │   probability + risk    │
                                                    │   band in real time     │
                                                    └─────────────────────────┘
```

## Features

- **Data pipeline** — schema validation, deduplication, median imputation, negative clipping
- **Feature engineering** — spend-per-tenure ratio, annualized ticket rate, new-customer flag
- **Two model backends** — Logistic Regression (fast, interpretable) and Gradient Boosting (higher accuracy), both behind a scikit-learn `Pipeline` with scaling
- **Rigorous evaluation** — precision, recall, F1, ROC-AUC, confusion matrix, k-fold cross-validation
- **Real-time API** — `/predict` returns churn probability + risk band (`low` / `medium` / `high`)
- **Synthetic data generator** — deterministic, learnable dataset so everything runs with zero external data
- **Production-ready** — model artifact persistence, Dockerfile, CI (lint + tests on Python 3.11/3.12)

## Quickstart

```bash
git clone https://github.com/saianthireddy/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# run the test suite
pytest -q

# train a model (synthetic data by default) and print metrics
python scripts/train.py --model gradient_boosting

# start the API
export PYTHONPATH=src
uvicorn churn.api.main:app --reload
```

Score a customer:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 285.0,
    "support_tickets": 6, "avg_call_minutes": 250.0, "data_usage_gb": 12.5,
    "late_payments": 3, "contract_monthly": 1
  }'
```

```json
{ "churn_probability": 0.91, "will_churn": true, "risk_band": "high" }
```

## Train on your own data

Provide a CSV with the columns `tenure_months, monthly_charges, total_charges, support_tickets, avg_call_minutes, data_usage_gb, late_payments, contract_monthly, churned`:

```bash
python scripts/train.py --data path/to/customers.csv --model gradient_boosting
```

The trained pipeline is saved to `artifacts/churn_model.joblib` and picked up automatically by the API.

## Project structure

```
src/churn/
├── config.py             # env-driven settings
├── data/                 # CSV loader + synthetic data generator
├── features/             # feature engineering
├── models/               # pipeline build, train, save/load
├── evaluation/           # holdout metrics + cross-validation
└── api/                  # FastAPI app + schemas
scripts/train.py          # training CLI with metrics report
tests/                    # 16 deterministic tests
```

## Testing & CI

```bash
pytest -q                       # 16 tests
ruff check src tests scripts
```

CI runs lint and the full suite on every push and pull request.

## License

MIT
