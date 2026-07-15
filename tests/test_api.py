from fastapi.testclient import TestClient

from churn.api.main import app

client = TestClient(app)

CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "support_tickets": 6,
    "avg_call_minutes": 250.0,
    "data_usage_gb": 12.5,
    "late_payments": 3,
    "contract_monthly": 1,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_high_risk_customer():
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "will_churn", "risk_band"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_validates_input():
    bad = dict(CUSTOMER, tenure_months=-1)
    assert client.post("/predict", json=bad).status_code == 422
