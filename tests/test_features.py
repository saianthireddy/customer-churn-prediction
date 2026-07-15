from churn.data.synth import generate_customers
from churn.features.engineering import ALL_FEATURES, engineer_features, split_xy


def test_derived_columns_added():
    frame = engineer_features(generate_customers(100))
    for col in ["charges_per_month_of_tenure", "tickets_per_year", "is_new_customer"]:
        assert col in frame.columns


def test_new_customer_flag():
    frame = engineer_features(generate_customers(500))
    assert ((frame["tenure_months"] <= 6) == (frame["is_new_customer"] == 1)).all()


def test_split_xy_shapes():
    frame = engineer_features(generate_customers(120))
    X, y = split_xy(frame)
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y) == 120
