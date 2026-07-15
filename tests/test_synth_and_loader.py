import pandas as pd
import pytest

from churn.data.loader import clean, load_customers
from churn.data.synth import generate_customers


def test_synth_is_deterministic():
    a = generate_customers(200, seed=7)
    b = generate_customers(200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synth_has_both_classes():
    df = generate_customers(1000)
    assert set(df["churned"].unique()) == {0, 1}


def test_clean_fills_missing_and_clips_negatives():
    df = pd.DataFrame({"a": [1.0, None, -5.0], "b": [2.0, 2.0, 2.0]})
    out = clean(df)
    assert out["a"].isna().sum() == 0
    assert (out["a"] >= 0).all()


def test_loader_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"tenure_months": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_customers(bad)


def test_loader_roundtrip(tmp_path):
    path = tmp_path / "customers.csv"
    generate_customers(50).to_csv(path, index=False)
    df = load_customers(path)
    assert len(df) == 50
