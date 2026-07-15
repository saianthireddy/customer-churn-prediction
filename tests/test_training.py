import pytest

from churn.data.synth import generate_customers
from churn.evaluation.metrics import cross_validate_f1, evaluate
from churn.features.engineering import engineer_features, split_xy
from churn.models.trainer import build_pipeline, load_model, save_model, train


def make_xy(n=800):
    frame = engineer_features(generate_customers(n))
    return split_xy(frame)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_pipeline("neural-net-3000")


def test_model_beats_chance():
    X, y = make_xy()
    pipeline = train(X, y)
    result = evaluate(pipeline, X, y)
    assert result.roc_auc > 0.7
    assert result.f1 > 0.5


def test_confusion_matrix_shape():
    X, y = make_xy(400)
    result = evaluate(train(X, y), X, y)
    assert len(result.confusion) == 2
    assert sum(sum(row) for row in result.confusion) == 400


def test_cross_validation_reports_folds():
    X, y = make_xy(500)
    cv = cross_validate_f1(build_pipeline(), X, y, folds=3)
    assert cv["folds"] == 3
    assert 0.0 <= cv["f1_mean"] <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    X, y = make_xy(300)
    pipeline = train(X, y)
    path = save_model(pipeline, tmp_path / "model.joblib")
    loaded = load_model(path)
    assert (loaded.predict(X[:10]) == pipeline.predict(X[:10])).all()
