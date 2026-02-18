"""Tests for ml/predictor.py"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS
from ml.predictor import (
    LABEL_NAMES,
    Prediction,
    _classify_confidence,
    load_models,
    predict_single,
)


def _make_ohlcv(n: int = 200, base_price: float = 10000, seed: int = 42) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(0.001, 0.02, n)
    close = base_price * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": close * (1 + np.random.uniform(0, 0.02, n)),
            "low": close * (1 - np.random.uniform(0, 0.02, n)),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n),
            "trading_value": np.random.uniform(1e9, 1e10, n),
        },
        index=dates,
    )


def _train_and_save_models(model_dir, cache):
    """Helper to train and save models for testing."""
    from ml.features import build_feature_matrix
    from ml.trainer import save_models, train_classifier, train_regressor

    # Insert data
    for i, ticker in enumerate(["069500", "091160", "091170"]):
        df = _make_ohlcv(n=200, seed=42 + i)
        cache.upsert_ohlcv(ticker, df)

    matrix = build_feature_matrix(
        cache, tickers=["069500", "091160", "091170"], add_label=True
    )
    if matrix.empty:
        pytest.skip("Feature matrix is empty")

    clf, clf_result = train_classifier(matrix, n_splits=2)
    reg, reg_result = train_regressor(matrix, n_splits=2)
    features = [c for c in FEATURE_COLUMNS if c in matrix.columns]

    save_models(clf, reg, clf_result, reg_result, features, model_dir=model_dir)

    return clf, reg, features


class TestClassifyConfidence:
    def test_high_confidence(self):
        assert _classify_confidence(0.75) == "high"

    def test_medium_confidence(self):
        assert _classify_confidence(0.50) == "medium"

    def test_low_confidence(self):
        assert _classify_confidence(0.35) == "low"

    def test_boundary_high(self):
        assert _classify_confidence(0.60) == "high"

    def test_boundary_medium(self):
        assert _classify_confidence(0.45) == "medium"


class TestLoadModels:
    def test_load_saved_models(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        _train_and_save_models(model_dir, cache)

        clf, reg, features = load_models(model_dir)
        assert clf is not None
        assert reg is not None
        assert len(features) > 0

    def test_no_models_raises(self, tmp_path):
        model_dir = tmp_path / "empty_models"
        model_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_models(model_dir)


class TestPredictSingle:
    def test_generates_prediction(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        clf, reg, features = _train_and_save_models(model_dir, cache)

        pred = predict_single("069500", cache, clf, reg, features)
        assert pred is not None
        assert isinstance(pred, Prediction)
        assert pred.ticker == "069500"
        assert pred.predicted_label in ("up", "flat", "down")
        assert pred.confidence in ("high", "medium", "low")

    def test_prediction_probabilities_sum_to_one(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        clf, reg, features = _train_and_save_models(model_dir, cache)

        pred = predict_single("069500", cache, clf, reg, features)
        if pred is not None:
            prob_sum = sum(pred.probabilities.values())
            assert abs(prob_sum - 1.0) < 0.01

    def test_missing_ticker_returns_none(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        clf, reg, features = _train_and_save_models(model_dir, cache)

        pred = predict_single("999999", cache, clf, reg, features)
        assert pred is None

    def test_confidence_score_in_range(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        clf, reg, features = _train_and_save_models(model_dir, cache)

        pred = predict_single("069500", cache, clf, reg, features)
        if pred is not None:
            assert 0 <= pred.confidence_score <= 1


class TestLabelNames:
    def test_all_labels_present(self):
        assert 0 in LABEL_NAMES
        assert 1 in LABEL_NAMES
        assert 2 in LABEL_NAMES

    def test_label_values(self):
        assert LABEL_NAMES[0] == "down"
        assert LABEL_NAMES[1] == "flat"
        assert LABEL_NAMES[2] == "up"
