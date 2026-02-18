"""Tests for ml/trainer.py"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS
from ml.trainer import (
    TrainingResult,
    save_models,
    train_classifier,
    train_regressor,
)


def _make_synthetic_matrix(n: int = 500, n_tickers: int = 5) -> pd.DataFrame:
    """Create a synthetic feature matrix for testing.

    Generates realistic-looking feature data with known patterns
    to make classification/regression learnable.
    """
    np.random.seed(42)
    rows_per_ticker = n // n_tickers

    frames = []
    for i in range(n_tickers):
        ticker = f"TICK{i:02d}"
        dates = pd.bdate_range("2024-01-01", periods=rows_per_ticker)

        data = {}
        for col in FEATURE_COLUMNS:
            if "ratio" in col:
                data[col] = np.random.normal(0, 0.05, rows_per_ticker)
            elif "ret" in col:
                data[col] = np.random.normal(0.001, 0.02, rows_per_ticker)
            elif "volatility" in col:
                data[col] = np.abs(np.random.normal(0.02, 0.005, rows_per_ticker))
            elif col == "rsi":
                data[col] = np.random.uniform(20, 80, rows_per_ticker)
            elif col == "bb_percent":
                data[col] = np.random.uniform(0, 1, rows_per_ticker)
            elif col == "bb_bandwidth":
                data[col] = np.random.uniform(0.02, 0.15, rows_per_ticker)
            elif col in ("volume_ratio", "market_volume_ratio", "trading_value_ma20_ratio"):
                data[col] = np.abs(np.random.normal(1.0, 0.3, rows_per_ticker))
            else:
                data[col] = np.random.normal(0, 1, rows_per_ticker)

        df = pd.DataFrame(data, index=dates)
        df["ticker"] = ticker

        # Create labels with some signal from features
        signal = df["ret_1d"] + df["rsi"] / 1000 + np.random.normal(0, 0.02, rows_per_ticker)
        df["fwd_return"] = signal
        df["label"] = 1  # flat
        df.loc[df["fwd_return"] > 0.02, "label"] = 2  # up
        df.loc[df["fwd_return"] < -0.02, "label"] = 0  # down

        frames.append(df)

    return pd.concat(frames, axis=0)


class TestTrainClassifier:
    def test_returns_model_and_result(self):
        matrix = _make_synthetic_matrix()
        model, result = train_classifier(matrix, n_splits=3)
        assert model is not None
        assert isinstance(result, TrainingResult)
        assert result.model_type == "classifier"

    def test_cv_scores(self):
        matrix = _make_synthetic_matrix()
        _, result = train_classifier(matrix, n_splits=3)
        assert len(result.cv_scores) == 3
        assert all(0 <= s <= 1 for s in result.cv_scores)

    def test_accuracy_and_f1(self):
        matrix = _make_synthetic_matrix()
        _, result = train_classifier(matrix, n_splits=3)
        assert result.accuracy is not None
        assert 0 <= result.accuracy <= 1
        assert result.f1_macro is not None
        assert 0 <= result.f1_macro <= 1

    def test_feature_importance(self):
        matrix = _make_synthetic_matrix()
        _, result = train_classifier(matrix, n_splits=3)
        assert len(result.feature_importance) > 0
        assert all(v >= 0 for v in result.feature_importance.values())

    def test_class_distribution(self):
        matrix = _make_synthetic_matrix()
        _, result = train_classifier(matrix, n_splits=3)
        assert result.class_distribution is not None
        total = sum(result.class_distribution.values())
        assert total == len(matrix)

    def test_model_can_predict(self):
        matrix = _make_synthetic_matrix()
        model, _ = train_classifier(matrix, n_splits=3)
        features = [c for c in FEATURE_COLUMNS if c in matrix.columns]
        X_sample = matrix[features].iloc[:5].values
        preds = model.predict(X_sample)
        assert len(preds) == 5
        assert all(p in [0, 1, 2] for p in preds)


class TestTrainRegressor:
    def test_returns_model_and_result(self):
        matrix = _make_synthetic_matrix()
        model, result = train_regressor(matrix, n_splits=3)
        assert model is not None
        assert isinstance(result, TrainingResult)
        assert result.model_type == "regressor"

    def test_cv_scores_negative_mae(self):
        matrix = _make_synthetic_matrix()
        _, result = train_regressor(matrix, n_splits=3)
        assert len(result.cv_scores) == 3
        # Negative MAE should be <= 0
        assert all(s <= 0 for s in result.cv_scores)

    def test_mae(self):
        matrix = _make_synthetic_matrix()
        _, result = train_regressor(matrix, n_splits=3)
        assert result.mae is not None
        assert result.mae >= 0

    def test_model_can_predict(self):
        matrix = _make_synthetic_matrix()
        model, _ = train_regressor(matrix, n_splits=3)
        features = [c for c in FEATURE_COLUMNS if c in matrix.columns]
        X_sample = matrix[features].iloc[:5].values
        preds = model.predict(X_sample)
        assert len(preds) == 5
        # Return predictions should be reasonable (not extreme)
        assert all(-1 < p < 1 for p in preds)


class TestSaveModels:
    def test_saves_files(self, tmp_path):
        import joblib

        matrix = _make_synthetic_matrix(n=200)
        clf, clf_result = train_classifier(matrix, n_splits=2)
        reg, reg_result = train_regressor(matrix, n_splits=2)
        features = [c for c in FEATURE_COLUMNS if c in matrix.columns]

        meta_path = save_models(clf, reg, clf_result, reg_result, features, model_dir=tmp_path)
        assert meta_path.exists()

        # Check files exist
        files = list(tmp_path.glob("*"))
        names = [f.name for f in files]
        assert any("classifier_" in n for n in names)
        assert any("regressor_" in n for n in names)
        assert any("metadata_" in n for n in names)

        # Check metadata content
        metadata = json.loads(meta_path.read_text())
        assert "features" in metadata
        assert "classifier" in metadata
        assert "regressor" in metadata
        assert len(metadata["features"]) == len(features)

        # Check models load correctly
        clf_files = list(tmp_path.glob("classifier_*.joblib"))
        loaded_clf = joblib.load(clf_files[0])
        assert loaded_clf is not None
