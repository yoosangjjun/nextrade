"""Tests for analysis/indicators.py"""

import pandas as pd
import numpy as np
import pytest

from analysis.indicators import (
    add_moving_averages,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_volume_ma,
    calculate_all_indicators,
)


def _make_ohlcv(n: int = 200, base_price: float = 10000) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame with realistic price data."""
    np.random.seed(42)
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
        },
        index=dates,
    )


class TestMovingAverages:
    def test_adds_ma_columns(self):
        df = _make_ohlcv()
        result = add_moving_averages(df)
        for period in [5, 20, 60, 120]:
            assert f"ma{period}" in result.columns

    def test_ma5_values(self):
        df = _make_ohlcv()
        result = add_moving_averages(df)
        # MA5 at index 4 should equal mean of first 5 closes
        expected = df["close"].iloc[:5].mean()
        assert abs(result["ma5"].iloc[4] - expected) < 0.01

    def test_ma120_has_nans_initially(self):
        df = _make_ohlcv()
        result = add_moving_averages(df)
        assert pd.isna(result["ma120"].iloc[0])
        assert not pd.isna(result["ma120"].iloc[119])


class TestRSI:
    def test_adds_rsi_column(self):
        df = _make_ohlcv()
        result = add_rsi(df)
        assert "rsi" in result.columns

    def test_rsi_range(self):
        df = _make_ohlcv()
        result = add_rsi(df)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestMACD:
    def test_adds_macd_columns(self):
        df = _make_ohlcv()
        result = add_macd(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns


class TestBollingerBands:
    def test_adds_bb_columns(self):
        df = _make_ohlcv()
        result = add_bollinger_bands(df)
        assert "bb_lower" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_upper" in result.columns

    def test_upper_above_lower(self):
        df = _make_ohlcv()
        result = add_bollinger_bands(df)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


class TestVolumeMA:
    def test_adds_volume_columns(self):
        df = _make_ohlcv()
        result = add_volume_ma(df)
        assert "volume_ma20" in result.columns
        assert "volume_ratio" in result.columns


class TestCalculateAll:
    def test_all_indicators_added(self):
        df = _make_ohlcv()
        result = calculate_all_indicators(df)
        expected_cols = [
            "ma5", "ma20", "ma60", "ma120",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_lower", "bb_middle", "bb_upper",
            "volume_ma20", "volume_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = calculate_all_indicators(df)
        assert result.empty

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_all_indicators(df)

    def test_does_not_modify_original(self):
        df = _make_ohlcv()
        original_cols = set(df.columns)
        calculate_all_indicators(df)
        assert set(df.columns) == original_cols
