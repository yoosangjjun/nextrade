"""Tests for ml/features.py"""

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_COLUMNS,
    add_derived_features,
    add_labels,
    add_market_features,
    add_return_features,
    add_sector_features,
    build_feature_matrix,
)


def _make_ohlcv(n: int = 200, base_price: float = 10000, seed: int = 42) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame with realistic price data."""
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


def _make_ohlcv_with_indicators(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create OHLCV data with indicator columns pre-computed."""
    from analysis.indicators import calculate_all_indicators

    df = _make_ohlcv(n=n, seed=seed)
    return calculate_all_indicators(df)


class TestReturnFeatures:
    def test_adds_return_columns(self):
        df = _make_ohlcv()
        result = add_return_features(df)
        assert "ret_1d" in result.columns
        assert "ret_5d" in result.columns
        assert "ret_20d" in result.columns
        assert "volatility_20d" in result.columns

    def test_ret_1d_values(self):
        df = _make_ohlcv()
        result = add_return_features(df)
        # First row should be NaN
        assert pd.isna(result["ret_1d"].iloc[0])
        # Subsequent rows should be close-to-close pct change
        expected = (df["close"].iloc[1] - df["close"].iloc[0]) / df["close"].iloc[0]
        assert abs(result["ret_1d"].iloc[1] - expected) < 1e-10

    def test_volatility_starts_nan(self):
        df = _make_ohlcv()
        result = add_return_features(df)
        assert pd.isna(result["volatility_20d"].iloc[10])
        # Should have values after 20+ rows
        assert not pd.isna(result["volatility_20d"].iloc[25])


class TestMarketFeatures:
    def test_adds_market_columns(self):
        df = _make_ohlcv(seed=42)
        df = add_return_features(df)
        market_df = _make_ohlcv(seed=99)
        result = add_market_features(df, market_df)
        assert "market_ret_1d" in result.columns
        assert "market_ret_5d" in result.columns
        assert "market_volume_ratio" in result.columns
        assert "relative_ret_1d" in result.columns

    def test_relative_returns(self):
        df = _make_ohlcv(seed=42)
        df = add_return_features(df)
        market_df = _make_ohlcv(seed=99)
        result = add_market_features(df, market_df)
        # relative_ret = ret - market_ret
        valid = result.dropna(subset=["relative_ret_1d", "ret_1d", "market_ret_1d"])
        if len(valid) > 0:
            diff = valid["relative_ret_1d"] - (valid["ret_1d"] - valid["market_ret_1d"])
            assert (diff.abs() < 1e-10).all()


class TestSectorFeatures:
    def test_adds_sector_columns(self):
        df = _make_ohlcv(seed=42)
        df = add_return_features(df)

        # Create mock ticker returns
        peer_df = _make_ohlcv(seed=99)
        all_returns = {
            "091170": pd.DataFrame(
                {"ret_1d": peer_df["close"].pct_change(1), "ret_5d": peer_df["close"].pct_change(5)},
                index=peer_df.index,
            )
        }
        result = add_sector_features(df, "091160", all_returns)  # 091160 is sector category
        assert "sector_avg_ret_1d" in result.columns
        assert "sector_avg_ret_5d" in result.columns
        assert "relative_sector_ret_1d" in result.columns

    def test_no_peers_gives_zeros(self):
        df = _make_ohlcv(seed=42)
        df = add_return_features(df)
        result = add_sector_features(df, "UNKNOWN", {})
        assert (result["sector_avg_ret_1d"] == 0.0).all()


class TestDerivedFeatures:
    def test_adds_ratio_columns(self):
        df = _make_ohlcv_with_indicators()
        df = add_return_features(df)
        result = add_derived_features(df)
        assert "close_ma5_ratio" in result.columns
        assert "close_ma20_ratio" in result.columns
        assert "ma5_ma20_ratio" in result.columns
        assert "ma20_ma60_ratio" in result.columns
        assert "trading_value_ma20_ratio" in result.columns

    def test_ratio_values_reasonable(self):
        df = _make_ohlcv_with_indicators()
        df = add_return_features(df)
        result = add_derived_features(df)
        valid = result["close_ma5_ratio"].dropna()
        # Ratio should be close to 0 (since price ~ MA)
        assert valid.abs().max() < 0.5  # Within 50% of MA


class TestLabels:
    def test_adds_label_columns(self):
        df = _make_ohlcv()
        result = add_labels(df)
        assert "fwd_return" in result.columns
        assert "label" in result.columns

    def test_label_values(self):
        df = _make_ohlcv()
        result = add_labels(df, horizon=5, threshold=0.02)
        valid = result.dropna(subset=["fwd_return"])
        assert set(valid["label"].unique()).issubset({0, 1, 2})

    def test_label_consistency(self):
        df = _make_ohlcv()
        result = add_labels(df, threshold=0.02)
        valid = result.dropna(subset=["fwd_return"])
        # Up labels should have positive returns > threshold
        up = valid[valid["label"] == 2]
        if len(up) > 0:
            assert (up["fwd_return"] > 0.02).all()
        # Down labels should have negative returns < -threshold
        down = valid[valid["label"] == 0]
        if len(down) > 0:
            assert (down["fwd_return"] < -0.02).all()

    def test_last_rows_have_nan_fwd_return(self):
        df = _make_ohlcv()
        result = add_labels(df, horizon=5)
        # Last 5 rows should have NaN fwd_return
        assert pd.isna(result["fwd_return"].iloc[-1])
        assert pd.isna(result["fwd_return"].iloc[-5])


class TestBuildFeatureMatrix:
    def _insert_benchmark(self, cache):
        """Insert benchmark (IDX_1028) data into cache."""
        from config.settings import MARKET_BENCHMARK_TICKER

        benchmark_df = _make_ohlcv(n=200, seed=99)
        cache.upsert_ohlcv(MARKET_BENCHMARK_TICKER, benchmark_df)

    def test_with_mock_cache(self, tmp_path):
        """Test build_feature_matrix with a temporary cache."""
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)

        # Insert benchmark and stock data
        self._insert_benchmark(cache)
        for i, ticker in enumerate(["005930", "000660", "066570"]):
            df = _make_ohlcv(n=200, seed=42 + i)
            cache.upsert_ohlcv(ticker, df)

        matrix = build_feature_matrix(
            cache, tickers=["005930", "000660", "066570"], add_label=True
        )

        assert not matrix.empty
        assert "label" in matrix.columns
        assert "ticker" in matrix.columns
        assert matrix["ticker"].nunique() >= 1

    def test_prediction_mode_no_labels(self, tmp_path):
        """Test build_feature_matrix in prediction mode (no labels)."""
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)

        self._insert_benchmark(cache)
        df = _make_ohlcv(n=200, seed=42)
        cache.upsert_ohlcv("005930", df)

        matrix = build_feature_matrix(
            cache, tickers=["005930"], add_label=False
        )

        assert not matrix.empty
        assert "label" not in matrix.columns
        assert "fwd_return" not in matrix.columns

    def test_empty_cache_returns_empty(self, tmp_path):
        from data.cache import OHLCVCache

        db_path = tmp_path / "test.db"
        cache = OHLCVCache(db_path=db_path)
        matrix = build_feature_matrix(cache, tickers=["999999"])
        assert matrix.empty


class TestFeatureColumns:
    def test_feature_columns_nonempty(self):
        assert len(FEATURE_COLUMNS) > 20

    def test_no_duplicate_features(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
