"""Tests for analysis/relative_strength.py"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from analysis.relative_strength import (
    RelativeStrength,
    _calc_return,
    calculate_relative_strength,
    get_relative_strength_by_category,
)


def _make_ohlcv_df(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(trend, 0.01, n)
    close = 10000 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 100000,
    }, index=dates)


class TestCalcReturn:
    def test_positive_return(self):
        series = pd.Series([100.0, 110.0])
        assert abs(_calc_return(series, 1) - 0.1) < 1e-9

    def test_negative_return(self):
        series = pd.Series([100.0, 90.0])
        assert abs(_calc_return(series, 1) - (-0.1)) < 1e-9

    def test_insufficient_data(self):
        series = pd.Series([100.0])
        assert _calc_return(series, 1) is None


class TestCalculateRelativeStrength:
    def test_returns_list(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_relative_strength(cache)
        assert len(results) > 0
        assert all(isinstance(r, RelativeStrength) for r in results)

    def test_sorted_by_rs_score(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_relative_strength(cache)
        scores = [r.rs_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_sector_ranks_assigned(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_relative_strength(cache)
        for r in results:
            assert r.sector_rank >= 1
            assert r.sector_size >= 1
            assert r.sector_rank <= r.sector_size

    def test_rs_score_is_weighted_sum(self):
        """rs_score should equal 0.2*rs_5d + 0.5*rs_20d + 0.3*rs_60d."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_relative_strength(cache)
        for r in results:
            expected = r.rs_5d * 0.2 + r.rs_20d * 0.5 + r.rs_60d * 0.3
            assert abs(r.rs_score - expected) < 1e-9

    def test_empty_benchmark(self):
        """Should return empty list if benchmark has no data."""
        cache = MagicMock()

        def side_effect(ticker):
            if ticker == "IDX_1028":
                return pd.DataFrame()
            return _make_ohlcv_df()

        cache.get_ohlcv.side_effect = side_effect
        results = calculate_relative_strength(cache)
        assert results == []

    def test_benchmark_relative(self):
        """Benchmark itself should have RS near zero."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_relative_strength(cache, benchmark_ticker="IDX_1028")
        bench_results = [r for r in results if r.ticker == "IDX_1028"]
        if bench_results:
            # All stocks get same data, so RS should be ~0
            assert abs(bench_results[0].rs_score) < 1e-9


class TestGetRelativeStrengthByCategory:
    def test_filters_by_category(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = get_relative_strength_by_category(cache, "electronics")
        # All results should be from the electronics category
        from config.watchlist import _get_watchlist
        wl = _get_watchlist()
        electronics_tickers = {t for t, _n in wl["electronics"].stocks}
        for r in results:
            assert r.ticker in electronics_tickers

    def test_invalid_category_raises(self):
        cache = MagicMock()
        try:
            get_relative_strength_by_category(cache, "nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
