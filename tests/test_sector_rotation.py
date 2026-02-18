"""Tests for analysis/sector_rotation.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from analysis.sector_rotation import (
    SectorMomentum,
    _calc_return,
    _calc_return_at_offset,
    calculate_sector_momentum,
    get_sector_momentum_detail,
)


def _make_close_series(n: int = 100, base: float = 10000, trend: float = 0.001) -> pd.Series:
    """Create a synthetic close price series with a given trend."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(trend, 0.01, n)
    close = base * np.cumprod(1 + returns)
    return pd.Series(close, index=dates, name="close")


def _make_ohlcv_df(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    close = _make_close_series(n, trend=trend)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 100000,
    }, index=close.index)


class TestCalcReturn:
    def test_basic_return(self):
        series = pd.Series([100.0, 110.0])
        ret = _calc_return(series, 1)
        assert ret is not None
        assert abs(ret - 0.1) < 1e-9

    def test_insufficient_data(self):
        series = pd.Series([100.0])
        ret = _calc_return(series, 1)
        assert ret is None

    def test_multi_period(self):
        series = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
        ret = _calc_return(series, 5)
        assert ret is not None
        assert abs(ret - 0.25) < 1e-9


class TestCalcReturnAtOffset:
    def test_basic_offset(self):
        # [100, 110, 120, 130, 140]
        series = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0])
        # offset=1: end at index 3 (130), periods=2: start at index 1 (110)
        ret = _calc_return_at_offset(series, 2, 1)
        assert ret is not None
        expected = (130.0 / 110.0) - 1
        assert abs(ret - expected) < 1e-9

    def test_insufficient_data(self):
        series = pd.Series([100.0, 110.0])
        ret = _calc_return_at_offset(series, 5, 0)
        assert ret is None


class TestCalculateSectorMomentum:
    def test_returns_all_categories(self):
        """Should return momentum for each category that has data."""
        cache = MagicMock()

        # Return synthetic data for any ticker
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_sector_momentum(cache)
        assert len(results) > 0
        assert all(isinstance(r, SectorMomentum) for r in results)

    def test_sorted_by_20d_momentum(self):
        """Results should be sorted by momentum_20d descending."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_sector_momentum(cache)
        momentums = [r.momentum_20d for r in results]
        assert momentums == sorted(momentums, reverse=True)

    def test_ranks_are_valid(self):
        """Each rank should be between 1 and total categories."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_sector_momentum(cache)
        n = len(results)
        for r in results:
            assert 1 <= r.rank_5d <= n
            assert 1 <= r.rank_20d <= n
            assert 1 <= r.rank_60d <= n

    def test_empty_cache_returns_empty(self):
        """Should return empty list if no OHLCV data."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = pd.DataFrame()

        results = calculate_sector_momentum(cache)
        assert results == []

    def test_rank_change_computed(self):
        """rank_change_20d should be an integer."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_sector_momentum(cache)
        for r in results:
            assert isinstance(r.rank_change_20d, int)

    def test_top_stock_not_empty(self):
        """top_stock should be populated when data is available."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        results = calculate_sector_momentum(cache)
        for r in results:
            assert r.top_stock != ""


class TestGetSectorMomentumDetail:
    def test_returns_detail_list(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        details = get_sector_momentum_detail(cache, "electronics")
        assert len(details) > 0
        assert "ticker" in details[0]
        assert "name" in details[0]
        assert "momentum_20d" in details[0]
        assert "rank" in details[0]

    def test_ranked_within_category(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=100)

        details = get_sector_momentum_detail(cache, "electronics")
        ranks = [d["rank"] for d in details]
        assert ranks == list(range(1, len(details) + 1))

    def test_invalid_category_raises(self):
        cache = MagicMock()
        try:
            get_sector_momentum_detail(cache, "nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_empty_cache(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = pd.DataFrame()

        details = get_sector_momentum_detail(cache, "electronics")
        assert details == []
