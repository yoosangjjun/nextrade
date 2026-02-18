"""Tests for screening/screener.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from screening.screener import ScreeningResult, _normalize_score, screen_all


def _make_ohlcv_df(n: int = 200, trend: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with enough rows for indicators."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(trend, 0.02, n)
    close = 10000 * np.cumprod(1 + returns)
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


class TestNormalizeScore:
    def test_mid_value(self):
        assert abs(_normalize_score(50, 0, 100) - 50.0) < 1e-9

    def test_min_value(self):
        assert abs(_normalize_score(0, 0, 100) - 0.0) < 1e-9

    def test_max_value(self):
        assert abs(_normalize_score(100, 0, 100) - 100.0) < 1e-9

    def test_clamp_above_max(self):
        assert abs(_normalize_score(150, 0, 100) - 100.0) < 1e-9

    def test_clamp_below_min(self):
        assert abs(_normalize_score(-50, 0, 100) - 0.0) < 1e-9

    def test_equal_min_max(self):
        assert abs(_normalize_score(42, 42, 42) - 50.0) < 1e-9

    def test_negative_range(self):
        result = _normalize_score(0, -100, 100)
        assert abs(result - 50.0) < 1e-9


class TestScreenAll:
    def test_returns_screening_results(self):
        """screen_all should return a list of ScreeningResult."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        assert len(results) > 0
        assert all(isinstance(r, ScreeningResult) for r in results)

    def test_sorted_by_composite_score(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        ranks = [r.composite_rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_no_ml_fallback_weights(self):
        """Without ML models, should use fallback weights (tech×0.4 + sector×0.3 + rs×0.3)."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        # All results should have no ML data
        for r in results:
            assert r.ml_predicted_return is None
            assert r.ml_label is None

    def test_technical_score_range(self):
        """Technical scores should be normalized to 0~100."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        for r in results:
            assert 0.0 <= r.technical_score <= 100.0

    def test_composite_score_range(self):
        """Composite score should be between 0 and 100."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=200)

        results = screen_all(cache)
        for r in results:
            assert 0.0 <= r.composite_score <= 100.0

    def test_empty_cache(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = pd.DataFrame()

        results = screen_all(cache)
        assert results == []

    def test_insufficient_data(self):
        """Should skip tickers with < 120 rows."""
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(n=50)

        results = screen_all(cache)
        assert results == []
