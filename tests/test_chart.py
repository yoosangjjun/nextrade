"""Tests for notification/chart.py"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from notification.chart import generate_price_chart


def _make_ohlcv_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(0.001, 0.02, n)
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


class TestGeneratePriceChart:
    def test_returns_png_bytes(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(200)

        result = generate_price_chart("069500", cache, name="KODEX 200")
        assert result is not None
        assert isinstance(result, bytes)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_insufficient_data_returns_none(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(30)

        result = generate_price_chart("069500", cache)
        assert result is None

    def test_empty_data_returns_none(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = pd.DataFrame()

        result = generate_price_chart("069500", cache)
        assert result is None

    def test_custom_days(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(200)

        result = generate_price_chart("069500", cache, days=30)
        assert result is not None
        assert isinstance(result, bytes)

    def test_no_name(self):
        cache = MagicMock()
        cache.get_ohlcv.return_value = _make_ohlcv_df(200)

        result = generate_price_chart("069500", cache)
        assert result is not None
