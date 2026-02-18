"""Tests for data/cache.py"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.cache import OHLCVCache


class TestOHLCVCache:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.cache = OHLCVCache(db_path=self.db_path)

    def _make_sample_df(self, dates, close_prices):
        data = {
            "open": close_prices,
            "high": [p * 1.01 for p in close_prices],
            "low": [p * 0.99 for p in close_prices],
            "close": close_prices,
            "volume": [1000] * len(close_prices),
            "trading_value": [10000.0] * len(close_prices),
        }
        index = pd.to_datetime(dates)
        return pd.DataFrame(data, index=index)

    def test_upsert_and_retrieve(self):
        df = self._make_sample_df(["2025-01-06", "2025-01-07"], [10000, 10100])
        rows = self.cache.upsert_ohlcv("069500", df)
        assert rows == 2

        result = self.cache.get_ohlcv("069500")
        assert len(result) == 2
        assert result.iloc[0]["close"] == 10000

    def test_get_last_cached_date(self):
        df = self._make_sample_df(
            ["2025-01-06", "2025-01-07", "2025-01-08"],
            [10000, 10100, 10200],
        )
        self.cache.upsert_ohlcv("069500", df)

        last = self.cache.get_last_cached_date("069500")
        assert last == date(2025, 1, 8)

    def test_get_last_cached_date_empty(self):
        last = self.cache.get_last_cached_date("999999")
        assert last is None

    def test_date_range_query(self):
        df = self._make_sample_df(
            ["2025-01-06", "2025-01-07", "2025-01-08"],
            [10000, 10100, 10200],
        )
        self.cache.upsert_ohlcv("069500", df)

        result = self.cache.get_ohlcv(
            "069500",
            start_date=date(2025, 1, 7),
            end_date=date(2025, 1, 7),
        )
        assert len(result) == 1
        assert result.iloc[0]["close"] == 10100

    def test_upsert_overwrites_existing(self):
        df1 = self._make_sample_df(["2025-01-06"], [10000])
        self.cache.upsert_ohlcv("069500", df1)

        df2 = self._make_sample_df(["2025-01-06"], [99999])
        self.cache.upsert_ohlcv("069500", df2)

        result = self.cache.get_ohlcv("069500")
        assert len(result) == 1
        assert result.iloc[0]["close"] == 99999

    def test_cache_stats(self):
        df = self._make_sample_df(["2025-01-06", "2025-01-07"], [10000, 10100])
        self.cache.upsert_ohlcv("069500", df)
        self.cache.upsert_ohlcv("091160", df)

        stats = self.cache.get_cache_stats()
        assert stats["ticker_count"] == 2
        assert stats["total_rows"] == 4

    def test_get_cached_tickers(self):
        df = self._make_sample_df(["2025-01-06"], [10000])
        self.cache.upsert_ohlcv("069500", df)
        self.cache.upsert_ohlcv("091160", df)

        tickers = self.cache.get_cached_tickers()
        assert tickers == ["069500", "091160"]

    def test_empty_dataframe_upsert(self):
        count = self.cache.upsert_ohlcv("069500", pd.DataFrame())
        assert count == 0
