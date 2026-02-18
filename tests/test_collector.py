"""Tests for data/collector.py"""

from datetime import date
from unittest.mock import patch

import pandas as pd

from data.collector import StockCollector


class TestStockCollector:

    @patch("data.collector.stock.get_market_ohlcv_by_date")
    def test_fetch_ohlcv_renames_columns(self, mock_fetch):
        mock_df = pd.DataFrame(
            {
                "시가": [10000],
                "고가": [10100],
                "저가": [9900],
                "종가": [10050],
                "거래량": [5000],
                "거래대금": [50000000],
            },
            index=pd.to_datetime(["2025-01-06"]),
        )
        mock_fetch.return_value = mock_df

        collector = StockCollector()
        result = collector.fetch_ohlcv("069500", date(2025, 1, 6), date(2025, 1, 6))

        assert "close" in result.columns
        assert "종가" not in result.columns
        assert result.iloc[0]["close"] == 10050

    @patch("data.collector.stock.get_market_ohlcv_by_date")
    def test_fetch_ohlcv_retry_on_failure(self, mock_fetch):
        mock_fetch.side_effect = [
            Exception("Network error"),
            pd.DataFrame(
                {
                    "시가": [100],
                    "고가": [100],
                    "저가": [100],
                    "종가": [100],
                    "거래량": [100],
                    "거래대금": [100],
                },
                index=pd.to_datetime(["2025-01-06"]),
            ),
        ]

        collector = StockCollector()
        with patch("data.collector.time.sleep"):
            result = collector.fetch_ohlcv(
                "069500", date(2025, 1, 6), date(2025, 1, 6)
            )

        assert not result.empty

    @patch("data.collector.stock.get_market_ohlcv_by_date")
    def test_fetch_ohlcv_all_retries_fail(self, mock_fetch):
        mock_fetch.side_effect = Exception("Persistent failure")

        collector = StockCollector()
        with patch("data.collector.time.sleep"):
            result = collector.fetch_ohlcv(
                "069500", date(2025, 1, 6), date(2025, 1, 6)
            )

        assert result.empty
