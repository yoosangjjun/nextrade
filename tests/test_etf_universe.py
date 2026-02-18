"""Tests for data/universe.py"""

from datetime import date
from unittest.mock import patch

import pytest

from data.universe import get_stock_ticker_list, validate_tickers


class TestStockUniverse:

    @patch("data.universe.stock.get_market_ticker_list")
    def test_get_stock_ticker_list_success(self, mock_get):
        mock_get.return_value = ["069500", "091160", "122630"]
        result = get_stock_ticker_list(date(2025, 1, 6))
        assert len(result) == 3
        assert "069500" in result

    @patch("data.universe.stock.get_market_ticker_list")
    def test_get_stock_ticker_list_retries_on_empty(self, mock_get):
        mock_get.side_effect = [[], ["069500"]]
        with patch("data.universe.time.sleep"):
            result = get_stock_ticker_list(date(2025, 1, 6))
        assert result == ["069500"]

    @patch("data.universe.stock.get_market_ticker_list")
    def test_get_stock_ticker_list_all_fail(self, mock_get):
        mock_get.side_effect = Exception("Down")
        with patch("data.universe.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                get_stock_ticker_list(date(2025, 1, 6))

    @patch("data.universe.get_stock_ticker_list")
    def test_validate_tickers(self, mock_list):
        mock_list.return_value = ["069500", "091160"]
        result = validate_tickers(["069500", "999999"])
        assert result == ["069500"]
