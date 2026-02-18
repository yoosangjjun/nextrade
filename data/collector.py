"""
Stock OHLCV data collector using pykrx.

Handles fetching, incremental collection, rate limiting, and retry logic.
Supports both individual stock OHLCV and market index OHLCV.
"""

import logging
import time
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from pykrx import stock

from config.settings import (
    COLLECTION_DELAY_SEC,
    DATE_FORMAT_KRX,
    DEFAULT_LOOKBACK_DAYS,
    INDEX_OHLCV_COLUMN_MAP,
    MARKET_BENCHMARK_INDEX,
    MARKET_BENCHMARK_TICKER,
    MAX_RETRIES,
    OHLCV_COLUMN_MAP,
    RETRY_DELAY_SEC,
)
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


def _is_index_ticker(ticker: str) -> bool:
    """Check if a ticker is an internal index ticker (e.g. IDX_1028)."""
    return ticker.startswith("IDX_")


def _get_index_code(ticker: str) -> str:
    """Extract index code from internal index ticker (IDX_1028 -> 1028)."""
    return ticker.replace("IDX_", "")


class StockCollector:
    """Collects stock OHLCV data from KRX and stores it in the local cache."""

    def __init__(self, cache: Optional[OHLCVCache] = None):
        self.cache = cache or OHLCVCache()

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single stock from pykrx with retry logic.

        Returns:
            DataFrame with English column names and DatetimeIndex.
            Empty DataFrame on failure.
        """
        start_str = start_date.strftime(DATE_FORMAT_KRX)
        end_str = end_date.strftime(DATE_FORMAT_KRX)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if _is_index_ticker(ticker):
                    index_code = _get_index_code(ticker)
                    df = stock.get_index_ohlcv_by_date(
                        start_str, end_str, index_code
                    )
                    col_map = INDEX_OHLCV_COLUMN_MAP
                else:
                    df = stock.get_market_ohlcv_by_date(
                        start_str, end_str, ticker
                    )
                    col_map = OHLCV_COLUMN_MAP

                if df is None or df.empty:
                    logger.debug(
                        "No data for %s (%s ~ %s)", ticker, start_str, end_str
                    )
                    return pd.DataFrame()

                # Rename Korean columns to English
                df = df.rename(columns=col_map)

                # Keep only expected columns
                expected_cols = list(col_map.values())
                available_cols = [c for c in expected_cols if c in df.columns]
                df = df[available_cols]

                logger.debug(
                    "Fetched %d rows for %s (%s ~ %s)",
                    len(df),
                    ticker,
                    start_str,
                    end_str,
                )
                return df

            except Exception as e:
                logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s",
                    ticker,
                    attempt,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)

        logger.error("All retries failed for %s", ticker)
        return pd.DataFrame()

    def collect_ticker(
        self,
        ticker: str,
        end_date: Optional[date] = None,
        force_full: bool = False,
    ) -> Tuple[int, bool]:
        """
        Collect data for a single ticker with incremental update.

        Returns:
            Tuple of (rows_added, success).
        """
        if end_date is None:
            end_date = date.today()

        if force_full:
            start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        else:
            last_cached = self.cache.get_last_cached_date(ticker)
            if last_cached:
                start_date = last_cached + timedelta(days=1)
                if start_date > end_date:
                    logger.debug("Cache is up-to-date for %s", ticker)
                    return (0, True)
            else:
                start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        df = self.fetch_ohlcv(ticker, start_date, end_date)

        if df.empty:
            return (0, True)

        count = self.cache.upsert_ohlcv(ticker, df)
        return (count, True)

    def collect_benchmark(
        self,
        end_date: Optional[date] = None,
        force_full: bool = False,
    ) -> Tuple[int, bool]:
        """Collect benchmark index data (KOSPI200).

        Returns:
            Tuple of (rows_added, success).
        """
        return self.collect_ticker(
            MARKET_BENCHMARK_TICKER, end_date, force_full
        )

    def collect_multiple(
        self,
        tickers: List[str],
        end_date: Optional[date] = None,
        force_full: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Collect data for multiple tickers sequentially with rate limiting.

        Automatically includes benchmark index data collection.

        Args:
            progress_callback: Optional callable(ticker, index, total, rows_added)

        Returns:
            Dict with collection summary.
        """
        # Always collect benchmark first
        bench_rows, bench_ok = self.collect_benchmark(end_date, force_full)
        if bench_ok:
            logger.info(
                "Benchmark %s: %d rows collected",
                MARKET_BENCHMARK_TICKER,
                bench_rows,
            )

        results = {
            "total_tickers": len(tickers),
            "success_count": 0,
            "fail_count": 0,
            "total_rows": bench_rows,
            "skipped_count": 0,
            "failed_tickers": [],
        }

        for i, ticker in enumerate(tickers):
            try:
                rows_added, success = self.collect_ticker(
                    ticker, end_date, force_full
                )

                if success:
                    results["success_count"] += 1
                    results["total_rows"] += rows_added
                    if rows_added == 0:
                        results["skipped_count"] += 1
                else:
                    results["fail_count"] += 1
                    results["failed_tickers"].append(ticker)

                if progress_callback:
                    progress_callback(ticker, i + 1, len(tickers), rows_added)

            except Exception as e:
                logger.error("Unexpected error collecting %s: %s", ticker, e)
                results["fail_count"] += 1
                results["failed_tickers"].append(ticker)

            # Rate limiting between API calls
            if i < len(tickers) - 1:
                time.sleep(COLLECTION_DELAY_SEC)

        return results
