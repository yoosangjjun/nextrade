"""
Relative strength comparison.

Compares each stock's returns against a benchmark (KOSPI200 index)
and ranks stocks within their category.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config.watchlist import (
    _get_watchlist,
    get_all_watchlist_tickers,
    get_category_for_ticker,
    get_ticker_name_map,
)
from config.settings import MARKET_BENCHMARK_TICKER
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


@dataclass
class RelativeStrength:
    """Relative strength of a stock vs benchmark."""

    ticker: str
    name: str
    rs_5d: float   # 5d return - benchmark 5d return
    rs_20d: float  # 20d return - benchmark 20d return
    rs_60d: float  # 60d return - benchmark 60d return
    rs_score: float  # Weighted composite: 5d×0.2 + 20d×0.5 + 60d×0.3
    sector_rank: int   # Rank within same category (1 = best)
    sector_size: int   # Number of stocks in same category


def _calc_return(series: pd.Series, periods: int) -> Optional[float]:
    """Calculate percentage return over N periods from end of series."""
    if len(series) < periods + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-1 - periods]) - 1)


def calculate_relative_strength(
    cache: OHLCVCache,
    benchmark_ticker: str = MARKET_BENCHMARK_TICKER,
) -> List[RelativeStrength]:
    """Calculate relative strength for all watchlist stocks vs benchmark.

    For each stock:
    - Computes 5d/20d/60d return minus benchmark return.
    - Computes weighted rs_score = 5d×0.2 + 20d×0.5 + 60d×0.3.
    - Ranks within same category.

    Args:
        cache: OHLCVCache instance.
        benchmark_ticker: Ticker to use as benchmark. Defaults to KOSPI200 index.

    Returns:
        List of RelativeStrength sorted by rs_score descending.
    """
    name_map = get_ticker_name_map()

    # Load benchmark returns
    bench_df = cache.get_ohlcv(benchmark_ticker)
    if bench_df.empty:
        logger.warning("No benchmark data for %s", benchmark_ticker)
        return []

    bench_close = bench_df["close"].dropna()
    bench_5d = _calc_return(bench_close, 5) or 0.0
    bench_20d = _calc_return(bench_close, 20) or 0.0
    bench_60d = _calc_return(bench_close, 60) or 0.0

    tickers = get_all_watchlist_tickers()
    results: List[RelativeStrength] = []

    for ticker in tickers:
        df = cache.get_ohlcv(ticker)
        if df.empty or len(df) < 5:
            continue

        close = df["close"].dropna()
        if close.empty:
            continue

        r5 = _calc_return(close, 5)
        r20 = _calc_return(close, 20)
        r60 = _calc_return(close, 60)

        rs_5d = (r5 - bench_5d) if r5 is not None else 0.0
        rs_20d = (r20 - bench_20d) if r20 is not None else 0.0
        rs_60d = (r60 - bench_60d) if r60 is not None else 0.0

        rs_score = rs_5d * 0.2 + rs_20d * 0.5 + rs_60d * 0.3

        results.append(
            RelativeStrength(
                ticker=ticker,
                name=name_map.get(ticker, ticker),
                rs_5d=rs_5d,
                rs_20d=rs_20d,
                rs_60d=rs_60d,
                rs_score=rs_score,
                sector_rank=0,  # filled below
                sector_size=0,  # filled below
            )
        )

    # Compute sector ranks
    _assign_sector_ranks(results)

    results.sort(key=lambda r: r.rs_score, reverse=True)
    return results


def _assign_sector_ranks(results: List[RelativeStrength]) -> None:
    """Assign sector_rank and sector_size in-place."""
    # Group by category
    by_category: Dict[str, List[RelativeStrength]] = {}
    for rs in results:
        cat = get_category_for_ticker(rs.ticker)
        by_category.setdefault(cat, []).append(rs)

    for cat, members in by_category.items():
        members.sort(key=lambda r: r.rs_score, reverse=True)
        for i, rs in enumerate(members):
            rs.sector_rank = i + 1
            rs.sector_size = len(members)


def get_relative_strength_by_category(
    cache: OHLCVCache,
    category_key: str,
    benchmark_ticker: str = MARKET_BENCHMARK_TICKER,
) -> List[RelativeStrength]:
    """Get relative strength for stocks in a specific category.

    Args:
        cache: OHLCVCache instance.
        category_key: Category key from watchlist.
        benchmark_ticker: Benchmark ticker.

    Returns:
        List of RelativeStrength for the category, sorted by rs_score descending.

    Raises:
        ValueError: If category_key is not in watchlist.
    """
    wl = _get_watchlist()
    if category_key not in wl:
        raise ValueError(
            f"Unknown category: {category_key}. "
            f"Available: {list(wl.keys())}"
        )

    all_rs = calculate_relative_strength(cache, benchmark_ticker)
    category_tickers = {t for t, _n in wl[category_key].stocks}
    return [rs for rs in all_rs if rs.ticker in category_tickers]
