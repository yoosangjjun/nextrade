"""
Sector rotation analysis.

Calculates momentum (5d/20d/60d returns) per category,
ranks categories, and tracks rank changes over time.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config.watchlist import (
    _get_watchlist,
    get_ticker_name_map,
    get_tickers_grouped_by_category,
)
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


@dataclass
class SectorMomentum:
    """Momentum analysis for a single category (sector)."""

    category: str
    name_kr: str
    momentum_5d: float
    momentum_20d: float
    momentum_60d: float
    rank_5d: int
    rank_20d: int
    rank_60d: int
    rank_change_20d: int  # positive = improved
    top_stock: str  # Name of the best-performing stock in this category


def _calc_return(series: pd.Series, periods: int) -> Optional[float]:
    """Calculate percentage return over N periods from the end of a series."""
    if len(series) < periods + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-1 - periods]) - 1)


def _calc_return_at_offset(series: pd.Series, periods: int, offset: int) -> Optional[float]:
    """Calculate return ending at `offset` rows from the end."""
    end_idx = len(series) - 1 - offset
    start_idx = end_idx - periods
    if start_idx < 0 or end_idx < 0:
        return None
    return float((series.iloc[end_idx] / series.iloc[start_idx]) - 1)


def calculate_sector_momentum(cache: OHLCVCache) -> List[SectorMomentum]:
    """Calculate momentum for each watchlist category.

    For each category:
    - Computes the average close-price return (5d/20d/60d) across member stocks.
    - Ranks categories by each momentum period.
    - Tracks 20d rank change (current rank vs rank 20 trading days ago).
    - Identifies the top-performing ETF within the category.

    Args:
        cache: OHLCVCache instance with cached OHLCV data.

    Returns:
        List of SectorMomentum sorted by 20d momentum descending.
    """
    grouped = get_tickers_grouped_by_category()
    name_map = get_ticker_name_map()

    # Collect category-level average returns
    cat_data: Dict[str, dict] = {}

    wl = _get_watchlist()

    for cat_key, tickers in grouped.items():
        cat_info = wl[cat_key]
        returns_5d = []
        returns_20d = []
        returns_60d = []
        prev_returns_20d = []  # 20 days ago snapshot for rank change
        stock_momentum: Dict[str, float] = {}  # ticker -> 20d return

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

            if r5 is not None:
                returns_5d.append(r5)
            if r20 is not None:
                returns_20d.append(r20)
                stock_momentum[ticker] = r20
            if r60 is not None:
                returns_60d.append(r60)

            # Previous 20d return (ending 20 days ago) for rank change tracking
            prev_r20 = _calc_return_at_offset(close, 20, 20)
            if prev_r20 is not None:
                prev_returns_20d.append(prev_r20)

        # Skip categories with no valid return data at all
        if not returns_5d and not returns_20d and not returns_60d:
            continue

        avg_5d = sum(returns_5d) / len(returns_5d) if returns_5d else 0.0
        avg_20d = sum(returns_20d) / len(returns_20d) if returns_20d else 0.0
        avg_60d = sum(returns_60d) / len(returns_60d) if returns_60d else 0.0
        prev_avg_20d = (
            sum(prev_returns_20d) / len(prev_returns_20d) if prev_returns_20d else None
        )

        # Best stock in category by 20d return
        if stock_momentum:
            best_ticker = max(stock_momentum, key=stock_momentum.get)  # type: ignore[arg-type]
            top_stock = name_map.get(best_ticker, best_ticker)
        else:
            top_stock = "-"

        cat_data[cat_key] = {
            "name_kr": cat_info.name_kr,
            "avg_5d": avg_5d,
            "avg_20d": avg_20d,
            "avg_60d": avg_60d,
            "prev_avg_20d": prev_avg_20d,
            "top_stock": top_stock,
        }

    if not cat_data:
        return []

    # Rank by each momentum period (1 = best)
    sorted_5d = sorted(cat_data.keys(), key=lambda k: cat_data[k]["avg_5d"], reverse=True)
    sorted_20d = sorted(cat_data.keys(), key=lambda k: cat_data[k]["avg_20d"], reverse=True)
    sorted_60d = sorted(cat_data.keys(), key=lambda k: cat_data[k]["avg_60d"], reverse=True)

    rank_5d = {k: i + 1 for i, k in enumerate(sorted_5d)}
    rank_20d = {k: i + 1 for i, k in enumerate(sorted_20d)}
    rank_60d = {k: i + 1 for i, k in enumerate(sorted_60d)}

    # Previous 20d ranks for rank change
    cats_with_prev = {
        k: v for k, v in cat_data.items() if v["prev_avg_20d"] is not None
    }
    if cats_with_prev:
        sorted_prev_20d = sorted(
            cats_with_prev.keys(),
            key=lambda k: cats_with_prev[k]["prev_avg_20d"],
            reverse=True,
        )
        prev_rank_20d = {k: i + 1 for i, k in enumerate(sorted_prev_20d)}
    else:
        prev_rank_20d = {}

    results = []
    for cat_key, data in cat_data.items():
        current_rank = rank_20d[cat_key]
        prev_rank = prev_rank_20d.get(cat_key)
        rank_change = (prev_rank - current_rank) if prev_rank is not None else 0

        results.append(
            SectorMomentum(
                category=cat_key,
                name_kr=data["name_kr"],
                momentum_5d=data["avg_5d"],
                momentum_20d=data["avg_20d"],
                momentum_60d=data["avg_60d"],
                rank_5d=rank_5d[cat_key],
                rank_20d=current_rank,
                rank_60d=rank_60d[cat_key],
                rank_change_20d=rank_change,
                top_stock=data["top_stock"],
            )
        )

    results.sort(key=lambda s: s.momentum_20d, reverse=True)
    return results


def get_sector_momentum_detail(
    cache: OHLCVCache, category_key: str
) -> List[Dict]:
    """Get per-stock momentum detail for a specific category.

    Args:
        cache: OHLCVCache instance.
        category_key: Category key from watchlist (e.g. "electronics").

    Returns:
        List of dicts with keys: ticker, name, momentum_5d, momentum_20d,
        momentum_60d, rank (within category, by 20d momentum).

    Raises:
        ValueError: If category_key is not in watchlist.
    """
    wl = _get_watchlist()
    if category_key not in wl:
        raise ValueError(
            f"Unknown category: {category_key}. "
            f"Available: {list(wl.keys())}"
        )

    stocks = wl[category_key].stocks
    name_map = get_ticker_name_map()
    details = []

    for ticker, _name in stocks:
        df = cache.get_ohlcv(ticker)
        if df.empty or len(df) < 5:
            continue

        close = df["close"].dropna()
        if close.empty:
            continue

        r5 = _calc_return(close, 5) or 0.0
        r20 = _calc_return(close, 20) or 0.0
        r60 = _calc_return(close, 60) or 0.0

        details.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "momentum_5d": r5,
            "momentum_20d": r20,
            "momentum_60d": r60,
        })

    # Rank within category by 20d momentum
    details.sort(key=lambda d: d["momentum_20d"], reverse=True)
    for i, d in enumerate(details):
        d["rank"] = i + 1

    return details
