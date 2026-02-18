"""
Stock watchlist based on KOSPI200 constituents.

Dynamically fetches KOSPI200 constituents from KRX and groups them
by industry sector. Results are cached in memory for the session.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from pykrx import stock

from config.settings import DATE_FORMAT_KRX, MARKET_BENCHMARK_INDEX

logger = logging.getLogger(__name__)

KRX_API_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.krx.co.kr/",
}

# KRX sector key mapping (Korean -> English key)
SECTOR_KEY_MAP = {
    "음식료품": "food",
    "섬유의복": "textile",
    "종이목재": "paper",
    "화학": "chemical",
    "의약품": "pharma",
    "비금속광물": "mineral",
    "철강금속": "steel",
    "기계": "machinery",
    "전기전자": "electronics",
    "의료정밀": "medical",
    "운수장비": "transport_equip",
    "유통업": "retail",
    "전기가스업": "utilities",
    "건설업": "construction",
    "운수창고업": "logistics",
    "통신업": "telecom",
    "금융업": "finance",
    "은행": "banking",
    "증권": "securities",
    "보험": "insurance",
    "서비스업": "services",
}

SECTOR_EN_MAP = {
    "food": "Food & Beverage",
    "textile": "Textile & Apparel",
    "paper": "Paper & Wood",
    "chemical": "Chemical",
    "pharma": "Pharmaceutical",
    "mineral": "Non-metallic Mineral",
    "steel": "Steel & Metal",
    "machinery": "Machinery",
    "electronics": "Electronics",
    "medical": "Medical Precision",
    "transport_equip": "Transport Equipment",
    "retail": "Retail",
    "utilities": "Utilities",
    "construction": "Construction",
    "logistics": "Logistics",
    "telecom": "Telecom",
    "finance": "Finance",
    "banking": "Banking",
    "securities": "Securities",
    "insurance": "Insurance",
    "services": "Services",
    "others": "Others",
}


@dataclass
class StockCategory:
    name_kr: str
    name_en: str
    stocks: List[Tuple[str, str]] = field(default_factory=list)


# Module-level cache
_watchlist_cache: Optional[Dict[str, StockCategory]] = None


def _fetch_kospi200_constituents() -> List[str]:
    """Fetch KOSPI200 constituent tickers from pykrx."""
    try:
        tickers = stock.get_index_portfolio_deposit_file(MARKET_BENCHMARK_INDEX)
        if tickers is not None and len(tickers) > 0:
            logger.info("Fetched %d KOSPI200 constituents", len(tickers))
            return list(tickers)
    except Exception as e:
        logger.warning("Failed to fetch KOSPI200 constituents via pykrx: %s", e)

    # Fallback: KRX direct API
    try:
        data = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
            "locale": "ko_KR",
            "idx_cd": MARKET_BENCHMARK_INDEX,
        }
        resp = requests.post(KRX_API_URL, headers=KRX_HEADERS, data=data, timeout=10)
        resp.raise_for_status()
        rows = resp.json().get("output", [])
        tickers = []
        for row in rows:
            ticker = row.get("ISU_SRT_CD", "")
            if ticker and ticker.isdigit() and len(ticker) == 6:
                tickers.append(ticker)
        if tickers:
            logger.info("Fetched %d KOSPI200 constituents via KRX API", len(tickers))
            return tickers
    except Exception as e:
        logger.warning("Failed to fetch KOSPI200 constituents via KRX API: %s", e)

    return []


def _fetch_sector_classification() -> Dict[str, str]:
    """Fetch sector classification for all KOSPI stocks from KRX API.

    Returns:
        Dict mapping ticker -> sector name (Korean).
    """
    today_str = date.today().strftime(DATE_FORMAT_KRX)

    try:
        data = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT03901",
            "locale": "ko_KR",
            "mktId": "STK",
            "trdDd": today_str,
        }
        resp = requests.post(KRX_API_URL, headers=KRX_HEADERS, data=data, timeout=10)
        resp.raise_for_status()
        rows = resp.json().get("output", [])

        sector_map: Dict[str, str] = {}
        for row in rows:
            ticker = row.get("ISU_SRT_CD", "")
            sector = row.get("IDX_IND_NM", "")
            if ticker and sector:
                sector_map[ticker] = sector

        if sector_map:
            logger.info("Fetched sector classification for %d stocks", len(sector_map))
            return sector_map
    except Exception as e:
        logger.warning("Failed to fetch sector classification: %s", e)

    return {}


def build_watchlist() -> Dict[str, StockCategory]:
    """Build watchlist by fetching KOSPI200 constituents and grouping by sector.

    Returns:
        Dict mapping sector key -> StockCategory.
    """
    global _watchlist_cache
    if _watchlist_cache is not None:
        return _watchlist_cache

    tickers = _fetch_kospi200_constituents()
    if not tickers:
        logger.error("No KOSPI200 constituents fetched, returning empty watchlist")
        _watchlist_cache = {}
        return _watchlist_cache

    # Get names for all tickers
    name_map: Dict[str, str] = {}
    for ticker in tickers:
        try:
            name = stock.get_market_ticker_name(ticker)
            name_map[ticker] = name if name else ticker
        except Exception:
            name_map[ticker] = ticker

    # Get sector classifications
    sector_map = _fetch_sector_classification()

    # Group by sector
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for ticker in tickers:
        sector_kr = sector_map.get(ticker, "기타")
        grouped.setdefault(sector_kr, []).append(
            (ticker, name_map.get(ticker, ticker))
        )

    # Build watchlist dict
    watchlist: Dict[str, StockCategory] = {}
    for sector_kr, stocks_list in grouped.items():
        key = SECTOR_KEY_MAP.get(sector_kr, "others")
        name_en = SECTOR_EN_MAP.get(key, key.replace("_", " ").title())

        if key in watchlist:
            watchlist[key].stocks.extend(stocks_list)
        else:
            watchlist[key] = StockCategory(
                name_kr=sector_kr,
                name_en=name_en,
                stocks=stocks_list,
            )

    _watchlist_cache = watchlist
    logger.info(
        "Built watchlist: %d sectors, %d stocks",
        len(watchlist),
        sum(len(c.stocks) for c in watchlist.values()),
    )
    return watchlist


def _get_watchlist() -> Dict[str, StockCategory]:
    """Get or build the watchlist (lazy initialization)."""
    return build_watchlist()


def get_all_watchlist_tickers() -> List[str]:
    """Return a flat list of all ticker codes in the watchlist."""
    wl = _get_watchlist()
    tickers = []
    for category in wl.values():
        for ticker, _name in category.stocks:
            tickers.append(ticker)
    return tickers


def get_ticker_name_map() -> Dict[str, str]:
    """Return a dict mapping ticker -> stock name for all watchlist stocks."""
    wl = _get_watchlist()
    mapping = {}
    for category in wl.values():
        for ticker, name in category.stocks:
            mapping[ticker] = name
    return mapping


def get_tickers_by_category(category_key: str) -> List[Tuple[str, str]]:
    """Return list of (ticker, name) for a given category key."""
    wl = _get_watchlist()
    if category_key not in wl:
        raise ValueError(
            f"Unknown category: {category_key}. "
            f"Available: {list(wl.keys())}"
        )
    return wl[category_key].stocks


def get_category_for_ticker(ticker: str) -> str:
    """Return the category key for a given ticker, or 'unknown' if not found."""
    wl = _get_watchlist()
    for key, category in wl.items():
        for t, _name in category.stocks:
            if t == ticker:
                return key
    return "unknown"


def get_tickers_grouped_by_category() -> Dict[str, List[str]]:
    """Return dict mapping category_key -> list of ticker codes."""
    wl = _get_watchlist()
    grouped: Dict[str, List[str]] = {}
    for key, category in wl.items():
        grouped[key] = [ticker for ticker, _name in category.stocks]
    return grouped


def clear_cache() -> None:
    """Clear the cached watchlist data (force re-fetch on next access)."""
    global _watchlist_cache
    _watchlist_cache = None
