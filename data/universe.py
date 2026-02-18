"""
Stock Universe Management.

Fetches the complete list of stocks currently listed on KRX
and provides ticker validation/name lookup.
"""

import logging
import time
from datetime import date
from typing import List, Optional

import pandas as pd
import requests
from pykrx import stock

from config.settings import DATE_FORMAT_KRX, MAX_RETRIES, RETRY_DELAY_SEC

logger = logging.getLogger(__name__)

KRX_API_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.krx.co.kr/",
}


def _fetch_stock_tickers_direct(market: str = "STK") -> List[str]:
    """Fetch stock tickers directly from KRX API.

    Args:
        market: "STK" for KOSPI, "KSQ" for KOSDAQ, "ALL" for both.
    """
    data = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT01901",
        "locale": "ko_KR",
        "mktId": market,
    }
    resp = requests.post(KRX_API_URL, headers=KRX_HEADERS, data=data, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    rows = result.get("output", [])
    if not rows:
        return []
    df = pd.DataFrame(rows)
    ticker_col = next(
        (c for c in df.columns if "ISU_SRT_CD" in c or "종목코드" in c),
        None,
    )
    if ticker_col is None and len(df.columns) > 0:
        for col in df.columns:
            sample = df[col].astype(str)
            if sample.str.match(r"^\d{6}$").all():
                ticker_col = col
                break
    if ticker_col is None:
        raise ValueError(f"Cannot find ticker column in KRX response: {df.columns.tolist()}")
    return df[ticker_col].tolist()


def get_stock_ticker_list(
    target_date: Optional[date] = None,
    market: str = "KOSPI",
) -> List[str]:
    """
    Fetch all stock tickers listed on KRX for the given date.

    Tries pykrx first, falls back to direct KRX API call if pykrx fails.

    Args:
        target_date: The date to query. Defaults to today.
        market: Market filter ("KOSPI", "KOSDAQ", "ALL").

    Returns:
        List of 6-digit ticker strings.

    Raises:
        RuntimeError: If the fetch fails after all retries.
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime(DATE_FORMAT_KRX)
    krx_market = {"KOSPI": "STK", "KOSDAQ": "KSQ", "ALL": "ALL"}.get(market, "STK")

    for attempt in range(1, MAX_RETRIES + 1):
        # Try pykrx first
        try:
            tickers = stock.get_market_ticker_list(date_str, market=market)
            if tickers:
                logger.info(
                    "Fetched %d stock tickers for %s (pykrx)",
                    len(tickers),
                    date_str,
                )
                return list(tickers)
        except Exception:
            pass

        # Fallback: direct KRX API
        try:
            tickers = _fetch_stock_tickers_direct(krx_market)
            if tickers:
                logger.info(
                    "Fetched %d stock tickers for %s (direct API)",
                    len(tickers),
                    date_str,
                )
                return tickers
            else:
                logger.warning(
                    "Empty ticker list for %s (attempt %d/%d)",
                    date_str,
                    attempt,
                    MAX_RETRIES,
                )
        except Exception as e:
            logger.warning(
                "Failed to fetch stock tickers (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                e,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Failed to fetch stock ticker list after {MAX_RETRIES} attempts"
    )


def get_stock_name(ticker: str) -> str:
    """Get the Korean name of a stock by ticker."""
    try:
        name = stock.get_market_ticker_name(ticker)
        return name if name else ticker
    except Exception:
        logger.warning("Could not fetch name for ticker %s", ticker)
        return ticker


def validate_tickers(
    tickers: List[str], target_date: Optional[date] = None
) -> List[str]:
    """
    Filter a list of tickers to only those currently listed on KRX.

    Returns:
        Filtered list containing only valid (currently listed) tickers.
    """
    universe = set(get_stock_ticker_list(target_date))
    valid = [t for t in tickers if t in universe]
    invalid = [t for t in tickers if t not in universe]

    if invalid:
        logger.warning("Tickers not found in KRX universe: %s", invalid)

    return valid
