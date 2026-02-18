"""
Feature engineering for ML prediction models.

Builds a feature matrix from OHLCV data + technical indicators for
predicting 5-day forward returns. Features include:
  - Return features (1/5/20-day returns, 20-day volatility)
  - Market features (KOSPI200 index returns/volume)
  - Sector features (same-category average return)
  - Derived features (price/MA ratios, relative returns)
  - Labels (3-class: up/flat/down based on ±2% threshold)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.indicators import calculate_all_indicators
from config.watchlist import (
    get_all_watchlist_tickers,
    get_category_for_ticker,
    get_tickers_grouped_by_category,
)
from config.settings import (
    MARKET_BENCHMARK_TICKER,
    ML_LABEL_THRESHOLD,
    ML_PREDICTION_HORIZON,
    ML_TRAIN_MIN_ROWS,
)
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)

# ~30 features used by the model
FEATURE_COLUMNS: List[str] = [
    # Return features
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volatility_20d",
    # Technical indicators
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_percent",
    "bb_bandwidth",
    "volume_ratio",
    # Price/MA ratios
    "close_ma5_ratio",
    "close_ma20_ratio",
    "close_ma60_ratio",
    "close_ma120_ratio",
    "ma5_ma20_ratio",
    "ma20_ma60_ratio",
    # Market-relative features
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_volatility_20d",
    "market_volume_ratio",
    "relative_ret_1d",
    "relative_ret_5d",
    "relative_ret_20d",
    # Sector features
    "sector_avg_ret_1d",
    "sector_avg_ret_5d",
    "relative_sector_ret_1d",
    # Derived
    "trading_value_ma20_ratio",
]


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return and volatility features."""
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    return df


def add_market_features(
    df: pd.DataFrame, market_df: pd.DataFrame
) -> pd.DataFrame:
    """Add KOSPI200 index return and volume features.

    Args:
        df: Single stock DataFrame with DatetimeIndex.
        market_df: KOSPI200 index DataFrame with DatetimeIndex.
    """
    market = market_df[["close", "volume"]].copy()
    market["market_ret_1d"] = market["close"].pct_change(1)
    market["market_ret_5d"] = market["close"].pct_change(5)
    market["market_ret_20d"] = market["close"].pct_change(20)
    market["market_volatility_20d"] = market["market_ret_1d"].rolling(20).std()
    vol_ma20 = market["volume"].rolling(20).mean()
    market["market_volume_ratio"] = market["volume"] / vol_ma20

    market_cols = [
        "market_ret_1d",
        "market_ret_5d",
        "market_ret_20d",
        "market_volatility_20d",
        "market_volume_ratio",
    ]
    df = df.join(market[market_cols], how="left")

    # Relative returns (stock - market)
    df["relative_ret_1d"] = df["ret_1d"] - df["market_ret_1d"]
    df["relative_ret_5d"] = df["ret_5d"] - df["market_ret_5d"]
    df["relative_ret_20d"] = df["ret_20d"] - df["market_ret_20d"]

    return df


def add_sector_features(
    df: pd.DataFrame,
    ticker: str,
    all_ticker_returns: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Add same-category average return features.

    Args:
        df: Single stock DataFrame with DatetimeIndex and ret_1d/ret_5d.
        ticker: The stock ticker code.
        all_ticker_returns: Dict mapping ticker -> Series of 1d returns.
    """
    category = get_category_for_ticker(ticker)
    grouped = get_tickers_grouped_by_category()
    peers = [t for t in grouped.get(category, []) if t != ticker]

    if not peers:
        df["sector_avg_ret_1d"] = 0.0
        df["sector_avg_ret_5d"] = 0.0
    else:
        peer_1d = pd.DataFrame(
            {t: all_ticker_returns[t]["ret_1d"] for t in peers if t in all_ticker_returns}
        )
        peer_5d = pd.DataFrame(
            {t: all_ticker_returns[t]["ret_5d"] for t in peers if t in all_ticker_returns}
        )
        if peer_1d.empty:
            df["sector_avg_ret_1d"] = 0.0
            df["sector_avg_ret_5d"] = 0.0
        else:
            sector_avg_1d = peer_1d.mean(axis=1)
            sector_avg_5d = peer_5d.mean(axis=1)
            df = df.join(sector_avg_1d.rename("sector_avg_ret_1d"), how="left")
            df = df.join(sector_avg_5d.rename("sector_avg_ret_5d"), how="left")

    df["relative_sector_ret_1d"] = df["ret_1d"] - df["sector_avg_ret_1d"]
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/MA ratios and trading value features."""
    # Price relative to moving averages
    for period in [5, 20, 60, 120]:
        ma_col = f"ma{period}"
        if ma_col in df.columns:
            df[f"close_ma{period}_ratio"] = df["close"] / df[ma_col] - 1

    # MA cross ratios
    if "ma5" in df.columns and "ma20" in df.columns:
        df["ma5_ma20_ratio"] = df["ma5"] / df["ma20"] - 1
    if "ma20" in df.columns and "ma60" in df.columns:
        df["ma20_ma60_ratio"] = df["ma20"] / df["ma60"] - 1

    # Trading value relative feature
    if "trading_value" in df.columns:
        tv_ma20 = df["trading_value"].rolling(20).mean()
        df["trading_value_ma20_ratio"] = df["trading_value"] / tv_ma20
    else:
        df["trading_value_ma20_ratio"] = np.nan

    return df


def add_labels(
    df: pd.DataFrame,
    horizon: int = ML_PREDICTION_HORIZON,
    threshold: float = ML_LABEL_THRESHOLD,
) -> pd.DataFrame:
    """Add forward return and 3-class label.

    Labels:
        2 = up   (return > +threshold)
        1 = flat (|return| <= threshold)
        0 = down (return < -threshold)
    """
    df["fwd_return"] = df["close"].pct_change(horizon).shift(-horizon)
    df["label"] = 1  # default flat
    df.loc[df["fwd_return"] > threshold, "label"] = 2
    df.loc[df["fwd_return"] < -threshold, "label"] = 0
    return df


def _prepare_single_ticker(
    ticker: str,
    cache: OHLCVCache,
    market_df: Optional[pd.DataFrame] = None,
    all_ticker_returns: Optional[Dict[str, pd.Series]] = None,
    add_label: bool = True,
) -> Optional[pd.DataFrame]:
    """Load and feature-engineer data for a single ticker.

    Returns None if insufficient data.
    """
    df = cache.get_ohlcv(ticker)
    if df.empty or len(df) < ML_TRAIN_MIN_ROWS:
        logger.debug("Skipping %s: only %d rows", ticker, len(df))
        return None

    # Technical indicators
    df = calculate_all_indicators(df)

    # Return features
    df = add_return_features(df)

    # Market features
    if market_df is not None:
        df = add_market_features(df, market_df)
    else:
        for col in [c for c in FEATURE_COLUMNS if c.startswith("market_") or c.startswith("relative_ret_")]:
            df[col] = np.nan

    # Sector features
    if all_ticker_returns is not None:
        df = add_sector_features(df, ticker, all_ticker_returns)
    else:
        df["sector_avg_ret_1d"] = 0.0
        df["sector_avg_ret_5d"] = 0.0
        df["relative_sector_ret_1d"] = 0.0

    # Derived features
    df = add_derived_features(df)

    # Labels (only for training)
    if add_label:
        df = add_labels(df)

    # Add ticker column for identification
    df["ticker"] = ticker

    return df


def build_feature_matrix(
    cache: OHLCVCache,
    tickers: Optional[List[str]] = None,
    add_label: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix for all tickers.

    This is the main orchestrator that:
    1. Loads market proxy data
    2. Computes per-ticker returns for sector features
    3. Runs the full feature pipeline for each ticker
    4. Concatenates into a single DataFrame

    Args:
        cache: OHLCVCache instance.
        tickers: List of tickers to include. Defaults to full watchlist.
        add_label: Whether to add forward return labels (False for prediction).

    Returns:
        DataFrame with FEATURE_COLUMNS + ticker + (label columns if add_label).
    """
    if tickers is None:
        tickers = get_all_watchlist_tickers()

    # 1. Load market benchmark
    market_df = cache.get_ohlcv(MARKET_BENCHMARK_TICKER)
    if market_df.empty:
        logger.warning("Market benchmark %s not found in cache", MARKET_BENCHMARK_TICKER)
        market_df = None

    # 2. Pre-compute per-ticker returns for sector features
    all_ticker_returns: Dict[str, pd.Series] = {}
    for ticker in tickers:
        df = cache.get_ohlcv(ticker)
        if not df.empty and len(df) >= 5:
            returns = pd.DataFrame(index=df.index)
            returns["ret_1d"] = df["close"].pct_change(1)
            returns["ret_5d"] = df["close"].pct_change(5)
            all_ticker_returns[ticker] = returns

    # 3. Build features for each ticker
    frames = []
    for ticker in tickers:
        result = _prepare_single_ticker(
            ticker=ticker,
            cache=cache,
            market_df=market_df,
            all_ticker_returns=all_ticker_returns,
            add_label=add_label,
        )
        if result is not None:
            frames.append(result)

    if not frames:
        logger.warning("No valid data for any ticker")
        return pd.DataFrame()

    # 4. Concatenate
    matrix = pd.concat(frames, axis=0)

    # Drop rows with NaN in critical feature columns (from rolling windows)
    available_features = [c for c in FEATURE_COLUMNS if c in matrix.columns]
    matrix = matrix.dropna(subset=available_features)

    if add_label:
        matrix = matrix.dropna(subset=["label", "fwd_return"])
        matrix["label"] = matrix["label"].astype(int)

    logger.info(
        "Feature matrix: %d rows, %d features, %d tickers",
        len(matrix),
        len(available_features),
        matrix["ticker"].nunique(),
    )
    return matrix
