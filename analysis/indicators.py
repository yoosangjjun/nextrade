"""
Technical indicator calculations for ETF analysis.

Computes: MA (5/20/60/120), RSI (14), MACD (12/26/9),
Bollinger Bands (20, 2σ), Volume MA (20).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5, MA20, MA60, MA120 columns."""
    for period in [5, 20, 60, 120]:
        df[f"ma{period}"] = _sma(df["close"], period)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column (default 14-period)."""
    df["rsi"] = _rsi(df["close"], period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, MACD signal, MACD histogram columns."""
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = fast_ema - slow_ema
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) columns."""
    df["bb_middle"] = _sma(df["close"], period)
    rolling_std = df["close"].rolling(window=period, min_periods=period).std()
    df["bb_upper"] = df["bb_middle"] + std * rolling_std
    df["bb_lower"] = df["bb_middle"] - std * rolling_std
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume moving average and volume ratio columns."""
    df["volume_ma20"] = _sma(df["volume"], period)
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators on an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Must have a DatetimeIndex.

    Returns:
        DataFrame with all indicator columns added.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to calculate_all_indicators")
        return df

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_ma(df)

    logger.debug(
        "Calculated indicators: %d rows, %d columns",
        len(df),
        len(df.columns),
    )
    return df
