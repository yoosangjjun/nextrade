"""Tests for analysis/signals.py"""

import pandas as pd
import numpy as np

from analysis.signals import (
    Signal,
    check_golden_dead_cross,
    check_rsi,
    check_macd,
    check_volume_surge,
    classify_signal,
    generate_signals,
)
from analysis.indicators import calculate_all_indicators


def _make_ohlcv(n: int = 200, base_price: float = 10000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = np.random.normal(0.001, 0.02, n)
    close = base_price * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": close * (1 + np.random.uniform(0, 0.02, n)),
            "low": close * (1 - np.random.uniform(0, 0.02, n)),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


class TestGoldenDeadCross:
    def test_golden_cross(self):
        prev = pd.Series({"ma5": 100, "ma20": 105})
        curr = pd.Series({"ma5": 106, "ma20": 105})
        signal = check_golden_dead_cross(curr, prev)
        assert signal.signal_type == "buy"
        assert signal.score == 25

    def test_dead_cross(self):
        prev = pd.Series({"ma5": 106, "ma20": 105})
        curr = pd.Series({"ma5": 104, "ma20": 105})
        signal = check_golden_dead_cross(curr, prev)
        assert signal.signal_type == "sell"
        assert signal.score == -25

    def test_ongoing_uptrend(self):
        prev = pd.Series({"ma5": 108, "ma20": 105})
        curr = pd.Series({"ma5": 110, "ma20": 106})
        signal = check_golden_dead_cross(curr, prev)
        assert signal.signal_type == "buy"
        assert signal.score == 10


class TestRSI:
    def test_oversold_bounce(self):
        prev = pd.Series({"rsi": 25})
        curr = pd.Series({"rsi": 32})
        signal = check_rsi(curr, prev)
        assert signal.signal_type == "buy"
        assert signal.score == 20

    def test_overbought_decline(self):
        prev = pd.Series({"rsi": 75})
        curr = pd.Series({"rsi": 68})
        signal = check_rsi(curr, prev)
        assert signal.signal_type == "sell"
        assert signal.score == -20


class TestMACD:
    def test_macd_bullish_crossover(self):
        prev = pd.Series({"macd": -1, "macd_signal": 0})
        curr = pd.Series({"macd": 1, "macd_signal": 0})
        signal = check_macd(curr, prev)
        assert signal.signal_type == "buy"
        assert signal.score == 25

    def test_macd_bearish_crossover(self):
        prev = pd.Series({"macd": 1, "macd_signal": 0})
        curr = pd.Series({"macd": -1, "macd_signal": 0})
        signal = check_macd(curr, prev)
        assert signal.signal_type == "sell"
        assert signal.score == -25


class TestVolumeSurge:
    def test_volume_surge_200pct(self):
        row = pd.Series({"volume_ratio": 2.5})
        signal = check_volume_surge(row)
        assert signal.signal_type == "buy"
        assert signal.score == 15

    def test_volume_normal(self):
        row = pd.Series({"volume_ratio": 1.0})
        signal = check_volume_surge(row)
        assert signal.signal_type == "neutral"


class TestClassifySignal:
    def test_strong_buy(self):
        assert classify_signal(75) == "strong_buy"

    def test_buy(self):
        assert classify_signal(50) == "buy"

    def test_neutral(self):
        assert classify_signal(10) == "neutral"

    def test_sell(self):
        assert classify_signal(-50) == "sell"

    def test_strong_sell(self):
        assert classify_signal(-75) == "strong_sell"


class TestGenerateSignals:
    def test_generates_composite_signal(self):
        df = _make_ohlcv()
        df = calculate_all_indicators(df)
        signal = generate_signals(df, ticker="069500")
        assert signal is not None
        assert signal.ticker == "069500"
        assert len(signal.signals) == 5
        assert signal.signal_type in [
            "strong_buy", "buy", "neutral", "sell", "strong_sell"
        ]

    def test_insufficient_data(self):
        df = _make_ohlcv(n=1)
        signal = generate_signals(df, ticker="069500")
        assert signal is None
