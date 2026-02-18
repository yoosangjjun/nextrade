"""
Rule-based trading signal generation.

Generates buy/sell signals from technical indicators and
combines them into a composite score (0-100).

Signal scoring (buy example):
  Golden cross (MA5 > MA20)        → +25
  RSI 30~50 range                  → +20
  MACD signal crossover up         → +25
  Bollinger lower band bounce      → +15
  Volume surge (>150% of MA20)     → +15
  ─────────────────────────────
  70+: Buy signal / 30-: Sell signal
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    name: str
    signal_type: str  # "buy", "sell", "neutral"
    score: int  # positive for buy, negative for sell
    description: str


@dataclass
class CompositeSignal:
    ticker: str
    date: str
    close: float
    total_score: int
    signal_type: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    signals: List[Signal] = field(default_factory=list)

    @property
    def summary(self) -> str:
        labels = {
            "strong_buy": "강력 매수",
            "buy": "매수",
            "neutral": "중립",
            "sell": "매도",
            "strong_sell": "강력 매도",
        }
        return labels.get(self.signal_type, self.signal_type)


def check_golden_dead_cross(row: pd.Series, prev_row: pd.Series) -> Signal:
    """Check MA5/MA20 golden cross or dead cross."""
    ma5 = row.get("ma5")
    ma20 = row.get("ma20")
    prev_ma5 = prev_row.get("ma5")
    prev_ma20 = prev_row.get("ma20")

    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(prev_ma5) or pd.isna(prev_ma20):
        return Signal("이동평균 크로스", "neutral", 0, "데이터 부족")

    # Golden cross: MA5 crosses above MA20
    if prev_ma5 <= prev_ma20 and ma5 > ma20:
        return Signal("골든크로스", "buy", 25, "MA5가 MA20 상향 돌파")

    # Dead cross: MA5 crosses below MA20
    if prev_ma5 >= prev_ma20 and ma5 < ma20:
        return Signal("데드크로스", "sell", -25, "MA5가 MA20 하향 돌파")

    # Ongoing trend
    if ma5 > ma20:
        return Signal("이동평균 정배열", "buy", 10, "MA5 > MA20 유지 중")

    if ma5 < ma20:
        return Signal("이동평균 역배열", "sell", -10, "MA5 < MA20 유지 중")

    return Signal("이동평균 크로스", "neutral", 0, "MA5 ≈ MA20")


def check_rsi(row: pd.Series, prev_row: pd.Series) -> Signal:
    """Check RSI overbought/oversold conditions."""
    rsi = row.get("rsi")
    prev_rsi = prev_row.get("rsi")

    if pd.isna(rsi) or pd.isna(prev_rsi):
        return Signal("RSI", "neutral", 0, "데이터 부족")

    # Oversold bounce: RSI was below 30, now rising
    if prev_rsi < 30 and rsi > prev_rsi:
        return Signal("RSI 과매도 반등", "buy", 20, f"RSI {rsi:.0f} (30 이하에서 반등)")

    # RSI in buy zone (30-50)
    if 30 <= rsi <= 50:
        return Signal("RSI 매수 구간", "buy", 10, f"RSI {rsi:.0f} (30~50)")

    # Overbought decline: RSI was above 70, now falling
    if prev_rsi > 70 and rsi < prev_rsi:
        return Signal("RSI 과매수 하락", "sell", -20, f"RSI {rsi:.0f} (70 이상에서 하락)")

    # RSI in sell zone (70+)
    if rsi >= 70:
        return Signal("RSI 과매수", "sell", -10, f"RSI {rsi:.0f} (70 이상)")

    return Signal("RSI", "neutral", 0, f"RSI {rsi:.0f} (중립)")


def check_macd(row: pd.Series, prev_row: pd.Series) -> Signal:
    """Check MACD signal line crossover."""
    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    prev_macd = prev_row.get("macd")
    prev_signal = prev_row.get("macd_signal")

    if pd.isna(macd) or pd.isna(macd_signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
        return Signal("MACD", "neutral", 0, "데이터 부족")

    # MACD crosses above signal line
    if prev_macd <= prev_signal and macd > macd_signal:
        return Signal("MACD 상향 돌파", "buy", 25, "MACD가 시그널 상향 돌파")

    # MACD crosses below signal line
    if prev_macd >= prev_signal and macd < macd_signal:
        return Signal("MACD 하향 돌파", "sell", -25, "MACD가 시그널 하향 돌파")

    # Ongoing bullish
    if macd > macd_signal:
        return Signal("MACD 양호", "buy", 10, "MACD > 시그널 유지")

    if macd < macd_signal:
        return Signal("MACD 약세", "sell", -10, "MACD < 시그널 유지")

    return Signal("MACD", "neutral", 0, "MACD ≈ 시그널")


def check_bollinger_bands(row: pd.Series, prev_row: pd.Series) -> Signal:
    """Check Bollinger Band touch and bounce."""
    close = row.get("close")
    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    prev_close = prev_row.get("close")
    prev_bb_lower = prev_row.get("bb_lower")

    if pd.isna(close) or pd.isna(bb_lower) or pd.isna(bb_upper):
        return Signal("볼린저밴드", "neutral", 0, "데이터 부족")

    # Lower band bounce: price was near/below lower band, now rising
    if not pd.isna(prev_close) and not pd.isna(prev_bb_lower):
        if prev_close <= prev_bb_lower * 1.01 and close > prev_close:
            return Signal(
                "볼린저 하단 반등", "buy", 15,
                "하단 밴드 근접 후 반등"
            )

    # Upper band rejection: price touches upper band and falls
    if not pd.isna(prev_close):
        prev_bb_upper = prev_row.get("bb_upper")
        if not pd.isna(prev_bb_upper):
            if prev_close >= prev_bb_upper * 0.99 and close < prev_close:
                return Signal(
                    "볼린저 상단 하락", "sell", -15,
                    "상단 밴드 근접 후 하락"
                )

    # Current position relative to bands
    bb_percent = row.get("bb_percent")
    if not pd.isna(bb_percent):
        if bb_percent < 0.2:
            return Signal("볼린저 하단 근접", "buy", 5, f"BB%: {bb_percent:.1%}")
        if bb_percent > 0.8:
            return Signal("볼린저 상단 근접", "sell", -5, f"BB%: {bb_percent:.1%}")

    return Signal("볼린저밴드", "neutral", 0, "밴드 중앙")


def check_volume_surge(row: pd.Series) -> Signal:
    """Check for volume surge relative to 20-day average."""
    volume_ratio = row.get("volume_ratio")

    if pd.isna(volume_ratio):
        return Signal("거래량", "neutral", 0, "데이터 부족")

    if volume_ratio >= 2.0:
        return Signal(
            "거래량 급등", "buy", 15,
            f"20일 평균 대비 {volume_ratio:.0%}"
        )

    if volume_ratio >= 1.5:
        return Signal(
            "거래량 증가", "buy", 10,
            f"20일 평균 대비 {volume_ratio:.0%}"
        )

    if volume_ratio < 0.5:
        return Signal(
            "거래량 급감", "sell", -5,
            f"20일 평균 대비 {volume_ratio:.0%}"
        )

    return Signal("거래량", "neutral", 0, f"20일 평균 대비 {volume_ratio:.0%}")


def classify_signal(total_score: int) -> str:
    """Classify the total score into a signal type."""
    if total_score >= 70:
        return "strong_buy"
    if total_score >= 40:
        return "buy"
    if total_score <= -70:
        return "strong_sell"
    if total_score <= -40:
        return "sell"
    return "neutral"


def generate_signals(
    df: pd.DataFrame,
    ticker: str = "",
) -> Optional[CompositeSignal]:
    """
    Generate composite trading signals from a DataFrame with indicators.

    Analyzes the most recent row (today) against the previous row.

    Args:
        df: DataFrame with OHLCV + indicator columns.
        ticker: ETF ticker code for labeling.

    Returns:
        CompositeSignal for the latest date, or None if insufficient data.
    """
    if len(df) < 2:
        logger.warning("Need at least 2 rows to generate signals for %s", ticker)
        return None

    row = df.iloc[-1]
    prev_row = df.iloc[-2]

    signals = [
        check_golden_dead_cross(row, prev_row),
        check_rsi(row, prev_row),
        check_macd(row, prev_row),
        check_bollinger_bands(row, prev_row),
        check_volume_surge(row),
    ]

    # Only sum scores from non-neutral signals for the total
    total_score = sum(s.score for s in signals)
    signal_type = classify_signal(total_score)

    date_str = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

    return CompositeSignal(
        ticker=ticker,
        date=date_str,
        close=float(row["close"]),
        total_score=total_score,
        signal_type=signal_type,
        signals=signals,
    )
