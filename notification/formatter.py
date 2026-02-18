"""HTML message formatting for Telegram notifications.

All functions return plain strings with HTML markup compatible with
Telegram's HTML parse mode. No telegram-api dependency.
"""

from datetime import datetime
from html import escape
from typing import List, Optional


def format_daily_report(results: list, top_n: int = 10) -> str:
    """Format daily screening report with top buy/sell signals.

    Args:
        results: List of ScreeningResult sorted by composite_score desc.
        top_n: Number of top/bottom results to show.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>Daily Report</b>", f"<i>{now}</i>", ""]

    if not results:
        lines.append("No screening results available.")
        return "\n".join(lines)

    label_emoji = {"up": "▲", "flat": "─", "down": "▼"}

    # Top buy signals
    top_buy = results[:top_n]
    lines.append(f"<b>Top {len(top_buy)} Buy Signals</b>")
    for i, r in enumerate(top_buy, 1):
        ml_str = ""
        if r.ml_predicted_return is not None:
            ml_dir = label_emoji.get(r.ml_label, "")
            ml_str = f" | ML {ml_dir}{r.ml_predicted_return:+.1%}"
        lines.append(
            f"{i}. <b>{escape(r.name)}</b> ({r.ticker})"
            f"  Score: {r.composite_score:.1f}"
            f"  Tech: {r.technical_score:.0f}{ml_str}"
        )

    # Top sell signals (bottom of ranking)
    if len(results) > top_n:
        lines.append("")
        top_sell = results[-top_n:][::-1]  # worst first
        lines.append(f"<b>Bottom {len(top_sell)} (Sell Candidates)</b>")
        for i, r in enumerate(top_sell, 1):
            ml_str = ""
            if r.ml_predicted_return is not None:
                ml_dir = label_emoji.get(r.ml_label, "")
                ml_str = f" | ML {ml_dir}{r.ml_predicted_return:+.1%}"
            lines.append(
                f"{i}. <b>{escape(r.name)}</b> ({r.ticker})"
                f"  Score: {r.composite_score:.1f}"
                f"  Tech: {r.technical_score:.0f}{ml_str}"
            )

    return "\n".join(lines)


def format_urgent_signal(
    result,
    signal=None,
    prediction=None,
) -> str:
    """Format urgent buy/sell alert for a single stock.

    Triggered when technical_score >= 85 or <= 15.
    """
    if result.technical_score >= 85:
        header = "STRONG BUY SIGNAL"
    else:
        header = "STRONG SELL SIGNAL"

    lines = [
        f"<b>{header}</b>",
        "",
        f"<b>{escape(result.name)}</b> ({result.ticker})",
        f"Close: {result.close:,.0f} KRW",
        f"Composite Score: <b>{result.composite_score:.1f}</b>",
        f"Technical Score: <b>{result.technical_score:.0f}</b>",
    ]

    if signal and signal.signals:
        lines.append("")
        lines.append("<b>Signal Details:</b>")
        for s in signal.signals:
            if s.score != 0:
                icon = "+" if s.score > 0 else ""
                lines.append(f"  {s.name}: {escape(s.description)} ({icon}{s.score})")

    if prediction:
        label_kr = {"up": "상승", "flat": "보합", "down": "하락"}
        lines.append("")
        lines.append("<b>ML Prediction:</b>")
        lines.append(
            f"  Direction: {label_kr.get(prediction.predicted_label, '?')}"
            f" ({prediction.confidence.upper()})"
        )
        lines.append(f"  Expected return: {prediction.predicted_return:+.2%}")

    return "\n".join(lines)


def format_sector_report(sectors: list) -> str:
    """Format weekly sector momentum report.

    Args:
        sectors: List of SectorMomentum sorted by 20d momentum desc.
    """
    now = datetime.now().strftime("%Y-%m-%d")
    lines = [f"<b>Sector Momentum Report</b>", f"<i>{now}</i>", ""]

    if not sectors:
        lines.append("No sector data available.")
        return "\n".join(lines)

    for sm in sectors:
        arrow = ""
        if sm.rank_change_20d > 0:
            arrow = f" (+{sm.rank_change_20d})"
        elif sm.rank_change_20d < 0:
            arrow = f" ({sm.rank_change_20d})"

        lines.append(
            f"<b>#{sm.rank_20d} {escape(sm.name_kr)}</b>{arrow}\n"
            f"  5D: {sm.momentum_5d:+.2%}"
            f"  20D: {sm.momentum_20d:+.2%}"
            f"  60D: {sm.momentum_60d:+.2%}\n"
            f"  Top: {escape(sm.top_stock)}"
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def format_stock_detail(
    result,
    signal=None,
    prediction=None,
) -> str:
    """Format detailed stock analysis response for /stock command."""
    lines = [
        f"<b>{escape(result.name)}</b> ({result.ticker})",
        f"Category: {escape(result.category)}",
        f"Close: {result.close:,.0f} KRW",
        "",
        "<b>Scores:</b>",
        f"  Composite: <b>{result.composite_score:.1f}</b> (Rank #{result.composite_rank})",
        f"  Technical: {result.technical_score:.0f}",
        f"  Sector Rank: #{result.sector_momentum_rank}",
        f"  RS Score: {result.relative_strength_score:+.2%}",
    ]

    if signal and signal.signals:
        signal_kr = {
            "strong_buy": "강력 매수",
            "buy": "매수",
            "neutral": "중립",
            "sell": "매도",
            "strong_sell": "강력 매도",
        }
        lines.append("")
        lines.append(
            f"<b>Signal:</b> {signal_kr.get(signal.signal_type, signal.signal_type)}"
            f" (Score: {signal.total_score})"
        )
        for s in signal.signals:
            if s.score != 0:
                icon = "+" if s.score > 0 else ""
                lines.append(f"  {s.name}: {escape(s.description)} ({icon}{s.score})")

    if prediction:
        label_kr = {"up": "상승", "flat": "보합", "down": "하락"}
        lines.append("")
        lines.append("<b>ML Prediction (5D):</b>")
        lines.append(
            f"  {label_kr.get(prediction.predicted_label, '?')}"
            f" | Return: {prediction.predicted_return:+.2%}"
            f" | Confidence: {prediction.confidence.upper()}"
        )
        lines.append(
            f"  P(Up)={prediction.probabilities['up']:.0%}"
            f"  P(Flat)={prediction.probabilities['flat']:.0%}"
            f"  P(Down)={prediction.probabilities['down']:.0%}"
        )

    return "\n".join(lines)


def format_model_retrain_report(
    clf_cv_mean: float,
    reg_mae: float,
    n_samples: int,
    trigger: str,
) -> str:
    """Format model retraining notification.

    Args:
        clf_cv_mean: Classifier cross-validation F1 mean.
        reg_mae: Regressor MAE.
        n_samples: Number of training samples.
        trigger: What triggered retraining (e.g., "monthly", "performance_degradation").
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    trigger_kr = {
        "monthly": "월간 정기 재학습",
        "performance_degradation": "성능 저하 감지",
    }

    lines = [
        "<b>Model Retrain Complete</b>",
        f"<i>{now}</i>",
        "",
        f"Trigger: {trigger_kr.get(trigger, trigger)}",
        f"Training samples: {n_samples:,}",
        f"Classifier CV F1: {clf_cv_mean:.3f}",
        f"Regressor MAE: {reg_mae:.4f}",
    ]

    return "\n".join(lines)
