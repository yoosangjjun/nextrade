"""Price chart generation for Telegram notifications."""

import logging
from io import BytesIO
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analysis.indicators import calculate_all_indicators
from data.cache import OHLCVCache

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def generate_price_chart(
    ticker: str,
    cache: OHLCVCache,
    name: str = "",
    days: int = 60,
) -> Optional[bytes]:
    """Generate a price chart with technical indicators as PNG bytes.

    Upper panel (75%): Close price, MA5/20/60, Bollinger Band shading.
    Lower panel (25%): Volume bars colored by price direction.

    Returns None if insufficient data.
    """
    df = cache.get_ohlcv(ticker)
    if len(df) < 60:
        logger.warning("Not enough data for chart: %s (%d rows)", ticker, len(df))
        return None

    df = calculate_all_indicators(df)
    df = df.tail(days)

    if len(df) < 5:
        return None

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.15},
    )

    title = f"{name} ({ticker})" if name else ticker
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    x = range(len(df))
    dates = df.index

    # --- Upper panel: price + indicators ---
    ax_price.plot(x, df["close"], color="#1f77b4", linewidth=1.5, label="Close")

    for ma, color, ls in [
        ("ma5", "#ff7f0e", "-"),
        ("ma20", "#2ca02c", "-"),
        ("ma60", "#d62728", "--"),
    ]:
        if ma in df.columns and df[ma].notna().any():
            ax_price.plot(x, df[ma], color=color, linewidth=0.8, linestyle=ls, label=ma.upper())

    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        ax_price.fill_between(
            x, df["bb_lower"], df["bb_upper"],
            alpha=0.1, color="#1f77b4", label="BB",
        )

    ax_price.legend(loc="upper left", fontsize=8, ncol=5)
    ax_price.set_ylabel("Price (KRW)")
    ax_price.grid(True, alpha=0.3)
    ax_price.set_xlim(0, len(df) - 1)

    # --- Lower panel: volume ---
    colors = [
        "#d62728" if c < o else "#2ca02c"
        for c, o in zip(df["close"], df["open"])
    ]
    ax_vol.bar(x, df["volume"], color=colors, alpha=0.7, width=0.8)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, alpha=0.3)
    ax_vol.set_xlim(0, len(df) - 1)

    # X-axis date labels
    n_labels = min(6, len(df))
    step = max(1, len(df) // n_labels)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [dates[i].strftime("%m/%d") for i in tick_positions]
    ax_vol.set_xticks(tick_positions)
    ax_vol.set_xticklabels(tick_labels, fontsize=8)
    ax_price.set_xticks([])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
