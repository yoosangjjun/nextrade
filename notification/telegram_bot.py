"""Telegram bot for stock trading signal notifications and commands."""

import asyncio
import logging
from io import BytesIO
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from analysis.indicators import calculate_all_indicators
from analysis.sector_rotation import calculate_sector_momentum
from analysis.signals import generate_signals
from config.watchlist import get_ticker_name_map
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from data.cache import OHLCVCache
from ml.predictor import load_models, predict_single
from notification.chart import generate_price_chart
from notification.formatter import (
    format_daily_report,
    format_stock_detail,
    format_sector_report,
    format_urgent_signal,
)
from screening.screener import screen_all

logger = logging.getLogger(__name__)


async def _run_sync(func, *args):
    """Run a synchronous function in the default executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


# ---- Command Handlers ----

async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /today and /top10 commands: send daily screening report."""
    await update.message.reply_text("Analyzing... please wait.")

    try:
        cache = OHLCVCache()
        results = await _run_sync(screen_all, cache)
        text = format_daily_report(results, top_n=10)
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        logger.error("Error in /today: %s", e, exc_info=True)
        await update.message.reply_text(f"Error: {e}")


async def cmd_sector(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sector command: send sector momentum report."""
    try:
        cache = OHLCVCache()
        sectors = await _run_sync(calculate_sector_momentum, cache)
        text = format_sector_report(sectors)
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        logger.error("Error in /sector: %s", e, exc_info=True)
        await update.message.reply_text(f"Error: {e}")


async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stock <ticker> command: send detailed stock analysis with chart."""
    if not context.args:
        await update.message.reply_text("Usage: /stock <ticker>\nExample: /stock 005930")
        return

    ticker = context.args[0].strip()
    await update.message.reply_text(f"Analyzing {ticker}...")

    try:
        cache = OHLCVCache()
        name_map = get_ticker_name_map()

        # Get screening result
        results = await _run_sync(screen_all, cache)
        result = next((r for r in results if r.ticker == ticker), None)

        if result is None:
            await update.message.reply_text(
                f"Ticker {ticker} not found in screening results."
            )
            return

        # Get detailed signal
        signal = None
        df = cache.get_ohlcv(ticker)
        if len(df) >= 120:
            df_ind = calculate_all_indicators(df)
            signal = generate_signals(df_ind, ticker=ticker)

        # Get ML prediction
        prediction = None
        try:
            clf, reg, features = load_models()
            prediction = predict_single(ticker, cache, clf, reg, features)
        except FileNotFoundError:
            pass

        # Send detail text
        text = format_stock_detail(result, signal=signal, prediction=prediction)
        await update.message.reply_text(text, parse_mode="HTML")

        # Send chart
        name = name_map.get(ticker, "")
        chart_bytes = await _run_sync(generate_price_chart, ticker, cache, name)
        if chart_bytes:
            await update.message.reply_photo(
                photo=BytesIO(chart_bytes),
                caption=f"{name} ({ticker}) - 60 Day Chart",
            )

    except Exception as e:
        logger.error("Error in /stock %s: %s", ticker, e, exc_info=True)
        await update.message.reply_text(f"Error: {e}")


# ---- Scheduled Send Functions ----

async def send_daily_report(app: Application) -> None:
    """Send daily screening report and urgent signals to the configured chat."""
    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not set, skipping daily report")
        return

    try:
        cache = OHLCVCache()
        results = await _run_sync(screen_all, cache)

        if not results:
            logger.warning("No screening results for daily report")
            return

        # Main report
        text = format_daily_report(results, top_n=10)
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML",
        )

        # Urgent signals (technical_score >= 85 or <= 15)
        name_map = get_ticker_name_map()
        for r in results:
            if r.technical_score >= 85 or r.technical_score <= 15:
                # Get signal details
                signal = None
                cache_inst = OHLCVCache()
                df = cache_inst.get_ohlcv(r.ticker)
                if len(df) >= 120:
                    df_ind = calculate_all_indicators(df)
                    signal = generate_signals(df_ind, ticker=r.ticker)

                prediction = None
                try:
                    clf, reg, features = load_models()
                    prediction = predict_single(r.ticker, cache_inst, clf, reg, features)
                except FileNotFoundError:
                    pass

                alert_text = format_urgent_signal(r, signal=signal, prediction=prediction)
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, text=alert_text, parse_mode="HTML",
                )

                # Send chart with urgent signal
                name = name_map.get(r.ticker, "")
                chart_bytes = await _run_sync(
                    generate_price_chart, r.ticker, cache_inst, name,
                )
                if chart_bytes:
                    await app.bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID,
                        photo=BytesIO(chart_bytes),
                        caption=f"{name} ({r.ticker})",
                    )

        logger.info("Daily report sent successfully")

    except Exception as e:
        logger.error("Failed to send daily report: %s", e, exc_info=True)


async def send_sector_report(app: Application) -> None:
    """Send sector momentum report to the configured chat."""
    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not set, skipping sector report")
        return

    try:
        cache = OHLCVCache()
        sectors = await _run_sync(calculate_sector_momentum, cache)
        text = format_sector_report(sectors)
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML",
        )
        logger.info("Sector report sent successfully")
    except Exception as e:
        logger.error("Failed to send sector report: %s", e, exc_info=True)


async def send_retrain_notification(
    app: Application,
    clf_cv_mean: float,
    reg_mae: float,
    n_samples: int,
    trigger: str,
) -> None:
    """Send model retraining result notification."""
    if not TELEGRAM_CHAT_ID:
        return

    from notification.formatter import format_model_retrain_report

    text = format_model_retrain_report(clf_cv_mean, reg_mae, n_samples, trigger)
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Failed to send retrain notification: %s", e, exc_info=True)


def build_telegram_app() -> Application:
    """Build and return a configured Telegram Application.

    Raises ValueError if TELEGRAM_BOT_TOKEN is not set.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Set it in .env or environment variables."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("top10", cmd_today))
    app.add_handler(CommandHandler("sector", cmd_sector))
    app.add_handler(CommandHandler("stock", cmd_stock))

    logger.info("Telegram bot configured with 4 command handlers")
    return app
