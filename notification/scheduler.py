"""APScheduler configuration for automated tasks."""

import json
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application

from config.watchlist import get_all_watchlist_tickers
from config.settings import (
    ML_MODEL_DIR,
    MODEL_PERF_DEGRADATION_RATIO,
    MODEL_PERF_MIN_RECENT_DAYS,
    SCHEDULER_TIMEZONE,
)
from data.cache import OHLCVCache
from data.collector import StockCollector
from ml.features import FEATURE_COLUMNS, build_feature_matrix
from ml.predictor import load_models, predict_all
from ml.trainer import save_models, train_classifier, train_regressor
from notification.telegram_bot import (
    send_daily_report,
    send_retrain_notification,
    send_sector_report,
)
from screening.screener import screen_all

logger = logging.getLogger(__name__)


async def _job_collect_and_report(app: Application) -> None:
    """Daily job: collect data, run screening, send report.

    Scheduled at 15:40 KST (after market close at 15:30).
    """
    logger.info("Starting daily collect-and-report job")

    try:
        # Collect latest data
        cache = OHLCVCache()
        collector = StockCollector(cache=cache)
        tickers = get_all_watchlist_tickers()
        summary = collector.collect_multiple(tickers)
        logger.info(
            "Data collection complete: %d tickers, %d rows added",
            summary["success_count"],
            summary["total_rows"],
        )

        # Send daily report
        await send_daily_report(app)

    except Exception as e:
        logger.error("Daily collect-and-report failed: %s", e, exc_info=True)


async def _job_sector_report(app: Application) -> None:
    """Weekly job: send sector momentum report.

    Scheduled every Monday at 08:30 KST.
    """
    logger.info("Starting weekly sector report job")
    try:
        await send_sector_report(app)
    except Exception as e:
        logger.error("Sector report job failed: %s", e, exc_info=True)


async def _job_monthly_retrain(app: Application) -> None:
    """Monthly job: retrain ML models.

    Scheduled on the 1st of each month at 06:00 KST.
    """
    logger.info("Starting monthly retrain job")
    await _retrain_models(app, trigger="monthly")


async def _job_performance_check(app: Application) -> None:
    """Daily job: check model performance, retrain if degraded.

    Scheduled at 15:45 KST (after daily report).
    Triggers retraining if recent F1 < training CV F1 * degradation ratio.
    """
    logger.info("Starting performance check job")

    try:
        # Load model metadata for training CV F1
        try:
            _, _, _ = load_models()
        except FileNotFoundError:
            logger.info("No models found, skipping performance check")
            return

        metadata_files = sorted(ML_MODEL_DIR.glob("metadata_*.json"), reverse=True)
        if not metadata_files:
            return

        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        train_f1 = metadata.get("classifier", {}).get("f1_macro")
        if train_f1 is None:
            logger.warning("No F1 score in model metadata")
            return

        # Evaluate recent prediction accuracy
        cache = OHLCVCache()
        try:
            predictions = predict_all(cache)
        except Exception:
            logger.warning("Cannot generate predictions for performance check")
            return

        if len(predictions) < MODEL_PERF_MIN_RECENT_DAYS:
            logger.info(
                "Not enough predictions (%d) for performance check (need %d)",
                len(predictions),
                MODEL_PERF_MIN_RECENT_DAYS,
            )
            return

        # Simple heuristic: if very few high-confidence predictions, model may be degraded
        high_conf = sum(1 for p in predictions if p.confidence == "high")
        high_conf_ratio = high_conf / len(predictions)

        # If model's training F1 was decent but very few high-confidence predictions
        threshold = train_f1 * MODEL_PERF_DEGRADATION_RATIO
        if high_conf_ratio < threshold:
            logger.info(
                "Performance degradation detected: high_conf_ratio=%.3f < threshold=%.3f",
                high_conf_ratio,
                threshold,
            )
            await _retrain_models(app, trigger="performance_degradation")
        else:
            logger.info(
                "Model performance OK: high_conf_ratio=%.3f >= threshold=%.3f",
                high_conf_ratio,
                threshold,
            )

    except Exception as e:
        logger.error("Performance check failed: %s", e, exc_info=True)


async def _retrain_models(app: Application, trigger: str) -> None:
    """Retrain classifier and regressor, save models, send notification."""
    try:
        cache = OHLCVCache()

        # Build feature matrix
        matrix = build_feature_matrix(cache, add_label=True)
        if matrix.empty:
            logger.warning("Empty feature matrix, skipping retrain")
            return

        # Train models
        clf, clf_result = train_classifier(matrix)
        reg, reg_result = train_regressor(matrix)

        # Save
        features_used = [c for c in FEATURE_COLUMNS if c in matrix.columns]
        save_models(clf, reg, clf_result, reg_result, features_used)

        logger.info(
            "Retrain complete (%s): clf_f1=%.3f, reg_mae=%.4f, samples=%d",
            trigger,
            clf_result.cv_mean,
            reg_result.mae,
            clf_result.n_samples,
        )

        # Notify via Telegram
        await send_retrain_notification(
            app,
            clf_cv_mean=clf_result.cv_mean,
            reg_mae=reg_result.mae,
            n_samples=clf_result.n_samples,
            trigger=trigger,
        )

    except Exception as e:
        logger.error("Model retrain failed (%s): %s", trigger, e, exc_info=True)


def create_scheduler(app: Application) -> AsyncIOScheduler:
    """Create and configure the scheduler with all jobs.

    Args:
        app: Telegram Application instance for sending messages.

    Returns:
        Configured (but not started) AsyncIOScheduler.
    """
    scheduler = AsyncIOScheduler(timezone=SCHEDULER_TIMEZONE)

    # Daily: collect data + send report (15:40 KST, Mon-Fri)
    scheduler.add_job(
        _job_collect_and_report,
        "cron",
        args=[app],
        day_of_week="mon-fri",
        hour=15,
        minute=40,
        id="daily_report",
        name="Daily collect & report",
    )

    # Weekly: sector report (Monday 08:30 KST)
    scheduler.add_job(
        _job_sector_report,
        "cron",
        args=[app],
        day_of_week="mon",
        hour=8,
        minute=30,
        id="sector_report",
        name="Weekly sector report",
    )

    # Monthly: retrain models (1st of month, 06:00 KST)
    scheduler.add_job(
        _job_monthly_retrain,
        "cron",
        args=[app],
        day=1,
        hour=6,
        minute=0,
        id="monthly_retrain",
        name="Monthly model retrain",
    )

    # Daily: performance check (15:45 KST, Mon-Fri)
    scheduler.add_job(
        _job_performance_check,
        "cron",
        args=[app],
        day_of_week="mon-fri",
        hour=15,
        minute=45,
        id="performance_check",
        name="Daily performance check",
    )

    logger.info("Scheduler configured with 4 jobs (timezone=%s)", SCHEDULER_TIMEZONE)
    return scheduler
