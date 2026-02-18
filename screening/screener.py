"""
Comprehensive stock screening and ranking.

Combines technical signal scores, ML predictions, sector momentum,
and relative strength into a single composite score for ranking.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from analysis.indicators import calculate_all_indicators
from analysis.relative_strength import RelativeStrength, calculate_relative_strength
from analysis.sector_rotation import SectorMomentum, calculate_sector_momentum
from analysis.signals import generate_signals
from config.watchlist import (
    get_all_watchlist_tickers,
    get_category_for_ticker,
    get_ticker_name_map,
)
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Comprehensive screening result for a single stock."""

    ticker: str
    name: str
    category: str
    close: float
    technical_score: float       # 0~100 normalized
    ml_predicted_return: Optional[float]
    ml_label: Optional[str]      # "up", "flat", "down", or None
    sector_momentum_rank: int    # Category rank by 20d momentum (1=best)
    relative_strength_score: float
    composite_score: float
    composite_rank: int


def _normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0~100 scale.

    Args:
        value: The value to normalize.
        min_val: Minimum of the value range.
        max_val: Maximum of the value range.

    Returns:
        Normalized value between 0 and 100.
    """
    if max_val == min_val:
        return 50.0
    clamped = max(min_val, min(max_val, value))
    return ((clamped - min_val) / (max_val - min_val)) * 100


def screen_all(
    cache: OHLCVCache,
    model_dir: Optional[Path] = None,
) -> List[ScreeningResult]:
    """Run comprehensive screening on all watchlist stocks.

    Pipeline:
    1. Generate technical signal scores for each stock.
    2. Load ML predictions (if models available).
    3. Calculate sector momentum and relative strength.
    4. Compute composite score with weighted combination.

    Weights (with ML):   technical×0.3 + ml×0.3 + sector×0.2 + rs×0.2
    Weights (no ML):     technical×0.4 + sector×0.3 + rs×0.3

    Args:
        cache: OHLCVCache instance.
        model_dir: ML model directory. None = default path.

    Returns:
        List of ScreeningResult sorted by composite_score descending.
    """
    name_map = get_ticker_name_map()
    tickers = get_all_watchlist_tickers()

    # --- Step 1: Technical scores ---
    tech_scores: Dict[str, dict] = {}
    for ticker in tickers:
        df = cache.get_ohlcv(ticker)
        if df.empty or len(df) < 120:
            continue

        df = calculate_all_indicators(df)
        signal = generate_signals(df, ticker=ticker)
        if signal is None:
            continue

        tech_scores[ticker] = {
            "total_score": signal.total_score,
            "close": signal.close,
        }

    if not tech_scores:
        return []

    # Normalize technical scores to 0~100
    raw_scores = [v["total_score"] for v in tech_scores.values()]
    tech_min = min(raw_scores)
    tech_max = max(raw_scores)

    for data in tech_scores.values():
        data["normalized"] = _normalize_score(data["total_score"], tech_min, tech_max)

    # --- Step 2: ML predictions (optional) ---
    ml_predictions: Dict[str, dict] = {}
    has_ml = False
    try:
        from ml.predictor import predict_all as ml_predict_all

        predictions = ml_predict_all(cache, tickers=tickers, model_dir=model_dir)
        for pred in predictions:
            ml_predictions[pred.ticker] = {
                "predicted_return": pred.predicted_return,
                "label": pred.predicted_label,
            }
        has_ml = len(ml_predictions) > 0
    except (FileNotFoundError, ImportError, OSError):
        logger.info("ML models not available, skipping ML component")

    # Normalize ML predicted returns to 0~100
    if has_ml:
        ml_returns = [v["predicted_return"] for v in ml_predictions.values()]
        ml_min = min(ml_returns)
        ml_max = max(ml_returns)
        for data in ml_predictions.values():
            data["normalized"] = _normalize_score(
                data["predicted_return"], ml_min, ml_max
            )

    # --- Step 3: Sector momentum + relative strength ---
    sector_mom = calculate_sector_momentum(cache)
    sector_rank_map: Dict[str, int] = {}
    sector_total: Dict[str, int] = {}
    for sm in sector_mom:
        sector_rank_map[sm.category] = sm.rank_20d
        sector_total[sm.category] = len(sector_mom)

    rs_list = calculate_relative_strength(cache)
    rs_map: Dict[str, RelativeStrength] = {rs.ticker: rs for rs in rs_list}

    # Normalize sector momentum ranks: lower rank = better → invert
    n_sectors = len(sector_mom)
    # Normalize RS scores to 0~100
    rs_scores = [rs.rs_score for rs in rs_list] if rs_list else [0.0]
    rs_min = min(rs_scores)
    rs_max = max(rs_scores)

    # --- Step 4: Composite score ---
    results = []
    for ticker in tickers:
        if ticker not in tech_scores:
            continue

        category = get_category_for_ticker(ticker)
        tech_norm = tech_scores[ticker]["normalized"]

        # Sector momentum: convert rank to 0~100 (rank 1 → 100, rank N → 0)
        cat_rank = sector_rank_map.get(category, n_sectors)
        sector_norm = _normalize_score(
            n_sectors - cat_rank + 1, 1, n_sectors
        ) if n_sectors > 1 else 50.0

        # Relative strength
        rs = rs_map.get(ticker)
        rs_norm = _normalize_score(rs.rs_score, rs_min, rs_max) if rs else 50.0
        rs_score_raw = rs.rs_score if rs else 0.0

        # ML component
        ml_data = ml_predictions.get(ticker)
        ml_return = ml_data["predicted_return"] if ml_data else None
        ml_label = ml_data["label"] if ml_data else None

        if has_ml and ml_data:
            ml_norm = ml_data["normalized"]
            composite = (
                tech_norm * 0.3
                + ml_norm * 0.3
                + sector_norm * 0.2
                + rs_norm * 0.2
            )
        else:
            composite = (
                tech_norm * 0.4
                + sector_norm * 0.3
                + rs_norm * 0.3
            )

        results.append(
            ScreeningResult(
                ticker=ticker,
                name=name_map.get(ticker, ticker),
                category=category,
                close=tech_scores[ticker]["close"],
                technical_score=round(tech_norm, 1),
                ml_predicted_return=ml_return,
                ml_label=ml_label,
                sector_momentum_rank=cat_rank,
                relative_strength_score=round(rs_score_raw, 4),
                composite_score=round(composite, 1),
                composite_rank=0,  # filled below
            )
        )

    # Assign composite rank
    results.sort(key=lambda r: r.composite_score, reverse=True)
    for i, r in enumerate(results):
        r.composite_rank = i + 1

    return results
