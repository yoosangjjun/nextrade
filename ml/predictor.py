"""
ML prediction for ETF returns.

Loads trained LightGBM models and generates predictions with
confidence levels for individual or all watchlist ETFs.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from config.settings import ML_CONFIDENCE_THRESHOLDS, ML_MODEL_DIR
from data.cache import OHLCVCache
from ml.features import FEATURE_COLUMNS, build_feature_matrix

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}


@dataclass
class Prediction:
    """ML prediction for a single ETF."""

    ticker: str
    predicted_label: str  # "up", "flat", "down"
    predicted_return: float  # From regressor
    probabilities: Dict[str, float]  # {"up": 0.6, "flat": 0.3, "down": 0.1}
    confidence: str  # "high", "medium", "low"
    confidence_score: float  # Max class probability


def load_models(
    model_dir: Optional[Path] = None,
) -> Tuple[LGBMClassifier, LGBMRegressor, List[str]]:
    """Load the most recent trained models and feature list.

    Args:
        model_dir: Directory containing model files. Defaults to ML_MODEL_DIR.

    Returns:
        Tuple of (classifier, regressor, feature_columns).

    Raises:
        FileNotFoundError: If no models are found.
    """
    model_dir = model_dir or ML_MODEL_DIR

    # Find the most recent metadata file
    meta_files = sorted(model_dir.glob("metadata_*.json"), reverse=True)
    if not meta_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")

    meta_path = meta_files[0]
    metadata = json.loads(meta_path.read_text())
    date_str = metadata["date"]

    clf_path = model_dir / f"classifier_{date_str}.joblib"
    reg_path = model_dir / f"regressor_{date_str}.joblib"

    if not clf_path.exists() or not reg_path.exists():
        raise FileNotFoundError(
            f"Model files missing for date {date_str}: "
            f"clf={clf_path.exists()}, reg={reg_path.exists()}"
        )

    classifier = joblib.load(clf_path)
    regressor = joblib.load(reg_path)
    features = metadata["features"]

    logger.info("Loaded models from %s (%d features)", date_str, len(features))
    return classifier, regressor, features


def _classify_confidence(max_prob: float) -> str:
    """Classify prediction confidence based on max class probability."""
    if max_prob >= ML_CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if max_prob >= ML_CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def predict_single(
    ticker: str,
    cache: OHLCVCache,
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    features: List[str],
) -> Optional[Prediction]:
    """Generate ML prediction for a single ETF.

    Args:
        ticker: ETF ticker code.
        cache: OHLCVCache instance.
        classifier: Trained LGBMClassifier.
        regressor: Trained LGBMRegressor.
        features: List of feature column names the models expect.

    Returns:
        Prediction dataclass, or None if insufficient data.
    """
    # Build features without labels (prediction mode)
    matrix = build_feature_matrix(cache, tickers=[ticker], add_label=False)

    if matrix.empty:
        logger.warning("No feature data for %s", ticker)
        return None

    # Use the most recent row
    latest = matrix.iloc[[-1]]
    available_features = [f for f in features if f in latest.columns]
    if len(available_features) < len(features) * 0.5:
        logger.warning(
            "Too few features for %s: %d/%d",
            ticker,
            len(available_features),
            len(features),
        )
        return None

    X = latest[available_features].values

    # Handle any remaining NaN by filling with 0
    X = np.nan_to_num(X, nan=0.0)

    # Predictions
    label_idx = classifier.predict(X)[0]
    proba = classifier.predict_proba(X)[0]
    predicted_return = regressor.predict(X)[0]

    predicted_label = LABEL_NAMES.get(int(label_idx), "flat")
    probabilities = {
        "down": float(proba[0]),
        "flat": float(proba[1]),
        "up": float(proba[2]),
    }
    max_prob = float(np.max(proba))
    confidence = _classify_confidence(max_prob)

    return Prediction(
        ticker=ticker,
        predicted_label=predicted_label,
        predicted_return=float(predicted_return),
        probabilities=probabilities,
        confidence=confidence,
        confidence_score=max_prob,
    )


def predict_all(
    cache: OHLCVCache,
    tickers: Optional[List[str]] = None,
    model_dir: Optional[Path] = None,
) -> List[Prediction]:
    """Generate ML predictions for all watchlist ETFs.

    Args:
        cache: OHLCVCache instance.
        tickers: List of tickers. Defaults to full watchlist.
        model_dir: Model directory. Defaults to ML_MODEL_DIR.

    Returns:
        List of Prediction objects, sorted by predicted return descending.
    """
    classifier, regressor, features = load_models(model_dir)

    if tickers is None:
        from config.watchlist import get_all_watchlist_tickers
        tickers = get_all_watchlist_tickers()

    predictions = []
    for ticker in tickers:
        pred = predict_single(ticker, cache, classifier, regressor, features)
        if pred is not None:
            predictions.append(pred)

    # Sort by predicted return descending
    predictions.sort(key=lambda p: p.predicted_return, reverse=True)

    logger.info("Generated predictions for %d/%d tickers", len(predictions), len(tickers))
    return predictions
