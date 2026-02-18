"""
LightGBM model training for ETF return prediction.

Trains two models:
  - 3-class classifier (up/flat/down)
  - Regressor (5-day forward return)

Uses TimeSeriesSplit cross-validation and saves models with metadata.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config.settings import ML_CV_SPLITS, ML_MODEL_DIR
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from a training run."""

    model_type: str  # "classifier" or "regressor"
    timestamp: str
    n_samples: int
    n_features: int
    n_tickers: int
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    # Classifier-specific
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    class_distribution: Optional[Dict[str, int]] = None
    # Regressor-specific
    mae: Optional[float] = None


def _get_available_features(matrix: pd.DataFrame) -> List[str]:
    """Return the subset of FEATURE_COLUMNS actually present in the matrix."""
    return [c for c in FEATURE_COLUMNS if c in matrix.columns]


def train_classifier(
    matrix: pd.DataFrame,
    n_splits: int = ML_CV_SPLITS,
) -> Tuple[LGBMClassifier, TrainingResult]:
    """Train a LightGBM 3-class classifier with TimeSeriesSplit CV.

    Args:
        matrix: Feature matrix from build_feature_matrix() with 'label' column.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        Tuple of (trained model, TrainingResult).
    """
    features = _get_available_features(matrix)
    X = matrix[features].values
    y = matrix["label"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            verbose=-1,
            random_state=42,
        )
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)
        cv_scores.append(f1_score(y_val, y_pred, average="macro"))

    # Train final model on all data
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=3,
        verbose=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Feature importance
    importance = dict(zip(features, model.feature_importances_.tolist()))

    # Full-data metrics
    y_pred_full = model.predict(X)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    result = TrainingResult(
        model_type="classifier",
        timestamp=datetime.now().isoformat(),
        n_samples=len(X),
        n_features=len(features),
        n_tickers=matrix["ticker"].nunique(),
        cv_scores=cv_scores,
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        feature_importance=importance,
        accuracy=float(accuracy_score(y, y_pred_full)),
        f1_macro=float(f1_score(y, y_pred_full, average="macro")),
        class_distribution=class_dist,
    )

    logger.info(
        "Classifier trained: CV F1=%.3f±%.3f, accuracy=%.3f",
        result.cv_mean,
        result.cv_std,
        result.accuracy,
    )
    return model, result


def train_regressor(
    matrix: pd.DataFrame,
    n_splits: int = ML_CV_SPLITS,
) -> Tuple[LGBMRegressor, TrainingResult]:
    """Train a LightGBM regressor for 5-day return prediction.

    Args:
        matrix: Feature matrix from build_feature_matrix() with 'fwd_return' column.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        Tuple of (trained model, TrainingResult).
    """
    features = _get_available_features(matrix)
    X = matrix[features].values
    y = matrix["fwd_return"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
        )
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)
        # Use negative MAE so higher is better (consistent with sklearn convention)
        cv_scores.append(-mean_absolute_error(y_val, y_pred))

    # Train final model on all data
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Feature importance
    importance = dict(zip(features, model.feature_importances_.tolist()))

    # Full-data metrics
    y_pred_full = model.predict(X)

    result = TrainingResult(
        model_type="regressor",
        timestamp=datetime.now().isoformat(),
        n_samples=len(X),
        n_features=len(features),
        n_tickers=matrix["ticker"].nunique(),
        cv_scores=cv_scores,
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        feature_importance=importance,
        mae=float(mean_absolute_error(y, y_pred_full)),
    )

    logger.info(
        "Regressor trained: CV neg-MAE=%.4f±%.4f, full MAE=%.4f",
        result.cv_mean,
        result.cv_std,
        result.mae,
    )
    return model, result


def save_models(
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    clf_result: TrainingResult,
    reg_result: TrainingResult,
    features_used: List[str],
    model_dir: Optional[Path] = None,
) -> Path:
    """Save trained models and metadata to disk.

    Files saved:
        ml/models/classifier_YYYYMMDD.joblib
        ml/models/regressor_YYYYMMDD.joblib
        ml/models/metadata_YYYYMMDD.json

    Returns:
        Path to the metadata file.
    """
    model_dir = model_dir or ML_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    clf_path = model_dir / f"classifier_{date_str}.joblib"
    reg_path = model_dir / f"regressor_{date_str}.joblib"
    meta_path = model_dir / f"metadata_{date_str}.json"

    joblib.dump(classifier, clf_path)
    joblib.dump(regressor, reg_path)

    metadata = {
        "date": date_str,
        "features": features_used,
        "classifier": asdict(clf_result),
        "regressor": asdict(reg_result),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    logger.info("Models saved: %s, %s", clf_path.name, reg_path.name)
    return meta_path
