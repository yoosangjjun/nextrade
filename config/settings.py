"""Global settings for stock trading signal system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "nextrade.db"
LOG_DIR = BASE_DIR / "logs"
ML_MODEL_DIR = BASE_DIR / "ml" / "models"

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Date Formats ---
DATE_FORMAT_KRX = "%Y%m%d"  # pykrx expects "YYYYMMDD"
DATE_FORMAT_DISPLAY = "%Y-%m-%d"

# --- Data Collection ---
DEFAULT_LOOKBACK_DAYS = 365
COLLECTION_DELAY_SEC = 1.0
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5.0

# --- ML Model ---
ML_TRAIN_MIN_ROWS = 60  # Minimum rows per ticker after feature calculation
ML_PREDICTION_HORIZON = 5  # Predict N trading days ahead
ML_LABEL_THRESHOLD = 0.02  # ±2% for 3-class labeling (up/flat/down)
MARKET_BENCHMARK_INDEX = "1028"  # KOSPI200 index code for pykrx
MARKET_BENCHMARK_TICKER = "IDX_1028"  # Internal ticker for cached benchmark data
ML_CV_SPLITS = 5  # TimeSeriesSplit folds
ML_CONFIDENCE_THRESHOLDS = {
    "high": 0.6,
    "medium": 0.45,
}

# --- Scheduler ---
SCHEDULER_TIMEZONE = "Asia/Seoul"
MODEL_PERF_DEGRADATION_RATIO = 0.5  # Retrain if recent F1 < CV F1 * this
MODEL_PERF_MIN_RECENT_DAYS = 20  # Minimum days for performance check

# --- pykrx OHLCV Column Mapping (Korean -> English) ---
OHLCV_COLUMN_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "trading_value",
    "등락률": "change_rate",
}

# --- pykrx Index OHLCV Column Mapping ---
INDEX_OHLCV_COLUMN_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "trading_value",
}
