"""
SQLite cache for stock OHLCV data.

Provides incremental updates: only fetches data since the last cached date
for each ticker, avoiding redundant API calls.
"""

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import DB_PATH, DATE_FORMAT_DISPLAY

logger = logging.getLogger(__name__)


class OHLCVCache:
    """SQLite-backed cache for stock OHLCV data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker      TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    open        REAL,
                    high        REAL,
                    low         REAL,
                    close       REAL,
                    volume      INTEGER,
                    trading_value REAL,
                    updated_at  TEXT    NOT NULL,
                    PRIMARY KEY (ticker, date)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker
                    ON ohlcv(ticker);
                CREATE INDEX IF NOT EXISTS idx_ohlcv_date
                    ON ohlcv(date);

                CREATE TABLE IF NOT EXISTS universe (
                    ticker      TEXT    PRIMARY KEY,
                    name        TEXT,
                    updated_at  TEXT    NOT NULL
                );
            """)

            # Migrate from old schema if needed
            self._migrate_from_old_schema(conn)

            conn.commit()
        finally:
            conn.close()

        logger.debug("Database initialized at %s", self.db_path)

    def _migrate_from_old_schema(self, conn: sqlite3.Connection) -> None:
        """Migrate data from old etf_ohlcv/etf_universe tables if they exist."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='etf_ohlcv'"
        )
        if cursor.fetchone() is None:
            return

        logger.info("Migrating data from old etf_ohlcv table to ohlcv...")

        # Check if ohlcv table is empty before migrating
        count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        if count == 0:
            conn.execute("""
                INSERT OR IGNORE INTO ohlcv
                    (ticker, date, open, high, low, close, volume,
                     trading_value, updated_at)
                SELECT ticker, date, open, high, low, close, volume,
                       trading_value, updated_at
                FROM etf_ohlcv
            """)

        # Migrate universe
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='etf_universe'"
        )
        if cursor.fetchone() is not None:
            uni_count = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
            if uni_count == 0:
                conn.execute("""
                    INSERT OR IGNORE INTO universe (ticker, name, updated_at)
                    SELECT ticker, name, updated_at FROM etf_universe
                """)

        logger.info("Migration complete")

    def get_last_cached_date(self, ticker: str) -> Optional[date]:
        """Get the most recent cached date for a ticker."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker = ?",
                (ticker,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.strptime(row[0], DATE_FORMAT_DISPLAY).date()
            return None
        finally:
            conn.close()

    def upsert_ohlcv(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Insert or update OHLCV data for a ticker.

        The DataFrame should have a DatetimeIndex and columns:
        open, high, low, close, volume, trading_value.

        Returns:
            Number of rows upserted.
        """
        if df.empty:
            return 0

        now_str = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            rows = []
            for idx, row in df.iterrows():
                date_str = idx.strftime(DATE_FORMAT_DISPLAY)
                rows.append((
                    ticker,
                    date_str,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                    row.get("trading_value"),
                    now_str,
                ))

            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (ticker, date, open, high, low, close, volume,
                    trading_value, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            count = len(rows)
            logger.debug("Upserted %d rows for %s", count, ticker)
            return count
        finally:
            conn.close()

    def get_ohlcv(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Retrieve cached OHLCV data for a ticker.

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns, sorted by date.
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM ohlcv WHERE ticker = ?"
            params: list = [ticker]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime(DATE_FORMAT_DISPLAY))
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime(DATE_FORMAT_DISPLAY))

            query += " ORDER BY date ASC"

            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                return df

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.drop(columns=["ticker", "updated_at"], errors="ignore")

            return df
        finally:
            conn.close()

    def get_cached_tickers(self) -> List[str]:
        """Return list of all tickers that have cached data."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_cache_stats(self) -> dict:
        """Return summary statistics about the cache."""
        conn = self._get_connection()
        try:
            stats = {}
            cursor = conn.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv")
            stats["ticker_count"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
            stats["total_rows"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT MIN(date), MAX(date) FROM ohlcv")
            row = cursor.fetchone()
            stats["oldest_date"] = row[0]
            stats["newest_date"] = row[1]

            return stats
        finally:
            conn.close()

    def save_universe(self, tickers_with_names: List[tuple]) -> None:
        """Save the stock universe (ticker, name) pairs."""
        now_str = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO universe
                   (ticker, name, updated_at) VALUES (?, ?, ?)""",
                [(t, n, now_str) for t, n in tickers_with_names],
            )
            conn.commit()
        finally:
            conn.close()
