"""Shared test fixtures.

Provides a mock watchlist so tests don't need network access to KRX API.
"""

import pytest

from config.watchlist import StockCategory

# Test watchlist with 3 categories and known tickers
MOCK_WATCHLIST = {
    "electronics": StockCategory(
        name_kr="전기전자",
        name_en="Electronics",
        stocks=[
            ("005930", "삼성전자"),
            ("000660", "SK하이닉스"),
            ("066570", "LG전자"),
        ],
    ),
    "chemical": StockCategory(
        name_kr="화학",
        name_en="Chemical",
        stocks=[
            ("051910", "LG화학"),
            ("006400", "삼성SDI"),
        ],
    ),
    "finance": StockCategory(
        name_kr="금융업",
        name_en="Finance",
        stocks=[
            ("105560", "KB금융"),
            ("055550", "신한지주"),
        ],
    ),
}


@pytest.fixture(autouse=True)
def _mock_watchlist(monkeypatch):
    """Automatically provide a mock watchlist for all tests.

    This patches the module-level cache so that build_watchlist() / _get_watchlist()
    return test data without hitting the KRX API.
    """
    import config.watchlist as wl_mod

    monkeypatch.setattr(wl_mod, "_watchlist_cache", MOCK_WATCHLIST)
