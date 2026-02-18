"""Tests for notification/formatter.py"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from notification.formatter import (
    format_daily_report,
    format_stock_detail,
    format_model_retrain_report,
    format_sector_report,
    format_urgent_signal,
)


@dataclass
class FakeScreeningResult:
    ticker: str = "069500"
    name: str = "KODEX 200"
    category: str = "market"
    close: float = 35000.0
    technical_score: float = 72.5
    ml_predicted_return: Optional[float] = 0.015
    ml_label: Optional[str] = "up"
    sector_momentum_rank: int = 1
    relative_strength_score: float = 0.025
    composite_score: float = 68.3
    composite_rank: int = 1


@dataclass
class FakeSignal:
    name: str = "RSI"
    signal_type: str = "buy"
    score: int = 10
    description: str = "RSI bounced from oversold"


@dataclass
class FakeCompositeSignal:
    ticker: str = "069500"
    date: str = "2025-01-15"
    close: float = 35000.0
    total_score: int = 45
    signal_type: str = "buy"
    signals: list = field(default_factory=lambda: [
        FakeSignal(name="RSI", score=10, description="Oversold bounce"),
        FakeSignal(name="MACD", score=25, description="Bullish cross"),
        FakeSignal(name="Volume", score=0, description="Normal"),
    ])


@dataclass
class FakePrediction:
    ticker: str = "069500"
    predicted_label: str = "up"
    predicted_return: float = 0.023
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {"up": 0.65, "flat": 0.25, "down": 0.10}
    )
    confidence: str = "high"
    confidence_score: float = 0.65


@dataclass
class FakeSectorMomentum:
    category: str = "sector"
    name_kr: str = "섹터"
    momentum_5d: float = 0.015
    momentum_20d: float = 0.032
    momentum_60d: float = 0.078
    rank_5d: int = 1
    rank_20d: int = 1
    rank_60d: int = 2
    rank_change_20d: int = 1
    top_stock: str = "삼성전자"


class TestFormatDailyReport:
    def test_basic_output(self):
        results = [FakeScreeningResult(composite_rank=i) for i in range(1, 6)]
        text = format_daily_report(results, top_n=3)
        assert "<b>Daily Report</b>" in text
        assert "KODEX 200" in text
        assert "Top 3 Buy Signals" in text

    def test_empty_results(self):
        text = format_daily_report([], top_n=10)
        assert "No screening results" in text

    def test_html_tags_present(self):
        results = [FakeScreeningResult()]
        text = format_daily_report(results)
        assert "<b>" in text
        assert "<i>" in text

    def test_ml_data_shown(self):
        results = [FakeScreeningResult(ml_predicted_return=0.03, ml_label="up")]
        text = format_daily_report(results)
        assert "ML" in text
        assert "+3.0%" in text

    def test_no_ml_data(self):
        results = [FakeScreeningResult(ml_predicted_return=None, ml_label=None)]
        text = format_daily_report(results)
        assert "ML" not in text

    def test_sell_candidates_shown(self):
        results = [
            FakeScreeningResult(composite_rank=i, composite_score=100 - i)
            for i in range(1, 25)
        ]
        text = format_daily_report(results, top_n=5)
        assert "Bottom 5" in text


class TestFormatUrgentSignal:
    def test_strong_buy(self):
        result = FakeScreeningResult(technical_score=90)
        text = format_urgent_signal(result)
        assert "STRONG BUY" in text
        assert "069500" in text

    def test_strong_sell(self):
        result = FakeScreeningResult(technical_score=10)
        text = format_urgent_signal(result)
        assert "STRONG SELL" in text

    def test_with_signal_details(self):
        result = FakeScreeningResult(technical_score=90)
        signal = FakeCompositeSignal()
        text = format_urgent_signal(result, signal=signal)
        assert "Signal Details" in text
        assert "RSI" in text

    def test_with_prediction(self):
        result = FakeScreeningResult(technical_score=90)
        prediction = FakePrediction()
        text = format_urgent_signal(result, prediction=prediction)
        assert "ML Prediction" in text
        assert "상승" in text
        assert "+2.30%" in text


class TestFormatSectorReport:
    def test_basic_output(self):
        sectors = [FakeSectorMomentum()]
        text = format_sector_report(sectors)
        assert "Sector Momentum Report" in text
        assert "섹터" in text

    def test_empty_sectors(self):
        text = format_sector_report([])
        assert "No sector data" in text

    def test_rank_change_shown(self):
        sectors = [FakeSectorMomentum(rank_change_20d=2)]
        text = format_sector_report(sectors)
        assert "(+2)" in text

    def test_negative_rank_change(self):
        sectors = [FakeSectorMomentum(rank_change_20d=-1)]
        text = format_sector_report(sectors)
        assert "(-1)" in text

    def test_momentum_values(self):
        sectors = [FakeSectorMomentum(momentum_20d=0.032)]
        text = format_sector_report(sectors)
        assert "+3.20%" in text


class TestFormatStockDetail:
    def test_basic_output(self):
        result = FakeScreeningResult()
        text = format_stock_detail(result)
        assert "KODEX 200" in text
        assert "069500" in text
        assert "35,000" in text

    def test_with_signal(self):
        result = FakeScreeningResult()
        signal = FakeCompositeSignal()
        text = format_stock_detail(result, signal=signal)
        assert "매수" in text
        assert "RSI" in text

    def test_with_prediction(self):
        result = FakeScreeningResult()
        prediction = FakePrediction()
        text = format_stock_detail(result, prediction=prediction)
        assert "ML Prediction" in text
        assert "P(Up)" in text

    def test_all_scores_shown(self):
        result = FakeScreeningResult()
        text = format_stock_detail(result)
        assert "Composite" in text
        assert "Technical" in text
        assert "Sector Rank" in text
        assert "RS Score" in text


class TestFormatModelRetrainReport:
    def test_monthly_trigger(self):
        text = format_model_retrain_report(0.45, 0.018, 500, "monthly")
        assert "Retrain Complete" in text
        assert "월간 정기 재학습" in text
        assert "500" in text
        assert "0.450" in text
        assert "0.0180" in text

    def test_performance_degradation_trigger(self):
        text = format_model_retrain_report(0.35, 0.02, 300, "performance_degradation")
        assert "성능 저하 감지" in text

    def test_html_format(self):
        text = format_model_retrain_report(0.4, 0.015, 100, "monthly")
        assert "<b>" in text
        assert "<i>" in text
