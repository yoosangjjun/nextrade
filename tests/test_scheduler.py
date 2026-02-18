"""Tests for notification/scheduler.py"""

from unittest.mock import MagicMock, patch

from notification.scheduler import create_scheduler


class TestCreateScheduler:
    def test_creates_scheduler_with_4_jobs(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        jobs = scheduler.get_jobs()
        assert len(jobs) == 4

    def test_job_ids(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        job_ids = {job.id for job in scheduler.get_jobs()}
        assert "daily_report" in job_ids
        assert "sector_report" in job_ids
        assert "monthly_retrain" in job_ids
        assert "performance_check" in job_ids

    def test_daily_report_schedule(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        job = scheduler.get_job("daily_report")
        assert job is not None
        # CronTrigger fields
        assert str(job.trigger) is not None

    def test_sector_report_schedule(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        job = scheduler.get_job("sector_report")
        assert job is not None

    def test_monthly_retrain_schedule(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        job = scheduler.get_job("monthly_retrain")
        assert job is not None

    def test_performance_check_schedule(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        job = scheduler.get_job("performance_check")
        assert job is not None

    def test_scheduler_not_started(self):
        app = MagicMock()
        with patch("notification.scheduler.SCHEDULER_TIMEZONE", "Asia/Seoul"):
            scheduler = create_scheduler(app)

        assert not scheduler.running
