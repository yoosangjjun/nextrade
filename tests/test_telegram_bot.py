"""Tests for notification/telegram_bot.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from notification.telegram_bot import build_telegram_app


class TestBuildTelegramApp:
    def test_raises_without_token(self):
        with patch("notification.telegram_bot.TELEGRAM_BOT_TOKEN", ""):
            with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
                build_telegram_app()

    def test_builds_with_token(self):
        with patch("notification.telegram_bot.TELEGRAM_BOT_TOKEN", "test:token123"):
            app = build_telegram_app()
            # Should have 4 handlers registered (today, top10, sector, stock)
            # CommandHandler instances are in app.handlers[0]
            handlers = app.handlers.get(0, [])
            assert len(handlers) == 4

    def test_command_names(self):
        with patch("notification.telegram_bot.TELEGRAM_BOT_TOKEN", "test:token123"):
            app = build_telegram_app()
            handlers = app.handlers.get(0, [])
            commands = set()
            for h in handlers:
                commands.update(h.commands)
            assert "today" in commands
            assert "top10" in commands
            assert "sector" in commands
            assert "stock" in commands
