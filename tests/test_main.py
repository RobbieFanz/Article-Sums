from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from article_briefing.__main__ import main
from article_briefing.config import ConfigurationError, SmtpSettings


class MainFailureTests(unittest.TestCase):
    @patch("article_briefing.__main__.load_dotenv")
    @patch("article_briefing.__main__.SmtpSettings.from_environment")
    @patch("article_briefing.__main__.SmtpBriefingSender")
    @patch("article_briefing.__main__.Settings.from_loaded_environment")
    def test_configuration_failure_sends_notification_when_smtp_is_available(
        self,
        settings_factory: MagicMock,
        sender_factory: MagicMock,
        smtp_factory: MagicMock,
        load_dotenv: MagicMock,
    ) -> None:
        smtp_factory.return_value = SmtpSettings(enabled=True)
        settings_factory.side_effect = ConfigurationError("model not configured")
        with patch("sys.stderr"), self.assertRaises(SystemExit) as result:
            main()

        self.assertEqual(result.exception.code, 1)
        load_dotenv.assert_called_once()
        failure = sender_factory.return_value.send_failure.call_args.args[0]
        self.assertEqual(failure.error, "ConfigurationError: model not configured")
        self.assertEqual(failure.processed_article_count, 0)
        self.assertIsNone(failure.current_article)
        self.assertGreaterEqual(failure.duration_seconds, 0)


if __name__ == "__main__":
    unittest.main()
