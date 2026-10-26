from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from article_briefing.config import ConfigurationError, SmtpSettings


class SmtpSettingsTests(unittest.TestCase):
    def test_rejects_starttls_and_implicit_ssl_together(self) -> None:
        environment = {
            "SMTP_ENABLED": "true",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "reader@example.com",
            "SMTP_PASSWORD": "app-password",
            "SMTP_FROM": "reader@example.com",
            "SMTP_TO": "reader@example.com",
            "SMTP_USE_STARTTLS": "true",
            "SMTP_USE_SSL": "true",
        }

        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(
                ConfigurationError, "cannot both be true"
            ):
                SmtpSettings.from_environment()


if __name__ == "__main__":
    unittest.main()
