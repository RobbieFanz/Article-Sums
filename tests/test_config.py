from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from article_briefing.config import ConfigurationError, Settings, SmtpSettings


class ChunkSettingsTests(unittest.TestCase):
    def test_chunk_settings_are_parsed_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "criteria.txt").write_text("Relevant", encoding="utf-8")
            environment = {
                "OLLAMA_MODEL": "test",
                "RELEVANCE_CRITERIA_FILE": "criteria.txt",
                "CHUNKED_SUMMARIZATION": "true",
                "SUMMARY_CHUNK_CHARACTERS": "32000",
                "MAX_SUMMARY_CHUNKS": "3",
                "ARTICLE_HISTORY_PATH": "other/history.sqlite3",
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = Settings.from_loaded_environment(root)
                self.assertTrue(settings.chunked_summarization)
                self.assertEqual(settings.summary_chunk_characters, 32000)
                self.assertEqual(settings.max_summary_chunks, 3)
                self.assertEqual(settings.history_path, root / "other" / "history.sqlite3")

            for value in ("0", "999999999999999999999999999"):
                with self.subTest(value=value):
                    with patch.dict(os.environ, {**environment, "MAX_SUMMARY_CHUNKS": value}, clear=True):
                        self.assertEqual(Settings.from_loaded_environment(root).max_summary_chunks, int(value))

            with patch.dict(os.environ, {"OLLAMA_MODEL": "test", "RELEVANCE_CRITERIA_FILE": "criteria.txt"}, clear=True):
                self.assertEqual(Settings.from_loaded_environment(root).max_summary_chunks, 10)

            for variable, value in (
                ("CHUNKED_SUMMARIZATION", "maybe"),
                ("SUMMARY_CHUNK_CHARACTERS", "0"),
                ("MAX_SUMMARY_CHUNKS", "-1"),
                ("MAX_SUMMARY_CHUNKS", "invalid"),
                ("MAX_SUMMARY_CHUNKS", "1.5"),
            ):
                with self.subTest(variable=variable):
                    with patch.dict(os.environ, {**environment, variable: value}, clear=True):
                        with self.assertRaisesRegex(ConfigurationError, variable):
                            Settings.from_loaded_environment(root)


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
