from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from article_briefing.__main__ import main
from article_briefing.config import ConfigurationError, SmtpSettings


class MainFailureTests(unittest.TestCase):
    def test_command_logs_relevance_and_both_failed_summary_responses(self) -> None:
        class FakeOllamaHandler(BaseHTTPRequestHandler):
            calls = 0

            def do_POST(self) -> None:
                type(self).calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                content = (
                    {"include": True, "reason": "Relevant"}
                    if self.calls == 1
                    else {"summary": "", "attempt_marker": f"response-{self.calls}"}
                )
                body = json.dumps({
                    "message": {"content": json.dumps(content)},
                    "done_reason": "stop",
                }).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                landing = root / "landing"
                landing.mkdir()
                article = landing / "test.txt"
                article.write_text("Private article text.", encoding="utf-8")
                criteria = root / "criteria.txt"
                criteria.write_text("Technology", encoding="utf-8")
                environment = os.environ.copy()
                environment.update({
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                    "SMTP_ENABLED": "false",
                    "OLLAMA_MODEL": "test-model",
                    "OLLAMA_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    "LANDING_DIR": str(landing),
                    "ARCHIVE_DIR": str(root / "archive"),
                    "BRIEFING_DIR": str(root / "briefings"),
                    "RELEVANCE_CRITERIA_FILE": str(criteria),
                })
                result = subprocess.run(
                    [sys.executable, "-m", "article_briefing"],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Checking relevance of test.txt (attempt 1/3)", result.stderr)
                self.assertIn("Summarizing test.txt (attempt 1/2)", result.stderr)
                self.assertIn("Summary attempt 1/2 failed", result.stderr)
                self.assertIn("Summarizing test.txt (attempt 2/2)", result.stderr)
                self.assertIn('"attempt_marker": "response-2"', result.stderr)
                self.assertIn('"attempt_marker": "response-3"', result.stderr)
                self.assertIn('"done_reason": "stop"', result.stderr)
                self.assertNotIn("Private article text.", result.stderr)
                self.assertIn("Articles included: 0 | not included: 1", result.stdout)
                self.assertTrue(article.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

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
