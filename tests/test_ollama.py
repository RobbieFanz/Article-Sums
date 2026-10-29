from __future__ import annotations

import unittest
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from article_briefing.config import Settings
from article_briefing.models import Article
from article_briefing.ollama import (
    OllamaClient,
    OllamaError,
    OllamaOutputError,
    key_points_or_fallback,
)
from article_briefing.processor import summarize_article_with_retry


class OllamaResponseTests(unittest.TestCase):
    @patch("article_briefing.ollama.urlopen")
    def test_logs_each_complete_response_and_summary_retry(self, urlopen: MagicMock) -> None:
        client = make_client()
        response = urlopen.return_value.__enter__.return_value
        invalid = '{"summary": "", "other": "first response"}'
        valid = json.dumps({
            "summary": "A valid summary.",
            "key_points": ["A point"],
            "why_it_matters": "It matters.",
            "read_recommendation": "Read it.",
        })
        response.read.side_effect = [
            json.dumps({"message": {"content": invalid}, "done_reason": "stop"}).encode(),
            json.dumps({"message": {"content": valid}, "done_reason": "stop"}).encode(),
        ]

        with self.assertLogs("article_briefing", level="INFO") as logs:
            summary = summarize_article_with_retry(client, make_article())

        output = "\n".join(logs.output)
        self.assertIn("Summarizing article.txt (attempt 1/2)", output)
        self.assertIn("Summary attempt 1/2 failed", output)
        self.assertIn("Summarizing article.txt (attempt 2/2)", output)
        self.assertIn(invalid, output)
        self.assertIn(valid, output)
        self.assertIn('"done_reason": "stop"', output)
        self.assertNotIn("Example text.", output)
        self.assertEqual(summary.summary, "A valid summary.")

    def test_logs_complete_http_error_body_without_changing_reported_error(self) -> None:
        body = "a" * 550 + "end-of-error"
        error = HTTPError("http://localhost", 500, "error", {}, BytesIO(body.encode()))
        with patch("article_briefing.ollama.urlopen", side_effect=error):
            with self.assertLogs("article_briefing", level="INFO") as logs:
                with self.assertRaises(OllamaError) as raised:
                    make_client()._request_json("system", "user")
        self.assertIn(body, "\n".join(logs.output))
        self.assertNotIn("end-of-error", str(raised.exception))

    def test_missing_key_points_uses_fallback(self) -> None:
        self.assertEqual(
            key_points_or_fallback({}),
            ["No key points were found given."],
        )

    def test_empty_or_invalid_key_points_uses_fallback(self) -> None:
        self.assertEqual(
            key_points_or_fallback({"key_points": [None, "", 42]}),
            ["No key points were found given."],
        )

    def test_key_points_keeps_valid_strings_and_discards_invalid_values(self) -> None:
        self.assertEqual(
            key_points_or_fallback(
                {"key_points": [" First point ", None, "", "Second point"]}
            ),
            ["First point", "Second point"],
        )

    def test_relevance_accepts_a_valid_exclusion_without_retrying(self) -> None:
        client = make_client()
        with patch.object(
            client,
            "_request_json",
            return_value=(
                {"include": False, "reason": "The article is outside the configured topics."},
                '{"include": false, "reason": "The article is outside the configured topics."}',
            ),
        ) as request_json:
            decision = client.assess_relevance(make_article())

        self.assertFalse(decision.include)
        self.assertEqual(
            decision.reason, "The article is outside the configured topics."
        )
        request_json.assert_called_once()

    def test_relevance_preserves_raw_output_for_invalid_schema(self) -> None:
        client = make_client()
        raw_output = '{"include": "yes", "reason": "Invalid boolean"}'
        with patch.object(
            client,
            "_request_json",
            return_value=(
                {"include": "yes", "reason": "Invalid boolean"},
                raw_output,
            ),
        ):
            with self.assertRaises(OllamaOutputError) as error:
                client.assess_relevance(make_article())

        self.assertEqual(error.exception.raw_output, raw_output)

    @patch("article_briefing.ollama.urlopen", side_effect=TimeoutError())
    def test_timeout_is_reported_as_an_ollama_error(self, _: object) -> None:
        client = make_client()

        with self.assertRaisesRegex(OllamaError, "Ollama timed out"):
            client._request_json("System prompt", "User prompt")


def make_client() -> OllamaClient:
    root = Path("/tmp/article-briefing-test")
    return OllamaClient(
        Settings(
            project_root=root,
            ollama_model="test-model",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_timeout_seconds=30,
            landing_dir=root / "landing",
            archive_dir=root / "archive",
            briefing_dir=root / "briefings",
            max_article_characters=24000,
        )
    )


def make_article() -> Article:
    return Article(
        source_file="article.txt",
        title="Example Article",
        url="https://example.com/article",
        text="Example text.",
        was_truncated=False,
    )


if __name__ == "__main__":
    unittest.main()
