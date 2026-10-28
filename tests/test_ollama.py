from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from article_briefing.config import Settings
from article_briefing.models import Article
from article_briefing.ollama import (
    OllamaClient,
    OllamaError,
    OllamaOutputError,
    key_points_or_fallback,
)


class OllamaResponseTests(unittest.TestCase):
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
