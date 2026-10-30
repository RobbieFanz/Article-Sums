from __future__ import annotations

import unittest
import json
from dataclasses import replace
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
    split_article_text,
)
from article_briefing.processor import summarize_article_with_retry


class OllamaResponseTests(unittest.TestCase):
    def test_split_preserves_text_with_minimum_number_of_chunks(self) -> None:
        text = "a" * 7 + "\n\n" + "b" * 11
        chunks = split_article_text(text, 10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))

    def test_chunking_summarizes_all_chunks_then_combines_them(self) -> None:
        client = make_client(chunked=True, chunk_size=10)
        article = replace(make_article(), text="abcdefghijABCDEFGHIJ12345")
        final = {
            "summary": "Whole article.",
            "key_points": ["One point."],
            "why_it_matters": "It matters.",
            "read_recommendation": "Read it.",
        }
        responses = [
            ({"summary": value}, "") for value in ("1", "2", "3")
        ] + [(final, "")]
        with patch.object(client, "_request_json", side_effect=responses) as request:
            result = client.summarize_article(article)

        self.assertEqual(request.call_count, 4)
        prompts = [call.kwargs["user_prompt"] for call in request.call_args_list]
        for chunk, prompt in zip(("abcdefghij", "ABCDEFGHIJ", "12345"), prompts):
            self.assertIn(chunk, prompt)
        self.assertIn("1\n\n2\n\n3", prompts[-1])
        self.assertEqual(result.summary, "Whole article.")
        self.assertIn("summarized in chunks", result.input_note)

    def test_short_article_and_disabled_chunking_use_one_request(self) -> None:
        final = {
            "summary": "Whole article.",
            "key_points": ["One point."],
            "why_it_matters": "It matters.",
            "read_recommendation": "Read it.",
        }
        for enabled, text in ((True, "short"), (False, "x" * 25)):
            with self.subTest(enabled=enabled):
                client = make_client(chunked=enabled, chunk_size=10)
                with patch.object(client, "_request_json", return_value=(final, "")) as request:
                    summary = client.summarize_article(replace(make_article(), text=text))
                request.assert_called_once()
                self.assertIn(text, request.call_args.kwargs["user_prompt"])
                self.assertNotIn("summarized in chunks", summary.input_note)

    def test_large_intermediate_summaries_are_reduced_before_final_request(self) -> None:
        client = make_client(chunked=True, chunk_size=15)
        final = {
            "summary": "Combined.",
            "why_it_matters": "It matters.",
            "read_recommendation": "Read it.",
        }
        responses = [
            ({"summary": value}, "") for value in ("aaaaaa", "bbbbbb", "cccccc", "merged")
        ] + [(final, "")]
        with patch.object(client, "_request_json", side_effect=responses) as request:
            client.summarize_article(replace(make_article(), text="x" * 31))
        self.assertEqual(request.call_count, 5)
        self.assertIn("aaaaaa\n\nbbbbbb", request.call_args_list[3].kwargs["user_prompt"])
        self.assertIn("merged\n\ncccccc", request.call_args_list[4].kwargs["user_prompt"])

    def test_oversized_intermediate_summary_fails_explicitly(self) -> None:
        client = make_client(chunked=True, chunk_size=10)
        responses = [({"summary": "x" * 11}, ""), ({"summary": "other"}, "")]
        with patch.object(client, "_request_json", side_effect=responses):
            with self.assertRaisesRegex(OllamaError, "exceeds SUMMARY_CHUNK_CHARACTERS"):
                client.summarize_article(replace(make_article(), text="x" * 11))

    def test_invalid_chunk_output_fails_instead_of_combining_partial_results(self) -> None:
        client = make_client(chunked=True, chunk_size=10)
        with patch.object(client, "_request_json", return_value=({}, "")) as request:
            with self.assertRaisesRegex(OllamaError, "'summary'"):
                client.summarize_article(replace(make_article(), text="x" * 11))
        request.assert_called_once()

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


def make_client(*, chunked: bool = False, chunk_size: int = 100000) -> OllamaClient:
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
            chunked_summarization=chunked,
            summary_chunk_characters=chunk_size,
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
