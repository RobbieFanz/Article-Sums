from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from article_briefing.config import Settings
from article_briefing.email_delivery import EmailDeliveryError, render_html, render_plain_text
from article_briefing.models import (
    Article,
    ArticleSummary,
    Briefing,
    FailureReport,
    RelevanceDecision,
)
from article_briefing.ollama import OllamaClient, OllamaError, OllamaOutputError
from article_briefing.processor import ProcessingError, process_landing_folder


class FakeSummarizer:
    def assess_relevance(self, article: Article) -> RelevanceDecision:
        return RelevanceDecision(
            include=True,
            reason="Relevant to the configured briefing.",
            duration_seconds=0.0,
        )

    def summarize_article(self, article: Article) -> ArticleSummary:
        return ArticleSummary(
            title=article.title,
            url=article.url,
            source_file=article.source_file,
            summary="A concise summary.",
            key_points=["First point.", "Second point."],
            why_it_matters="It is useful context.",
            read_recommendation="Read for more detail.",
            input_note="The summary is based on the supplied article text.",
        )

class FakeSender:
    def __init__(self, source_file: Path, duration_seconds: float = 0.25) -> None:
        self.source_file = source_file
        self.duration_seconds = duration_seconds
        self.was_source_file_present_when_sent = False
        self.briefing: object | None = None
        self.failures: list[FailureReport] = []

    def send(self, briefing: object) -> float:
        self.was_source_file_present_when_sent = self.source_file.exists()
        self.briefing = briefing
        return self.duration_seconds

    def send_failure(self, failure: FailureReport) -> float:
        self.failures.append(failure)
        return self.duration_seconds


class FailingSender:
    def send(self, briefing: object) -> float:
        raise EmailDeliveryError("SMTP service unavailable")

    def send_failure(self, failure: FailureReport) -> float:
        raise EmailDeliveryError("SMTP service unavailable")


class RetrySummarizer(FakeSummarizer):
    def __init__(self, failures_by_file: dict[str, int]) -> None:
        self.failures_by_file = failures_by_file
        self.calls_by_file: dict[str, int] = {}

    def summarize_article(self, article: Article) -> ArticleSummary:
        calls = self.calls_by_file.get(article.source_file, 0) + 1
        self.calls_by_file[article.source_file] = calls
        if calls <= self.failures_by_file.get(article.source_file, 0):
            raise OllamaError("Ollama response field 'summary' must be a non-empty string.")
        return super().summarize_article(article)


class RelevanceRetrySummarizer(FakeSummarizer):
    def __init__(self, invalid_response_count: int, include: bool = True) -> None:
        self.invalid_response_count = invalid_response_count
        self.include = include
        self.relevance_calls = 0
        self.summary_calls = 0

    def assess_relevance(self, article: Article) -> RelevanceDecision:
        self.relevance_calls += 1
        if self.relevance_calls <= self.invalid_response_count:
            raise OllamaOutputError(
                "Ollama response field 'include' must be a boolean.",
                f"Invalid relevance output {self.relevance_calls}",
            )
        return RelevanceDecision(
            include=self.include,
            reason="Configured relevance decision.",
            duration_seconds=0.0,
        )

    def summarize_article(self, article: Article) -> ArticleSummary:
        self.summary_calls += 1
        return super().summarize_article(article)


class ProcessorTests(unittest.TestCase):
    def test_chunk_limit_is_saved_and_rendered_in_the_briefing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                make_settings(Path(directory)),
                chunked_summarization=True,
                summary_chunk_characters=10,
                max_summary_chunks=1,
            )
            settings.landing_dir.mkdir(parents=True)
            source = settings.landing_dir / "article.txt"
            source.write_text("abcdefghijABCDEFGHIJ", encoding="utf-8")
            client = OllamaClient(settings)
            sender = FakeSender(source)
            responses = [
                ({"include": True, "reason": "Relevant."}, ""),
                ({"summary": "First"}, ""),
                ({
                    "summary": "Partial summary.",
                    "why_it_matters": "It matters.",
                    "read_recommendation": "Read it.",
                    "relevance_score": 8,
                }, ""),
            ]
            with patch.object(client, "_request_json", side_effect=responses) as request:
                result = process_landing_folder(settings, client, sender=sender)

            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(
                "ABCDEFGHIJ" not in call.kwargs["user_prompt"]
                for call in request.call_args_list
            ))
            assert result.briefing_path is not None
            payload = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertTrue(payload["articles"][0]["chunk_limit_reached"])
            markdown = (result.briefing_path / "briefing.md").read_text()
            self.assertIn("[article](<article-summaries/article.md>) \\*", markdown)
            self.assertIn("Chunk limit reached", markdown)
            individual = (result.briefing_path / "article-summaries" / "article.md").read_text()
            self.assertIn("chunk limit was reached", individual)
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            for body in (render_plain_text(sender.briefing), render_html(sender.briefing)):
                self.assertIn("article *", body)
                self.assertIn("Chunk limit reached", body)

    def test_chunk_limit_is_marked_for_relevance_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                make_settings(Path(directory)),
                chunked_summarization=True,
                summary_chunk_characters=10,
                max_summary_chunks=1,
            )
            settings.landing_dir.mkdir(parents=True)
            source = settings.landing_dir / "article.txt"
            source.write_text("abcdefghijABCDEFGHIJ", encoding="utf-8")
            client = OllamaClient(settings)
            sender = FakeSender(source)
            responses = [({"include": "invalid", "reason": "bad"}, "bad")] * 3 + [
                ({"summary": "First"}, ""),
                ({
                    "summary": "Partial summary.",
                    "why_it_matters": "It matters.",
                    "read_recommendation": "Read it.",
                    "relevance_score": 8,
                }, ""),
            ]
            with patch.object(client, "_request_json", side_effect=responses):
                result = process_landing_folder(settings, client, sender=sender)

            assert result.briefing_path is not None
            payload = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertTrue(payload["relevance_failures"][0]["summary"]["chunk_limit_reached"])
            markdown = (result.briefing_path / "briefing.md").read_text()
            self.assertIn("### article \\*", markdown)
            self.assertIn("Chunk limit reached", markdown)
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            for body in (render_plain_text(sender.briefing), render_html(sender.briefing)):
                self.assertIn("article *", body)
                self.assertIn("Chunk limit reached", body)

    def test_articles_are_sorted_by_score_in_all_outputs(self) -> None:
        class ScoredSummarizer(FakeSummarizer):
            def summarize_article(self, article: Article) -> ArticleSummary:
                scores = {"a-low.txt": 1, "b-high.txt": 10, "c-unknown.txt": None, "d-high.txt": 10}
                return replace(
                    super().summarize_article(article), relevance_score=scores[article.source_file]
                )

        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            settings.landing_dir.mkdir(parents=True)
            for name in ("a-low.txt", "b-high.txt", "c-unknown.txt", "d-high.txt"):
                (settings.landing_dir / name).write_text("Article text", encoding="utf-8")
            sender = FakeSender(settings.landing_dir / "a-low.txt")
            result = process_landing_folder(settings, ScoredSummarizer(), sender=sender)

            assert result.briefing_path is not None
            payload = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertNotIn("overview", payload)
            self.assertEqual(
                [item["source_file"] for item in payload["articles"]],
                ["b-high.txt", "d-high.txt", "a-low.txt", "c-unknown.txt"],
            )
            self.assertEqual(
                [item["relevance_score"] for item in payload["articles"]],
                [10, 10, 1, None],
            )
            markdown = (result.briefing_path / "briefing.md").read_text()
            self.assertIn("Relevance: unavailable", markdown)
            self.assertIn("Relevance: 10/10", markdown)
            self.assertNotIn("Overview", markdown)
            self.assertIn(
                "**Relevance score:** 10/10",
                (result.briefing_path / "article-summaries" / "b-high.md").read_text(),
            )
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            for body in (markdown, render_plain_text(sender.briefing), render_html(sender.briefing)):
                positions = [body.index(title) for title in ("b high", "d high", "a low", "c unknown")]
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn("Overview", body)

    def test_chunking_loads_full_article_while_disabled_mode_retains_cap(self) -> None:
        class CaptureSummarizer(FakeSummarizer):
            def __init__(self) -> None:
                self.article: Article | None = None

            def summarize_article(self, article: Article) -> ArticleSummary:
                self.article = article
                return super().summarize_article(article)

        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                settings = replace(
                    make_settings(Path(directory)),
                    max_article_characters=10,
                    chunked_summarization=enabled,
                )
                settings.landing_dir.mkdir(parents=True)
                (settings.landing_dir / "article.txt").write_text("x" * 25, encoding="utf-8")
                summarizer = CaptureSummarizer()
                process_landing_folder(settings, summarizer)
                self.assertIsNotNone(summarizer.article)
                assert summarizer.article is not None
                self.assertEqual(len(summarizer.article.text), 25 if enabled else 10)
                self.assertEqual(summarizer.article.was_truncated, not enabled)

    def test_fatal_article_failure_sends_progress_and_preserves_original_error(self) -> None:
        class FatalSummarizer(FakeSummarizer):
            def assess_relevance(self, article: Article) -> RelevanceDecision:
                if article.source_file == "b-fail.txt":
                    raise OllamaError("model unavailable")
                return super().assess_relevance(article)

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(Path(temporary_directory))
            settings.landing_dir.mkdir(parents=True)
            for name in ("a-done.txt", "b-fail.txt", "c-pending.txt"):
                (settings.landing_dir / name).write_text("Article text", encoding="utf-8")
            sender = FakeSender(settings.landing_dir / "a-done.txt")

            with self.assertRaisesRegex(OllamaError, "model unavailable"):
                process_landing_folder(settings, FatalSummarizer(), sender=sender)

            self.assertEqual(len(sender.failures), 1)
            failure = sender.failures[0]
            self.assertEqual(failure.error, "OllamaError: model unavailable")
            self.assertEqual(failure.processed_article_count, 1)
            self.assertEqual(failure.current_article, "b-fail.txt")
            self.assertGreaterEqual(failure.duration_seconds, 0)
            self.assertEqual(len(list(settings.landing_dir.glob("*.txt"))), 3)
            self.assertFalse(settings.archive_dir.exists())

    def test_failure_after_articles_has_no_current_article(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(Path(temporary_directory))
            settings.landing_dir.mkdir(parents=True)
            source = settings.landing_dir / "done.txt"
            source.write_text("Article text", encoding="utf-8")
            sender = FakeSender(source)
            with patch("article_briefing.processor.write_briefing", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    process_landing_folder(settings, FakeSummarizer(), sender=sender)
            self.assertEqual(sender.failures[0].processed_article_count, 1)
            self.assertIsNone(sender.failures[0].current_article)
            self.assertTrue(source.exists())

    def test_archive_failure_reports_failing_article_and_restores_files(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(Path(temporary_directory))
            settings.landing_dir.mkdir(parents=True)
            source = settings.landing_dir / "done.txt"
            source.write_text("Article text", encoding="utf-8")
            sender = FakeSender(source)
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                # Fail while archiving, not while writing the briefing.
                with patch("article_briefing.processor.write_briefing"):
                    with self.assertRaisesRegex(ProcessingError, "disk full"):
                        process_landing_folder(settings, FakeSummarizer(), sender=sender)
            self.assertEqual(sender.failures[0].processed_article_count, 1)
            self.assertEqual(sender.failures[0].current_article, "done.txt")
            self.assertTrue(source.exists())

    def test_failure_notification_failure_does_not_hide_original(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(Path(temporary_directory))
            settings.landing_dir.mkdir(parents=True)
            (settings.landing_dir / "article.txt").write_text("Article text", encoding="utf-8")
            with patch("sys.stderr") as stderr:
                with self.assertRaisesRegex(EmailDeliveryError, "SMTP service unavailable"):
                    process_landing_folder(settings, FakeSummarizer(), sender=FailingSender())
            self.assertIn("Could not send failure notification", "".join(
                call.args[0] for call in stderr.write.call_args_list
            ))

    def test_delivery_failure_attempts_separate_failure_notification(self) -> None:
        class RecoverySender(FakeSender):
            def send(self, briefing: object) -> float:
                raise EmailDeliveryError("first delivery failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(Path(temporary_directory))
            settings.landing_dir.mkdir(parents=True)
            source = settings.landing_dir / "article.txt"
            source.write_text("Article text", encoding="utf-8")
            sender = RecoverySender(source)
            with self.assertRaisesRegex(EmailDeliveryError, "first delivery failed"):
                process_landing_folder(settings, FakeSummarizer(), sender=sender)
            self.assertEqual(len(sender.failures), 1)
            self.assertIn("first delivery failed", sender.failures[0].error)
            self.assertEqual(sender.failures[0].processed_article_count, 1)
            self.assertIsNone(sender.failures[0].current_article)
            self.assertTrue(source.exists())

    def test_mixed_articles_keep_rejected_out_of_summaries_but_in_email(self) -> None:
        class MixedSummarizer(FakeSummarizer):
            def __init__(self) -> None:
                self.summarized_files: list[str] = []

            def assess_relevance(self, article: Article) -> RelevanceDecision:
                return RelevanceDecision(
                    include=article.source_file == "accepted.txt",
                    reason="This story is not about the requested topics.",
                    duration_seconds=0,
                )

            def summarize_article(self, article: Article) -> ArticleSummary:
                self.summarized_files.append(article.source_file)
                return super().summarize_article(article)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            accepted = settings.landing_dir / "accepted.txt"
            accepted.write_text("https://example.com/accepted\n\nAI news.", encoding="utf-8")
            rejected = settings.landing_dir / "rejected.txt"
            rejected.write_text("https://example.com/rejected\n\nTravel news.", encoding="utf-8")
            summarizer = MixedSummarizer()
            sender = FakeSender(accepted)

            result = process_landing_folder(settings, summarizer, sender=sender)

            self.assertEqual(result.included_article_count, 1)
            self.assertEqual(result.not_included_article_count, 1)
            self.assertEqual(summarizer.summarized_files, ["accepted.txt"])
            self.assertFalse(accepted.exists())
            self.assertFalse(rejected.exists())
            self.assertEqual(
                {path.name for path in settings.archive_dir.glob("*/*.txt")},
                {"accepted.txt", "rejected.txt"},
            )
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            for body in (render_plain_text(sender.briefing), render_html(sender.briefing)):
                self.assertIn("https://example.com/accepted", body)
                self.assertIn("https://example.com/rejected", body)
                self.assertIn("This story is not about the requested topics.", body)

    def test_processes_and_archives_a_valid_landing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "model-release.txt"
            source_file.write_text(
                "https://example.com/model-release\n\nA new model was released.",
                encoding="utf-8",
            )

            result = process_landing_folder(
                settings,
                FakeSummarizer(),
                now=datetime(2026, 10, 24, 12, 8, 19, tzinfo=timezone.utc),
            )

            briefing_path = result.briefing_path
            assert briefing_path is not None
            self.assertFalse(source_file.exists())
            self.assertTrue(
                (
                    settings.archive_dir
                    / "2026-10-24_120819+0000"
                    / "model-release.txt"
                ).exists()
            )
            briefing = json.loads((briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["model"], "test-model")
            self.assertEqual(briefing["articles"][0]["url"], "https://example.com/model-release")
            self.assertTrue((briefing_path / "article-summaries" / "model-release.md").exists())
            briefing_markdown = (briefing_path / "briefing.md").read_text()
            self.assertIn("## Article summaries", briefing_markdown)
            self.assertIn("article-summaries/", briefing_markdown)
            self.assertGreater(briefing["generation_duration_seconds"], 0)
            self.assertGreater(briefing["total_duration_seconds"], 0)
            self.assertEqual(briefing["email_delivery_duration_seconds"], None)
            self.assertEqual(briefing["included_article_count"], 1)
            self.assertEqual(briefing["not_included_article_count"], 0)
            self.assertIn(
                "**Summarization time:** 0.0 seconds",
                (briefing_path / "article-summaries" / "model-release.md").read_text(),
            )

    def test_processes_article_without_a_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "article-without-url.txt"
            source_file.write_text("Article text without a source URL.", encoding="utf-8")

            result = process_landing_folder(settings, FakeSummarizer())

            assert result.briefing_path is not None
            self.assertEqual(result.warnings, ())
            self.assertFalse(source_file.exists())
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertIsNone(briefing["articles"][0]["url"])
            summary_markdown = (
                result.briefing_path / "article-summaries" / "article-without-url.md"
            ).read_text()
            self.assertIn("**Original article:** unavailable", summary_markdown)

    def test_invalid_url_header_does_not_block_valid_article(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            invalid_file = settings.landing_dir / "invalid.txt"
            invalid_file.write_text("URL: not-a-url\n\nArticle text", encoding="utf-8")
            valid_file = settings.landing_dir / "valid.txt"
            valid_file.write_text(
                "https://example.com/valid\n\nValid article text.", encoding="utf-8"
            )

            result = process_landing_folder(
                settings,
                FakeSummarizer(),
                now=datetime(2026, 10, 24, 12, 8, 19, tzinfo=timezone.utc),
            )

            assert result.briefing_path is not None
            self.assertFalse(invalid_file.exists())
            self.assertEqual(len(list((settings.archive_dir / "failed").glob("*/invalid.txt"))), 1)
            self.assertFalse(valid_file.exists())
            self.assertEqual(
                result.warnings,
                (
                    "invalid.txt has an invalid URL: header. "
                    "Use a complete http or https URL.",
                ),
            )
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["skipped_files"], list(result.warnings))
            self.assertEqual(briefing["not_included_article_count"], 1)
            self.assertEqual(briefing["articles_not_selected"][0]["source_file"], "invalid.txt")
            self.assertIsNone(briefing["articles_not_selected"][0]["url"])

    def test_url_header_is_saved_as_the_original_article_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "header-url.txt"
            source_file.write_text(
                "URL: https://example.com/header-url\n\nArticle text.",
                encoding="utf-8",
            )

            result = process_landing_folder(settings, FakeSummarizer())

            assert result.briefing_path is not None
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(
                briefing["articles"][0]["url"], "https://example.com/header-url"
            )

    def test_sends_before_archiving_and_records_delivery_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "email-test.txt"
            source_file.write_text(
                "https://example.com/email-test\n\nArticle text.",
                encoding="utf-8",
            )
            sender = FakeSender(source_file)

            result = process_landing_folder(settings, FakeSummarizer(), sender=sender)

            assert result.briefing_path is not None
            self.assertTrue(sender.was_source_file_present_when_sent)
            self.assertFalse(source_file.exists())
            self.assertEqual(result.email_delivery_duration_seconds, 0.25)
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["email_delivery_duration_seconds"], 0.25)
            self.assertGreater(briefing["total_duration_seconds"], 0)

    def test_delivery_failure_keeps_source_file_in_landing_folder(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "email-failure.txt"
            source_file.write_text(
                "https://example.com/email-failure\n\nArticle text.",
                encoding="utf-8",
            )

            with patch("sys.stderr") as stderr, self.assertRaises(EmailDeliveryError):
                process_landing_folder(settings, FakeSummarizer(), sender=FailingSender())

            self.assertIn("Could not send failure notification", "".join(
                call.args[0] for call in stderr.write.call_args_list
            ))
            self.assertTrue(source_file.exists())
            self.assertFalse(settings.archive_dir.exists())

    def test_retries_an_article_once_before_succeeding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "retry.txt"
            source_file.write_text(
                "https://example.com/retry\n\nArticle text.",
                encoding="utf-8",
            )
            summarizer = RetrySummarizer({"retry.txt": 1})

            result = process_landing_folder(settings, summarizer)

            assert result.briefing_path is not None
            self.assertEqual(summarizer.calls_by_file["retry.txt"], 2)
            self.assertFalse(source_file.exists())
            self.assertEqual(result.warnings, ())

    def test_skips_failed_article_after_two_attempts_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            failed_file = settings.landing_dir / "failed.txt"
            failed_file.write_text(
                "https://example.com/failed\n\nFailed article text.",
                encoding="utf-8",
            )
            valid_file = settings.landing_dir / "valid.txt"
            valid_file.write_text(
                "https://example.com/valid\n\nValid article text.",
                encoding="utf-8",
            )
            summarizer = RetrySummarizer({"failed.txt": 2})

            result = process_landing_folder(settings, summarizer)

            assert result.briefing_path is not None
            self.assertEqual(summarizer.calls_by_file["failed.txt"], 2)
            self.assertFalse(failed_file.exists())
            self.assertEqual(len(list((settings.archive_dir / "failed").glob("*/failed.txt"))), 1)
            self.assertFalse(valid_file.exists())
            self.assertEqual(len(result.warnings), 1)
            self.assertIn("failed.txt could not be summarized after 2 attempts", result.warnings[0])
            briefing_markdown = (result.briefing_path / "briefing.md").read_text()
            self.assertIn("## Articles not included", briefing_markdown)
            self.assertIn("failed.txt could not be summarized", briefing_markdown)
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(
                briefing["articles_not_selected"][0]["url"],
                "https://example.com/failed",
            )

    def test_archives_validly_rejected_article_without_summarizing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "rejected.txt"
            source_file.write_text(
                "https://example.com/rejected\n\nArticle text.",
                encoding="utf-8",
            )
            summarizer = RelevanceRetrySummarizer(0, include=False)
            sender = FakeSender(source_file)

            result = process_landing_folder(settings, summarizer, sender=sender)

            assert result.briefing_path is not None
            self.assertEqual(summarizer.relevance_calls, 1)
            self.assertEqual(summarizer.summary_calls, 0)
            self.assertFalse(source_file.exists())
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["included_article_count"], 0)
            self.assertEqual(briefing["not_included_article_count"], 1)
            self.assertEqual(briefing["articles"], [])
            self.assertEqual(
                briefing["articles_not_selected"],
                [{
                    "source_file": "rejected.txt",
                    "title": "rejected",
                    "url": "https://example.com/rejected",
                    "reason": "Configured relevance decision.",
                }],
            )
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            text = render_plain_text(sender.briefing)
            html = render_html(sender.briefing)
            for body in (text, html):
                self.assertIn("https://example.com/rejected", body)
                self.assertIn("Configured relevance decision.", body)
                self.assertIn("rejected", body)
            self.assertIn(
                "Configured relevance decision.",
                (result.briefing_path / "briefing.md").read_text(),
            )

    def test_all_invalid_files_produce_a_report_and_are_archived_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "empty.txt"
            source_file.write_text("", encoding="utf-8")
            sender = FakeSender(source_file)

            result = process_landing_folder(settings, FakeSummarizer(), sender=sender)

            self.assertIsNotNone(result.briefing_path)
            self.assertFalse(source_file.exists())
            self.assertEqual(len(list((settings.archive_dir / "failed").glob("*/empty.txt"))), 1)
            self.assertEqual(result.not_included_article_count, 1)
            self.assertIsInstance(sender.briefing, Briefing)
            assert isinstance(sender.briefing, Briefing)
            body = render_plain_text(sender.briefing)
            self.assertIn("empty.txt is empty", body)
            self.assertIn("Original article: unavailable", body)

    def test_invalid_relevance_output_three_times_adds_flagged_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "triage-failure.txt"
            source_file.write_text(
                "https://example.com/triage-failure\n\nArticle text.",
                encoding="utf-8",
            )
            summarizer = RelevanceRetrySummarizer(3)
            sender = FakeSender(source_file)

            result = process_landing_folder(settings, summarizer, sender=sender)

            assert result.briefing_path is not None
            self.assertEqual(summarizer.relevance_calls, 3)
            self.assertEqual(summarizer.summary_calls, 1)
            self.assertFalse(source_file.exists())
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["included_article_count"], 0)
            self.assertEqual(briefing["not_included_article_count"], 1)
            self.assertEqual(
                briefing["relevance_failures"][0]["invalid_outputs"],
                [
                    "Invalid relevance output 1",
                    "Invalid relevance output 2",
                    "Invalid relevance output 3",
                ],
            )
            assert sender.briefing is not None
            self.assertEqual(getattr(sender.briefing, "relevance_failures")[0].summary.title, "triage failure")


def make_settings(root: Path) -> Settings:
    return Settings(
        project_root=root,
        ollama_model="test-model",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_timeout_seconds=30,
        landing_dir=root / "data" / "landing",
        archive_dir=root / "data" / "archive",
        briefing_dir=root / "data" / "briefings",
        max_article_characters=24000,
    )


if __name__ == "__main__":
    unittest.main()
