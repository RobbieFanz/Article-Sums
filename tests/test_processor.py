from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from article_briefing.config import Settings
from article_briefing.email_delivery import EmailDeliveryError
from article_briefing.models import Article, ArticleSummary, BriefingOverview
from article_briefing.processor import ProcessingError, process_landing_folder


class FakeSummarizer:
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

    def summarize_briefing(self, summaries: list[ArticleSummary]) -> BriefingOverview:
        return BriefingOverview(
            overview=f"{len(summaries)} article was summarized.",
            major_themes=["A major theme.", "A second theme."],
            watch_for="Watch for follow-up coverage.",
        )


class FakeSender:
    def __init__(self, source_file: Path, duration_seconds: float = 0.25) -> None:
        self.source_file = source_file
        self.duration_seconds = duration_seconds
        self.was_source_file_present_when_sent = False

    def send(self, briefing: object) -> float:
        self.was_source_file_present_when_sent = self.source_file.exists()
        return self.duration_seconds


class FailingSender:
    def send(self, briefing: object) -> float:
        raise EmailDeliveryError("SMTP service unavailable")


class ProcessorTests(unittest.TestCase):
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
            self.assertNotIn("## Articles", briefing_markdown)
            self.assertIn("article-summaries/", briefing_markdown)
            self.assertGreater(briefing["generation_duration_seconds"], 0)
            self.assertGreater(briefing["total_duration_seconds"], 0)
            self.assertEqual(briefing["email_delivery_duration_seconds"], None)
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
            self.assertTrue(invalid_file.exists())
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
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "email-failure.txt"
            source_file.write_text(
                "https://example.com/email-failure\n\nArticle text.",
                encoding="utf-8",
            )

            with self.assertRaises(EmailDeliveryError):
                process_landing_folder(settings, FakeSummarizer(), sender=FailingSender())

            self.assertTrue(source_file.exists())
            self.assertFalse(settings.archive_dir.exists())


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
