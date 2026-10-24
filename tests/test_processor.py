from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from article_briefing.config import Settings
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

    def test_invalid_file_is_reported_and_remains_in_landing_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            source_file = settings.landing_dir / "invalid.txt"
            source_file.write_text("Not a URL\n\nArticle text", encoding="utf-8")

            result = process_landing_folder(settings, FakeSummarizer())

            self.assertIsNone(result.briefing_path)
            self.assertEqual(
                result.warnings,
                ("invalid.txt must start with a complete http or https article URL.",),
            )
            self.assertTrue(source_file.exists())
            self.assertFalse(settings.archive_dir.exists())

    def test_invalid_file_does_not_block_valid_article(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            settings.landing_dir.mkdir(parents=True)
            invalid_file = settings.landing_dir / "invalid.txt"
            invalid_file.write_text("Missing article URL", encoding="utf-8")
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
            self.assertEqual(len(result.warnings), 1)
            briefing = json.loads((result.briefing_path / "briefing.json").read_text())
            self.assertEqual(briefing["skipped_files"], list(result.warnings))


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
