from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from article_briefing.config import Settings
from article_briefing.email_delivery import EmailDeliveryError
from article_briefing.hacker_news import HackerNewsStory
from article_briefing.hacker_news_collect import (
    CollectionStatus,
    DownloadedPage,
    HackerNewsArticleCollector,
)
from article_briefing.history import ArticleHistory, HistoryError
from article_briefing.models import Article, ArticleSummary, RelevanceDecision
from article_briefing.ollama import OllamaError, OllamaOutputError
from article_briefing.processor import ProcessingError, process_landing_folder


ORIGINAL = "https://example.com/redirect"
FINAL = "https://example.com/article"


class FakeClient:
    def top_stories(self, limit: int) -> list[HackerNewsStory]:
        return []


class FakeSummarizer:
    def __init__(self, relevance: bool | None, fail_summary: bool = False) -> None:
        self.relevance = relevance
        self.fail_summary = fail_summary

    def assess_relevance(self, article: Article) -> RelevanceDecision:
        if self.relevance is None:
            raise OllamaOutputError("Invalid relevance", "invalid")
        return RelevanceDecision(self.relevance, "Decision", 0.0)

    def summarize_article(self, article: Article) -> ArticleSummary:
        if self.fail_summary:
            raise OllamaError("Summary failed")
        return ArticleSummary(
            title=article.title, url=article.url, source_file=article.source_file,
            summary="Private summary", key_points=["Point"], why_it_matters="Reason",
            read_recommendation="Read", input_note="Input",
        )


def story(url: str, identifier: int = 1) -> HackerNewsStory:
    return HackerNewsStory(
        id=identifier, title="Example", score=1, descendants=0, by="author",
        url=url, story_type="Linked article",
    )


def settings(root: Path) -> Settings:
    return Settings(
        project_root=root, ollama_model="test", ollama_base_url="http://localhost",
        ollama_timeout_seconds=1, landing_dir=root / "data" / "landing",
        archive_dir=root / "data" / "archive", briefing_dir=root / "data" / "briefings",
        max_article_characters=1000,
    )


class ArticleHistoryIntegrationTests(unittest.TestCase):
    def make_collector(self, root: Path, downloader: Mock) -> HackerNewsArticleCollector:
        (root / "data" / "landing").mkdir(parents=True, exist_ok=True)
        return HackerNewsArticleCollector(
            client=FakeClient(), landing_dir=root / "data" / "landing",
            timeout_seconds=1, minimum_text_characters=1,
            page_downloader=downloader, article_extractor=lambda html, url: "Private article text",
            history_path=root / "data" / "article-history.sqlite3",
        )

    def test_collection_and_processing_track_both_links_and_relevance(self) -> None:
        for relevance in (True, False, None):
            with self.subTest(relevance=relevance), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                downloader = Mock(return_value=DownloadedPage(FINAL, "<html>text</html>"))
                collector = self.make_collector(root, downloader)
                collected = collector.collect_story(story(ORIGINAL))
                self.assertEqual(collected.status, CollectionStatus.CREATED)
                self.assertEqual(collector.collect_story(story(ORIGINAL, 2)).status, CollectionStatus.SKIPPED)
                self.assertEqual(collector.collect_story(story(FINAL, 3)).status, CollectionStatus.SKIPPED)
                self.assertEqual(downloader.call_count, 1)

                process_landing_folder(settings(root), FakeSummarizer(relevance))
                self.assertEqual(collector.collect_story(story(ORIGINAL, 4)).status, CollectionStatus.SKIPPED)
                self.assertEqual(collector.collect_story(story(FINAL, 5)).status, CollectionStatus.SKIPPED)
                self.assertEqual(downloader.call_count, 1)
                with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                    row = connection.execute(
                        "SELECT status, relevance, landing_path, processed_at FROM articles"
                    ).fetchone()
                    urls = {row[0] for row in connection.execute("SELECT url FROM article_urls")}
                    columns = {row[1] for row in connection.execute("PRAGMA table_info(articles)")}
                self.assertEqual(row[:3], ("processed", relevance, None))
                self.assertIsNotNone(row[3])
                self.assertEqual(urls, {ORIGINAL, FINAL})
                self.assertNotIn("summary", columns)
                self.assertNotIn("text", columns)
                self.assertEqual(len(list((root / "data" / "archive").glob("*/*.txt"))), 1)

    def test_missing_pending_file_is_eligible_for_collection_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloader = Mock(return_value=DownloadedPage(FINAL, "html"))
            collector = self.make_collector(root, downloader)
            first = collector.collect_story(story(ORIGINAL))
            assert first.landing_file is not None
            first.landing_file.unlink()
            history = ArticleHistory(root / "data" / "article-history.sqlite3")
            self.assertFalse(history.contains(ORIGINAL))
            with sqlite3.connect(history.path) as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "missing")
            self.assertEqual(collector.collect_story(story(ORIGINAL)).status, CollectionStatus.CREATED)
            self.assertEqual(downloader.call_count, 2)

    def test_existing_database_is_migrated_without_losing_links_or_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE articles (
                        id INTEGER PRIMARY KEY,
                        status TEXT NOT NULL CHECK (status IN ('collected', 'missing', 'processed')),
                        relevance INTEGER CHECK (relevance IN (0, 1) OR relevance IS NULL),
                        landing_path TEXT,
                        collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        processed_at TEXT
                    );
                    CREATE TABLE article_urls (
                        url TEXT PRIMARY KEY,
                        article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE
                    );
                    INSERT INTO articles VALUES (42, 'processed', 1, NULL, '2026-01-01', '2026-01-02');
                    INSERT INTO article_urls VALUES ('https://example.com/redirect', 42);
                    INSERT INTO article_urls VALUES ('https://example.com/article', 42);
                    """
                )
            history = ArticleHistory(path)
            self.assertTrue(history.contains(ORIGINAL))
            self.assertTrue(history.contains(FINAL))
            with sqlite3.connect(path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT id, status, relevance, collected_at, processed_at, failed_at, failure_reason "
                        "FROM articles"
                    ).fetchone(),
                    (42, "processed", 1, "2026-01-01", "2026-01-02", None, None),
                )
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            history.record_outcomes([], [(FINAL, None, "Summary failed")])
            self.assertTrue(ArticleHistory(path).contains(ORIGINAL))
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT id, status FROM articles").fetchone(), (42, "failed"))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM article_urls").fetchone()[0], 2)

    def test_failed_migration_rolls_back_original_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE articles (
                        id INTEGER PRIMARY KEY, status TEXT, relevance INTEGER,
                        landing_path TEXT, collected_at TEXT, processed_at TEXT
                    );
                    CREATE TABLE articles_migrated (id INTEGER);
                    INSERT INTO articles VALUES (1, 'processed', 1, NULL, '2026-01-01', '2026-01-02');
                    """
                )
            with self.assertRaises(HistoryError):
                ArticleHistory(path)
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT id, status FROM articles").fetchone(), (1, "processed"))
                self.assertNotIn(
                    "failure_reason", {row[1] for row in connection.execute("PRAGMA table_info(articles)")}
                )

    def test_outcome_updates_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            history = ArticleHistory(path)
            landing = Path(directory) / "article.txt"
            landing.write_text("Article", encoding="utf-8")
            history.record_collected(ORIGINAL, FINAL, landing)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """CREATE TRIGGER prevent_processed BEFORE INSERT ON articles
                       WHEN NEW.status = 'processed'
                       BEGIN SELECT RAISE(ABORT, 'database error'); END"""
                )
            with self.assertRaisesRegex(HistoryError, "database error"):
                history.record_outcomes(
                    [("https://example.com/new", True)],
                    [(FINAL, True, "Summary failed")],
                )
            with sqlite3.connect(path) as connection:
                self.assertEqual(
                    connection.execute("SELECT status, failure_reason FROM articles").fetchall(),
                    [("collected", None)],
                )

    def test_failed_summary_is_archived_skipped_and_can_be_manually_reprocessed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloader = Mock(return_value=DownloadedPage(FINAL, "html"))
            collector = self.make_collector(root, downloader)
            collected = collector.collect_story(story(ORIGINAL))
            process_landing_folder(
                settings(root), FakeSummarizer(True, fail_summary=True),
                now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                status = connection.execute(
                    "SELECT status, relevance, failed_at, failure_reason FROM articles"
                ).fetchone()
                self.assertEqual(status[0:2], ("failed", 1))
                self.assertIsNotNone(status[2])
                self.assertEqual(status[3], "Summary failed after 2 attempts")
            self.assertEqual(collector.collect_story(story(ORIGINAL)).status, CollectionStatus.SKIPPED)
            self.assertEqual(collector.collect_story(story(FINAL)).status, CollectionStatus.SKIPPED)
            self.assertEqual(downloader.call_count, 1)
            failed_file = next((root / "data" / "archive" / "failed").glob("*/*.txt"))
            assert collected.landing_file is not None
            collected.landing_file.write_bytes(failed_file.read_bytes())
            process_landing_folder(
                settings(root), FakeSummarizer(True),
                now=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(
                    connection.execute("SELECT status, failed_at, failure_reason FROM articles").fetchone(),
                    ("processed", None, None),
                )

    def test_failed_link_remains_skipped_if_archived_file_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloader = Mock(return_value=DownloadedPage(FINAL, "html"))
            collector = self.make_collector(root, downloader)
            collector.collect_story(story(ORIGINAL))
            process_landing_folder(settings(root), FakeSummarizer(None, fail_summary=True))
            failed_file = next((root / "data" / "archive" / "failed").glob("*/*.txt"))
            failed_file.unlink()
            self.assertEqual(collector.collect_story(story(ORIGINAL)).status, CollectionStatus.SKIPPED)
            self.assertEqual(collector.collect_story(story(FINAL)).status, CollectionStatus.SKIPPED)
            self.assertEqual(downloader.call_count, 1)
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(
                    connection.execute("SELECT status, relevance, failure_reason FROM articles").fetchone(),
                    ("failed", None, "Relevance and summary attempts failed"),
                )

    def test_malformed_files_recover_urls_when_possible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(root)
            config.landing_dir.mkdir(parents=True)
            (config.landing_dir / "missing-text.txt").write_text(f"\nURL: {ORIGINAL}\n", encoding="utf-8")
            (config.landing_dir / "invalid-utf8.txt").write_bytes(f"URL: {FINAL}\n".encode() + b"\xff")
            (config.landing_dir / "bad-url.txt").write_text("URL: invalid\n\nText", encoding="utf-8")
            (config.landing_dir / "invalid-ipv6.txt").write_text("URL: http://[invalid\n\nText", encoding="utf-8")
            bom_url = "https://example.com/bom"
            (config.landing_dir / "bom.txt").write_bytes(
                b"\xef\xbb\xbf\n" + f"URL: {bom_url}\n".encode()
            )
            process_landing_folder(config, FakeSummarizer(True))
            archived = list((config.archive_dir / "failed").glob("*/*.txt"))
            self.assertEqual(
                {path.name for path in archived},
                {"missing-text.txt", "invalid-utf8.txt", "bad-url.txt", "invalid-ipv6.txt", "bom.txt"},
            )
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                rows = connection.execute(
                    "SELECT article_urls.url, articles.status, articles.failure_reason "
                    "FROM articles JOIN article_urls ON articles.id = article_urls.article_id"
                ).fetchall()
            self.assertEqual(
                set(rows), {
                    (ORIGINAL, "failed", "Invalid landing file"),
                    (FINAL, "failed", "Invalid landing file"),
                    (bom_url, "failed", "Invalid landing file"),
                }
            )

    def test_malformed_collected_file_uses_existing_database_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = self.make_collector(root, Mock(return_value=DownloadedPage(FINAL, "html")))
            collected = collector.collect_story(story(ORIGINAL))
            assert collected.landing_file is not None
            collected.landing_file.write_bytes(b"\xff")
            process_landing_folder(settings(root), FakeSummarizer(True))
            history = ArticleHistory(root / "data" / "article-history.sqlite3")
            self.assertTrue(history.contains(ORIGINAL))
            self.assertTrue(history.contains(FINAL))
            with sqlite3.connect(history.path) as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "failed")

    def test_existing_landing_link_and_redirect_are_recognized(self) -> None:
        for linked_url in (FINAL, ORIGINAL):
            with self.subTest(linked_url=linked_url), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                landing = root / "data" / "landing"
                landing.mkdir(parents=True)
                (landing / "manual.txt").write_text(f"URL: {FINAL}\n\nText", encoding="utf-8")
                downloader = Mock(return_value=DownloadedPage(FINAL, "html"))
                collector = self.make_collector(root, downloader)
                self.assertEqual(collector.collect_story(story(linked_url)).status, CollectionStatus.SKIPPED)
                self.assertEqual(downloader.call_count, 0 if linked_url == FINAL else 1)
                self.assertEqual(collector.collect_story(story(linked_url)).status, CollectionStatus.SKIPPED)
                self.assertEqual(downloader.call_count, 0 if linked_url == FINAL else 1)

    def test_history_write_failure_restores_archived_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(root)
            config.landing_dir.mkdir(parents=True)
            source = config.landing_dir / "article.txt"
            source.write_text(f"URL: {FINAL}\n\nText", encoding="utf-8")
            with patch.object(ArticleHistory, "record_outcomes", side_effect=HistoryError("disk full")):
                with self.assertRaisesRegex(ProcessingError, "disk full"):
                    process_landing_folder(config, FakeSummarizer(True))
            self.assertTrue(source.exists())
            self.assertEqual(list(config.archive_dir.glob("*/*.txt")), [])
            self.assertTrue(ArticleHistory(root / "data" / "article-history.sqlite3").contains(FINAL))
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "collected")

    def test_batch_history_failure_restores_successful_and_failed_files(self) -> None:
        class MixedSummarizer(FakeSummarizer):
            def summarize_article(self, article: Article) -> ArticleSummary:
                if article.source_file == "failed.txt":
                    raise OllamaError("summary failed")
                return super().summarize_article(article)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(root)
            config.landing_dir.mkdir(parents=True)
            successful = config.landing_dir / "successful.txt"
            failed = config.landing_dir / "failed.txt"
            successful.write_text(f"URL: {ORIGINAL}\n\nText", encoding="utf-8")
            failed.write_text(f"URL: {FINAL}\n\nText", encoding="utf-8")
            with patch.object(ArticleHistory, "record_outcomes", side_effect=HistoryError("disk full")):
                with self.assertRaisesRegex(ProcessingError, "disk full"):
                    process_landing_folder(config, MixedSummarizer(True))
            self.assertTrue(successful.exists())
            self.assertTrue(failed.exists())
            self.assertEqual(list(config.archive_dir.glob("*/*.txt")), [])
            self.assertEqual(list((config.archive_dir / "failed").glob("*/*.txt")), [])
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(
                    connection.execute("SELECT status FROM articles ORDER BY status").fetchall(),
                    [("collected",), ("collected",)],
                )

    def test_failure_archiving_failed_file_restores_earlier_success(self) -> None:
        class MixedSummarizer(FakeSummarizer):
            def summarize_article(self, article: Article) -> ArticleSummary:
                if article.source_file == "failed.txt":
                    raise OllamaError("summary failed")
                return super().summarize_article(article)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(root)
            config.landing_dir.mkdir(parents=True)
            successful = config.landing_dir / "successful.txt"
            failed = config.landing_dir / "failed.txt"
            successful.write_text(f"URL: {ORIGINAL}\n\nText", encoding="utf-8")
            failed.write_text(f"URL: {FINAL}\n\nText", encoding="utf-8")
            original_replace = Path.replace

            def replace(path: Path, target: Path) -> Path:
                if path == failed:
                    raise OSError("disk full")
                return original_replace(path, target)

            with patch.object(Path, "replace", replace):
                with self.assertRaisesRegex(ProcessingError, "disk full"):
                    process_landing_folder(config, MixedSummarizer(True))
            self.assertTrue(successful.exists())
            self.assertTrue(failed.exists())
            self.assertEqual(list(config.archive_dir.glob("*/*.txt")), [])

    def test_timing_write_failure_restores_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(root)
            config.landing_dir.mkdir(parents=True)
            source = config.landing_dir / "failed.txt"
            source.write_text(f"URL: {FINAL}\n\nText", encoding="utf-8")
            with patch(
                "article_briefing.processor.write_briefing",
                side_effect=[None, ProcessingError("disk full")],
            ):
                with self.assertRaisesRegex(ProcessingError, "disk full"):
                    process_landing_folder(config, FakeSummarizer(True, fail_summary=True))
            self.assertTrue(source.exists())
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "collected")

    def test_collection_history_write_failure_removes_new_landing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = self.make_collector(root, Mock(return_value=DownloadedPage(FINAL, "html")))
            with patch.object(ArticleHistory, "record_collected", side_effect=HistoryError("disk full")):
                with self.assertRaisesRegex(HistoryError, "disk full"):
                    collector.collect_story(story(ORIGINAL))
            self.assertEqual(list((root / "data" / "landing").glob("*.txt")), [])

    def test_email_failure_does_not_mark_article_processed(self) -> None:
        class FailingSender:
            def send(self, briefing: object) -> float:
                raise EmailDeliveryError("delivery failed")

            def send_failure(self, failure: object) -> float:
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = self.make_collector(root, Mock(return_value=DownloadedPage(FINAL, "html")))
            collected = collector.collect_story(story(ORIGINAL))
            with self.assertRaises(EmailDeliveryError):
                process_landing_folder(settings(root), FakeSummarizer(True), sender=FailingSender())
            assert collected.landing_file is not None
            self.assertTrue(collected.landing_file.exists())
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "collected")

    def test_email_failure_keeps_failed_article_in_landing(self) -> None:
        class FailingSender:
            def send(self, briefing: object) -> float:
                raise EmailDeliveryError("delivery failed")

            def send_failure(self, failure: object) -> float:
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = self.make_collector(root, Mock(return_value=DownloadedPage(FINAL, "html")))
            collected = collector.collect_story(story(ORIGINAL))
            with self.assertRaises(EmailDeliveryError):
                process_landing_folder(
                    settings(root), FakeSummarizer(True, fail_summary=True), sender=FailingSender()
                )
            assert collected.landing_file is not None
            self.assertTrue(collected.landing_file.exists())
            with sqlite3.connect(root / "data" / "article-history.sqlite3") as connection:
                self.assertEqual(connection.execute("SELECT status FROM articles").fetchone()[0], "collected")

    def test_new_redirect_to_known_article_is_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloader = Mock(return_value=DownloadedPage(FINAL, "html"))
            collector = self.make_collector(root, downloader)
            collector.collect_story(story(ORIGINAL))
            other_url = "https://example.com/another-redirect"
            self.assertEqual(collector.collect_story(story(other_url, 2)).status, CollectionStatus.SKIPPED)
            self.assertEqual(downloader.call_count, 2)
            self.assertEqual(collector.collect_story(story(other_url, 3)).status, CollectionStatus.SKIPPED)
            self.assertEqual(downloader.call_count, 2)

    def test_missing_link_can_be_associated_with_existing_processed_article(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = self.make_collector(root, Mock(return_value=DownloadedPage(ORIGINAL, "html")))
            collected = collector.collect_story(story(ORIGINAL))
            assert collected.landing_file is not None
            collected.landing_file.unlink()
            history = ArticleHistory(root / "data" / "article-history.sqlite3")
            self.assertFalse(history.contains(ORIGINAL))
            history.record_processed([(FINAL, True)])
            history.add_alias(FINAL, ORIGINAL)
            self.assertTrue(history.contains(ORIGINAL))
            with sqlite3.connect(history.path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM article_urls").fetchone()[0], 2)

    def test_failed_download_is_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloader = Mock(side_effect=OSError("network error"))
            collector = self.make_collector(root, downloader)
            self.assertEqual(collector.collect_story(story(ORIGINAL)).status, CollectionStatus.FAILED)
            self.assertEqual(collector.collect_story(story(ORIGINAL)).status, CollectionStatus.FAILED)
            self.assertEqual(downloader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
