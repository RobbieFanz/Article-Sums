from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from article_briefing.hacker_news import HackerNewsStory
from article_briefing.hacker_news_collect import (
    CollectionStatus,
    DownloadedPage,
    HackerNewsArticleCollector,
)


class FakeHackerNewsClient:
    def __init__(self, stories: list[HackerNewsStory]) -> None:
        self.stories = stories

    def top_stories(self, limit: int) -> list[HackerNewsStory]:
        return self.stories[:limit]


class HackerNewsArticleCollectorTests(unittest.TestCase):
    def test_writes_extracted_article_text_to_landing_folder(self) -> None:
        story = make_story()
        with tempfile.TemporaryDirectory() as temporary_directory:
            landing_dir = Path(temporary_directory) / "landing"
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([story]),
                landing_dir=landing_dir,
                timeout_seconds=5,
                minimum_text_characters=20,
                page_downloader=lambda url, timeout: DownloadedPage(
                    url="https://example.com/final", html="<html>article</html>"
                ),
                article_extractor=lambda html, url: "Readable article text." * 2,
            )

            results = collector.collect_top_stories(limit=1)

            self.assertEqual(results[0].status, CollectionStatus.CREATED)
            assert results[0].landing_file is not None
            self.assertEqual(
                results[0].landing_file.read_text(encoding="utf-8"),
                "URL: https://example.com/final\n\n"
                "Readable article text.Readable article text.\n",
            )

    def test_skips_story_when_no_external_article_url_exists(self) -> None:
        story = make_story(url=None)
        with tempfile.TemporaryDirectory() as temporary_directory:
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([story]),
                landing_dir=Path(temporary_directory) / "landing",
                timeout_seconds=5,
                minimum_text_characters=20,
            )

            results = collector.collect_top_stories(limit=1)

            self.assertEqual(results[0].status, CollectionStatus.SKIPPED)
            self.assertIn("No external article URL", results[0].detail)

    def test_skips_short_extracted_text(self) -> None:
        story = make_story()
        with tempfile.TemporaryDirectory() as temporary_directory:
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([story]),
                landing_dir=Path(temporary_directory) / "landing",
                timeout_seconds=5,
                minimum_text_characters=20,
                page_downloader=lambda url, timeout: DownloadedPage(
                    url=url, html="<html>short</html>"
                ),
                article_extractor=lambda html, url: "Too short",
            )

            results = collector.collect_top_stories(limit=1)

            self.assertEqual(results[0].status, CollectionStatus.SKIPPED)
            self.assertIn("enough readable", results[0].detail)
            self.assertEqual(list((Path(temporary_directory) / "landing").iterdir()), [])

    def test_does_not_overwrite_an_existing_landing_file(self) -> None:
        story = make_story()
        with tempfile.TemporaryDirectory() as temporary_directory:
            landing_dir = Path(temporary_directory) / "landing"
            landing_dir.mkdir()
            existing_file = landing_dir / "hn-42-example-story.txt"
            existing_file.write_text("Existing article", encoding="utf-8")
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([story]),
                landing_dir=landing_dir,
                timeout_seconds=5,
                minimum_text_characters=20,
            )

            results = collector.collect_top_stories(limit=1)

            self.assertEqual(results[0].status, CollectionStatus.SKIPPED)
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "Existing article")


def make_story(url: str | None = "https://example.com/article") -> HackerNewsStory:
    return HackerNewsStory(
        id=42,
        title="Example Story",
        score=100,
        descendants=25,
        by="example",
        url=url,
        story_type="Linked article" if url else "Ask HN",
    )


if __name__ == "__main__":
    unittest.main()
