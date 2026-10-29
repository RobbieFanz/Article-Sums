from __future__ import annotations

import gzip
import io
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from article_briefing.hacker_news import HackerNewsStory
from article_briefing.hacker_news_collect import (
    ArticleCollectionError,
    CollectionStatus,
    DownloadedPage,
    HackerNewsArticleCollector,
    download_page,
)


class FakePageResponse(io.BytesIO):
    def __init__(
        self, body: bytes, content_type: str = "text/html; charset=utf-8",
        content_encoding: str | None = None,
    ) -> None:
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding

    def geturl(self) -> str:
        return "https://example.com/final"


class DownloadPageTests(unittest.TestCase):
    def test_decodes_uncompressed_html(self) -> None:
        response = FakePageResponse(b"<html>plain</html>")
        with patch("article_briefing.hacker_news_collect.urlopen", return_value=response) as urlopen:
            page = download_page("https://example.com/article", 5)

        self.assertEqual(page, DownloadedPage("https://example.com/final", "<html>plain</html>"))
        self.assertEqual(urlopen.call_args.args[0].get_header("Accept-encoding"), "gzip")

    def test_decodes_gzip_html_using_declared_charset(self) -> None:
        html = "<html>caf\u00e9</html>"
        response = FakePageResponse(
            gzip.compress(html.encode("iso-8859-1")),
            "text/html; charset=iso-8859-1",
            "gzip",
        )
        with patch("article_briefing.hacker_news_collect.urlopen", return_value=response):
            page = download_page("https://example.com/article", 5)

        self.assertEqual(page, DownloadedPage("https://example.com/final", html))

    def test_rejects_decompressed_pages_over_limit(self) -> None:
        response = FakePageResponse(gzip.compress(b"x" * 101), content_encoding="gzip")
        with (
            patch("article_briefing.hacker_news_collect.MAX_PAGE_BYTES", 100),
            patch("article_briefing.hacker_news_collect.urlopen", return_value=response),
            self.assertRaisesRegex(ArticleCollectionError, "download limit"),
        ):
            download_page("https://example.com/article", 5)

    def test_rejects_invalid_gzip(self) -> None:
        response = FakePageResponse(b"not gzip", content_encoding="gzip")
        with (
            patch("article_briefing.hacker_news_collect.urlopen", return_value=response),
            self.assertRaisesRegex(ArticleCollectionError, "compressed"),
        ):
            download_page("https://example.com/article", 5)

    def test_rejects_truncated_gzip(self) -> None:
        response = FakePageResponse(gzip.compress(b"html")[:-4], content_encoding="gzip")
        with (
            patch("article_briefing.hacker_news_collect.urlopen", return_value=response),
            self.assertRaisesRegex(ArticleCollectionError, "compressed"),
        ):
            download_page("https://example.com/article", 5)

    def test_rejects_downloads_over_limit(self) -> None:
        response = FakePageResponse(b"x" * 101)
        with (
            patch("article_briefing.hacker_news_collect.MAX_PAGE_BYTES", 100),
            patch("article_briefing.hacker_news_collect.urlopen", return_value=response),
            self.assertRaisesRegex(ArticleCollectionError, "download limit"),
        ):
            download_page("https://example.com/article", 5)

    def test_rejects_unsupported_content_encoding(self) -> None:
        response = FakePageResponse(b"some data", content_encoding="br")
        with (
            patch("article_briefing.hacker_news_collect.urlopen", return_value=response),
            self.assertRaisesRegex(ArticleCollectionError, "content encoding"),
        ):
            download_page("https://example.com/article", 5)


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

    def test_invalid_unicode_in_one_article_does_not_stop_next_article(self) -> None:
        invalid_story = make_story(
            id=1,
            title="Invalid Unicode",
            url="https://example.com/invalid",
        )
        valid_story = make_story(
            id=2,
            title="Valid Article",
            url="https://example.com/valid",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            landing_dir = Path(temporary_directory) / "landing"
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([invalid_story, valid_story]),
                landing_dir=landing_dir,
                timeout_seconds=5,
                minimum_text_characters=20,
                page_downloader=download_page_with_invalid_unicode_for_first_story,
                article_extractor=lambda html, url: "Readable article text." * 2,
            )

            results = collector.collect_top_stories(limit=2)

            self.assertEqual(
                [result.status for result in results],
                [CollectionStatus.FAILED, CollectionStatus.CREATED],
            )
            self.assertIn("not valid UTF-8", results[0].detail)
            self.assertTrue((landing_dir / "hn-2-valid-article.txt").exists())

    def test_invalid_compressed_page_does_not_stop_next_article(self) -> None:
        stories = [
            make_story(id=1, title="Invalid", url="https://example.com/invalid"),
            make_story(id=2, title="Valid", url="https://example.com/valid"),
        ]
        responses = [
            FakePageResponse(b"not gzip", content_encoding="gzip"),
            FakePageResponse(gzip.compress(b"<html>valid</html>"), content_encoding="gzip"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient(stories),
                landing_dir=Path(temporary_directory),
                timeout_seconds=5,
                minimum_text_characters=1,
                article_extractor=lambda html, url: html,
            )
            with patch("article_briefing.hacker_news_collect.urlopen", side_effect=responses):
                results = collector.collect_top_stories(limit=2)

            self.assertEqual(
                [result.status for result in results],
                [CollectionStatus.FAILED, CollectionStatus.CREATED],
            )
            self.assertIn("invalid compressed data", results[0].detail)
            self.assertTrue((Path(temporary_directory) / "hn-2-valid.txt").exists())

    def test_non_utf8_text_cannot_be_written_and_is_skipped(self) -> None:
        story = make_story()
        with tempfile.TemporaryDirectory() as temporary_directory:
            collector = HackerNewsArticleCollector(
                client=FakeHackerNewsClient([story]),
                landing_dir=Path(temporary_directory) / "landing",
                timeout_seconds=5,
                minimum_text_characters=1,
                page_downloader=lambda url, timeout: DownloadedPage(url=url, html="<html>text</html>"),
                article_extractor=lambda html, url: "\ud800",
            )

            results = collector.collect_top_stories(limit=1)

            self.assertEqual(results[0].status, CollectionStatus.FAILED)
            self.assertIn("Could not write landing file", results[0].detail)
            self.assertEqual(list((Path(temporary_directory) / "landing").iterdir()), [])


def download_page_with_invalid_unicode_for_first_story(
    url: str, timeout: float
) -> DownloadedPage:
    if url.endswith("/invalid"):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    return DownloadedPage(url=url, html="<html>article</html>")


def make_story(
    url: str | None = "https://example.com/article",
    id: int = 42,
    title: str = "Example Story",
) -> HackerNewsStory:
    return HackerNewsStory(
        id=id,
        title=title,
        score=100,
        descendants=25,
        by="example",
        url=url,
        story_type="Linked article" if url else "Ask HN",
    )


if __name__ == "__main__":
    unittest.main()
