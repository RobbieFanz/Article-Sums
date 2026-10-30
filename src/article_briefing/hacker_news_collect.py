from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import ssl
import sys
import unicodedata
import zlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import certifi
import trafilatura
from lxml.etree import LxmlError

from article_briefing.config import ConfigurationError, load_dotenv, resolve_project_path
from article_briefing.hacker_news import (
    DEFAULT_STORY_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    HackerNewsClient,
    HackerNewsError,
    HackerNewsStory,
    positive_float,
    positive_integer,
)
from article_briefing.history import ArticleHistory, DEFAULT_HISTORY_PATH, HistoryError
from article_briefing.processor import ProcessingError, load_article


DEFAULT_LANDING_DIR = Path("data/landing")
DEFAULT_MINIMUM_TEXT_CHARACTERS = 500
MAX_PAGE_BYTES = 10 * 1024 * 1024


class ArticleCollectionError(RuntimeError):
    """Raised when an external article cannot be collected."""


class CollectionStatus(StrEnum):
    CREATED = "created"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class DownloadedPage:
    url: str
    html: str


@dataclass(frozen=True)
class CollectionResult:
    story: HackerNewsStory
    status: CollectionStatus
    detail: str
    landing_file: Path | None = None


class HackerNewsArticleCollector:
    def __init__(
        self,
        client: HackerNewsClient,
        landing_dir: Path,
        timeout_seconds: float,
        minimum_text_characters: int,
        page_downloader: Callable[[str, float], DownloadedPage] = None,
        article_extractor: Callable[[str, str], str | None] = None,
        history_path: Path | None = None,
    ) -> None:
        self._client = client
        self._landing_dir = landing_dir
        self._timeout_seconds = timeout_seconds
        self._minimum_text_characters = minimum_text_characters
        self._page_downloader = page_downloader or download_page
        self._article_extractor = article_extractor or extract_article_text
        default_directory = landing_dir.parent if landing_dir.name == "landing" else landing_dir
        self._history = ArticleHistory(history_path or default_directory / DEFAULT_HISTORY_PATH.name)

    def _find_landing_file(self, url: str) -> Path | None:
        for path in self._landing_dir.glob("*.txt"):
            try:
                article = load_article(path, None)
            except (OSError, ProcessingError) as error:
                print(f"Warning: Could not inspect {path}: {error}", file=sys.stderr)
                continue
            if article.url == url:
                return path
        return None

    def collect_top_stories(self, limit: int) -> list[CollectionResult]:
        self._landing_dir.mkdir(parents=True, exist_ok=True)
        results: list[CollectionResult] = []
        for story in self._client.top_stories(limit):
            try:
                results.append(self.collect_story(story))
            except (OSError, UnicodeError, ValueError) as error:
                results.append(
                    CollectionResult(
                        story,
                        CollectionStatus.FAILED,
                        f"Could not process story: {error}",
                    )
                )
        return results

    def collect_story(self, story: HackerNewsStory) -> CollectionResult:
        if story.url is None:
            return CollectionResult(
                story,
                CollectionStatus.SKIPPED,
                "No external article URL is available.",
            )
        if not is_public_http_url(story.url):
            return CollectionResult(
                story,
                CollectionStatus.SKIPPED,
                "The linked URL is not a public HTTP(S) URL.",
            )

        if self._history.contains(story.url):
            return CollectionResult(story, CollectionStatus.SKIPPED, "Article already collected or processed.")

        existing_file = self._find_landing_file(story.url)
        if existing_file is not None:
            self._history.record_collected(story.url, story.url, existing_file)
            return CollectionResult(
                story, CollectionStatus.SKIPPED, "Article already in the landing folder.", existing_file
            )

        try:
            destination = self._landing_dir / landing_filename(story)
        except (UnicodeError, ValueError) as error:
            return CollectionResult(
                story,
                CollectionStatus.FAILED,
                f"Could not parse story title: {error}",
            )
        if destination.exists():
            return CollectionResult(
                story,
                CollectionStatus.SKIPPED,
                f"Landing file already exists: {destination.name}",
                destination,
            )

        try:
            page = self._page_downloader(story.url, self._timeout_seconds)
            if not is_public_http_url(page.url):
                raise ArticleCollectionError(
                    "The final redirected URL is not a public HTTP(S) URL."
                )
            if self._history.contains(page.url):
                self._history.add_alias(page.url, story.url)
                return CollectionResult(
                    story, CollectionStatus.SKIPPED, "Redirected article already collected or processed."
                )
            existing_file = self._find_landing_file(page.url)
            if existing_file is not None:
                self._history.record_collected(story.url, page.url, existing_file)
                return CollectionResult(
                    story, CollectionStatus.SKIPPED, "Redirected article already in the landing folder.",
                    existing_file,
                )
            text = self._article_extractor(page.html, page.url)
        except ArticleCollectionError as error:
            return CollectionResult(story, CollectionStatus.FAILED, str(error))
        except UnicodeError as error:
            return CollectionResult(
                story,
                CollectionStatus.FAILED,
                f"Page text was not valid UTF-8: {error}",
            )
        except (OSError, ValueError) as error:
            return CollectionResult(story, CollectionStatus.FAILED, str(error))

        if text is None or len(text.strip()) < self._minimum_text_characters:
            return CollectionResult(
                story,
                CollectionStatus.SKIPPED,
                "The page did not contain enough readable article text.",
            )

        try:
            write_landing_file(destination, page.url, text.strip())
        except (OSError, UnicodeError) as error:
            return CollectionResult(
                story,
                CollectionStatus.FAILED,
                f"Could not write landing file: {error}",
            )
        try:
            self._history.record_collected(story.url, page.url, destination)
        except HistoryError:
            try:
                destination.unlink()
            except OSError as error:
                raise HistoryError(
                    f"Could not record collection or remove {destination}: {error}"
                ) from error
            raise
        return CollectionResult(
            story,
            CollectionStatus.CREATED,
            f"Saved {len(text.strip()):,} characters of readable text.",
            destination,
        )


def download_page(url: str, timeout_seconds: float) -> DownloadedPage:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "gzip",
            "User-Agent": "ArticleBriefing/0.1 (private self-hosted article reader)",
        },
    )
    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
            context=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ArticleCollectionError(
                    f"Expected an HTML page but received {content_type or 'unknown content'}."
                )
            body = response.read(MAX_PAGE_BYTES + 1)
            if len(body) > MAX_PAGE_BYTES:
                raise ArticleCollectionError(
                    f"Page exceeded the {MAX_PAGE_BYTES // (1024 * 1024)} MB download limit."
                )
            content_encoding = response.headers.get("Content-Encoding", "").strip().lower()
            if content_encoding == "gzip":
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                        body = compressed.read(MAX_PAGE_BYTES + 1)
                except (OSError, EOFError, zlib.error) as error:
                    raise ArticleCollectionError(
                        "Page contained invalid compressed data."
                    ) from error
                if len(body) > MAX_PAGE_BYTES:
                    raise ArticleCollectionError(
                        f"Page exceeded the {MAX_PAGE_BYTES // (1024 * 1024)} MB download limit."
                    )
            elif content_encoding not in {"", "identity"}:
                raise ArticleCollectionError(
                    f"Page used an unsupported content encoding: {content_encoding}."
                )
            charset = response.headers.get_content_charset() or "utf-8"
        html = body.decode(charset)
    except HTTPError as error:
        raise ArticleCollectionError(f"Page returned HTTP {error.code}.") from error
    except URLError as error:
        raise ArticleCollectionError(f"Could not download page: {error.reason}") from error
    except TimeoutError as error:
        raise ArticleCollectionError("Page download timed out.") from error
    except UnicodeDecodeError as error:
        raise ArticleCollectionError("Page text could not be decoded.") from error
    except LookupError as error:
        raise ArticleCollectionError(
            f"Page declared an unsupported character encoding: {error}."
        ) from error

    return DownloadedPage(url=final_url, html=html)


def extract_article_text(html: str, url: str) -> str | None:
    try:
        return trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=True,
        )
    except (LxmlError, TypeError, UnicodeError, ValueError) as error:
        raise ArticleCollectionError("Could not extract readable article text.") from error


def is_public_http_url(value: str) -> bool:
    try:
        parsed_url = urlparse(value)
    except ValueError:
        return False
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        return False
    hostname = parsed_url.hostname.lower()
    return hostname not in {"localhost", "localhost.localdomain"} and not re.fullmatch(
        r"127(?:\.\d{1,3}){3}|0\.0\.0\.0|::1", hostname
    )


def landing_filename(story: HackerNewsStory) -> str:
    normalized_title = unicodedata.normalize("NFKD", story.title).encode(
        "ascii", "ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_title.lower()).strip("-")
    return f"hn-{story.id}-{slug[:70] or 'article'}.txt"


def write_landing_file(path: Path, url: str, text: str) -> None:
    content = f"URL: {url}\n\n{text}\n"
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except (OSError, UnicodeError):
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def render_results(results: list[CollectionResult]) -> str:
    if not results:
        return "No readable Hacker News top stories were returned."

    lines = ["Hacker News article collection", ""]
    for result in results:
        lines.extend(
            [
                f"{result.status.value.upper()}: {result.story.title}",
                f"  {result.detail}",
            ]
        )
        if result.landing_file is not None:
            lines.append(f"  Landing file: {result.landing_file}")
        lines.append("")

    created_count = sum(
        result.status is CollectionStatus.CREATED for result in results
    )
    lines.append(f"Collected {created_count} of {len(results)} top stories.")
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch public linked articles from current Hacker News top stories and "
            "write extracted text files to the landing folder."
        )
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        default=DEFAULT_STORY_LIMIT,
        help=f"Number of top stories to inspect (default: {DEFAULT_STORY_LIMIT}).",
    )
    parser.add_argument(
        "--landing-dir",
        type=Path,
        default=DEFAULT_LANDING_DIR,
        help=f"Directory for extracted article text (default: {DEFAULT_LANDING_DIR}).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Article history database (defaults to ARTICLE_HISTORY_PATH or data/article-history.sqlite3).",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--minimum-text-characters",
        type=positive_integer,
        default=DEFAULT_MINIMUM_TEXT_CHARACTERS,
        help=(
            "Minimum extracted text length required to create a landing file "
            f"(default: {DEFAULT_MINIMUM_TEXT_CHARACTERS})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    try:
        project_root = Path.cwd()
        load_dotenv(project_root / ".env")
        history_path = arguments.database or resolve_project_path(
            project_root, os.getenv("ARTICLE_HISTORY_PATH", str(DEFAULT_HISTORY_PATH))
        )
        collector = HackerNewsArticleCollector(
            client=HackerNewsClient(timeout_seconds=arguments.timeout),
            landing_dir=arguments.landing_dir,
            timeout_seconds=arguments.timeout,
            minimum_text_characters=arguments.minimum_text_characters,
            history_path=history_path,
        )
        results = collector.collect_top_stories(arguments.limit)
    except (ConfigurationError, HackerNewsError, HistoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(render_results(results))


if __name__ == "__main__":
    main()
