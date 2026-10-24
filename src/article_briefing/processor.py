from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from article_briefing.config import Settings
from article_briefing.models import Article, ArticleSummary, BriefingOverview


class ProcessingError(RuntimeError):
    """Raised when a landing batch cannot be processed safely."""


@dataclass(frozen=True)
class ProcessingResult:
    briefing_path: Path | None
    warnings: tuple[str, ...]


class Summarizer(Protocol):
    def summarize_article(self, article: Article) -> ArticleSummary: ...

    def summarize_briefing(self, summaries: list[ArticleSummary]) -> BriefingOverview: ...


def process_landing_folder(
    settings: Settings, summarizer: Summarizer, now: datetime | None = None
) -> ProcessingResult:
    settings.landing_dir.mkdir(parents=True, exist_ok=True)
    landing_files = sorted(settings.landing_dir.glob("*.txt"))
    if not landing_files:
        return ProcessingResult(briefing_path=None, warnings=())

    articles: list[Article] = []
    processed_files: list[Path] = []
    warnings: list[str] = []
    for path in landing_files:
        try:
            article = load_article(path, settings.max_article_characters)
        except ProcessingError as error:
            warnings.append(str(error))
            continue
        articles.append(article)
        processed_files.append(path)

    if not articles:
        return ProcessingResult(briefing_path=None, warnings=tuple(warnings))

    summaries = [summarizer.summarize_article(article) for article in articles]
    overview = summarizer.summarize_briefing(summaries)

    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H%M%S%z")
    briefing_path = settings.briefing_dir / timestamp
    archive_path = settings.archive_dir / timestamp
    if briefing_path.exists() or archive_path.exists():
        raise ProcessingError(
            f"Briefing timestamp collision at {timestamp}. Rerun the command."
        )

    write_briefing(briefing_path, settings, timestamp, summaries, overview, warnings)
    archive_files(processed_files, archive_path)
    return ProcessingResult(briefing_path=briefing_path, warnings=tuple(warnings))


def load_article(path: Path, maximum_characters: int) -> Article:
    try:
        lines = path.read_text(encoding="utf-8").lstrip("\ufeff").splitlines()
    except UnicodeDecodeError as error:
        raise ProcessingError(f"{path.name} must be UTF-8 text.") from error

    first_content_line = next(
        (index for index, line in enumerate(lines) if line.strip()), None
    )
    if first_content_line is None:
        raise ProcessingError(f"{path.name} is empty.")

    url = lines[first_content_line].strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ProcessingError(
            f"{path.name} must start with a complete http or https article URL."
        )

    text = "\n".join(lines[first_content_line + 1 :]).strip()
    if not text:
        raise ProcessingError(f"{path.name} has no article text after its URL.")

    was_truncated = len(text) > maximum_characters
    return Article(
        source_file=path.name,
        title=path.stem.replace("_", " ").replace("-", " ").strip(),
        url=url,
        text=text[:maximum_characters],
        was_truncated=was_truncated,
    )


def write_briefing(
    briefing_path: Path,
    settings: Settings,
    timestamp: str,
    summaries: list[ArticleSummary],
    overview: BriefingOverview,
    warnings: list[str],
) -> None:
    briefing_path.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": timestamp,
        "model": settings.ollama_model,
        "overview": overview.as_dict(),
        "articles": [summary.as_dict() for summary in summaries],
        "skipped_files": warnings,
    }
    try:
        write_text(
            briefing_path / "briefing.json",
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )
        write_text(briefing_path / "briefing.md", render_markdown(payload))
        write_article_summaries(briefing_path / "article-summaries", summaries)
    except OSError as error:
        raise ProcessingError(f"Could not write briefing output: {error}") from error


def archive_files(landing_files: list[Path], archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    moved_files: list[tuple[Path, Path]] = []
    try:
        archive_path.mkdir()
        for source_path in landing_files:
            destination_path = archive_path / source_path.name
            source_path.replace(destination_path)
            moved_files.append((source_path, destination_path))
    except OSError as error:
        rollback_failures = []
        for source_path, destination_path in reversed(moved_files):
            try:
                destination_path.replace(source_path)
            except OSError as rollback_error:
                rollback_failures.append(str(rollback_error))
        if archive_path.exists() and not any(archive_path.iterdir()):
            archive_path.rmdir()
        detail = f" Could not restore all files: {'; '.join(rollback_failures)}." if rollback_failures else ""
        raise ProcessingError(f"Could not archive landing files: {error}.{detail}") from error


def write_text(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_article_summaries(
    summaries_path: Path, summaries: list[ArticleSummary]
) -> None:
    summaries_path.mkdir(exist_ok=False)
    for summary in summaries:
        filename = f"{Path(summary.source_file).stem}.md"
        write_text(summaries_path / filename, render_article_summary(summary))


def render_markdown(payload: dict[str, object]) -> str:
    overview = payload["overview"]
    assert isinstance(overview, dict)

    lines = [
        "# Briefing",
        "",
        f"Created: {payload['created_at']}",
        f"Model: `{payload['model']}`",
        "",
        "## Overview",
        "",
        str(overview["overview"]),
        "",
        "### Major themes",
        "",
        *[f"- {theme}" for theme in overview["major_themes"]],
        "",
        f"**Watch for:** {overview['watch_for']}",
        "",
        "Individual article summaries are in [`article-summaries/`](article-summaries/).",
    ]
    return "\n".join(lines) + "\n"


def render_article_summary(summary: ArticleSummary) -> str:
    lines = [
        f"# [{summary.title}]({summary.url})",
        "",
        "## Summary",
        "",
        summary.summary,
        "",
        "## Key points",
        "",
        *[f"- {point}" for point in summary.key_points],
        "",
        "## Why it matters",
        "",
        summary.why_it_matters,
        "",
        "## Read recommendation",
        "",
        summary.read_recommendation,
        "",
        f"*{summary.input_note}*",
    ]
    return "\n".join(lines) + "\n"
