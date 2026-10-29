from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol
from urllib.parse import urlparse

from article_briefing.config import Settings
from article_briefing.models import (
    Article,
    ArticleNotSelected,
    ArticleSummary,
    Briefing,
    BriefingOverview,
    FailureReport,
    RelevanceDecision,
    RelevanceFailure,
)
from article_briefing.ollama import OllamaError, OllamaOutputError


class ProcessingError(RuntimeError):
    """Raised when a landing batch cannot be processed safely."""


@dataclass(frozen=True)
class ProcessingResult:
    briefing_path: Path | None
    warnings: tuple[str, ...]
    generation_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    email_delivery_duration_seconds: float | None = None
    included_article_count: int = 0
    not_included_article_count: int = 0


class Summarizer(Protocol):
    def assess_relevance(self, article: Article) -> RelevanceDecision: ...

    def summarize_article(self, article: Article) -> ArticleSummary: ...

    def summarize_briefing(self, summaries: list[ArticleSummary]) -> BriefingOverview: ...


class BriefingSender(Protocol):
    def send(self, briefing: Briefing) -> float: ...

    def send_failure(self, failure: FailureReport) -> float: ...


@dataclass
class ProcessingProgress:
    processed_article_count: int = 0
    current_article: str | None = None


def notify_failure(
    sender: BriefingSender, error: Exception, started_at: float, progress: ProcessingProgress
) -> None:
    failure = FailureReport(
        error=f"{type(error).__name__}: {error}",
        duration_seconds=perf_counter() - started_at,
        processed_article_count=progress.processed_article_count,
        current_article=progress.current_article,
    )
    try:
        sender.send_failure(failure)
    except Exception as notification_error:
        print(
            f"Error: Could not send failure notification: {notification_error}",
            file=sys.stderr,
        )


def process_landing_folder(
    settings: Settings,
    summarizer: Summarizer,
    now: datetime | None = None,
    sender: BriefingSender | None = None,
) -> ProcessingResult:
    started_at = perf_counter()
    progress = ProcessingProgress()
    try:
        return _process_landing_folder(settings, summarizer, now, sender, started_at, progress)
    except Exception as error:
        if sender is not None:
            notify_failure(sender, error, started_at, progress)
        raise


def _process_landing_folder(
    settings: Settings,
    summarizer: Summarizer,
    now: datetime | None,
    sender: BriefingSender | None,
    started_at: float,
    progress: ProcessingProgress,
) -> ProcessingResult:
    settings.landing_dir.mkdir(parents=True, exist_ok=True)
    landing_files = sorted(settings.landing_dir.glob("*.txt"))
    if not landing_files:
        return ProcessingResult(briefing_path=None, warnings=())

    articles: list[Article] = []
    processed_files: list[Path] = []
    warnings: list[str] = []
    articles_not_selected: list[ArticleNotSelected] = []
    for path in landing_files:
        progress.current_article = path.name
        try:
            article = load_article(path, settings.max_article_characters)
        except ProcessingError as error:
            warnings.append(str(error))
            articles_not_selected.append(
                ArticleNotSelected(
                    source_file=path.name,
                    title=path.stem.replace("_", " ").replace("-", " ").strip(),
                    url=None,
                    reason=str(error),
                )
            )
            progress.processed_article_count += 1
            progress.current_article = None
            continue
        articles.append(article)
        processed_files.append(path)
        progress.current_article = None

    summaries: list[ArticleSummary] = []
    relevance_failures: list[RelevanceFailure] = []
    archived_files: list[Path] = []
    not_included_article_count = len(articles_not_selected)
    for article, source_file in zip(articles, processed_files, strict=True):
        progress.current_article = source_file.name
        relevance_decision, invalid_outputs = assess_relevance_with_retry(
            summarizer, article
        )
        if relevance_decision is None:
            try:
                summary = summarize_article_with_retry(summarizer, article)
            except OllamaError as error:
                warning = f"{source_file.name} could not be summarized after 2 attempts: {error}"
                warnings.append(warning)
                articles_not_selected.append(
                    ArticleNotSelected(
                        source_file=article.source_file,
                        title=article.title,
                        url=article.url,
                        reason=(
                            "Relevance check returned invalid output after 3 attempts. "
                            f"{warning}"
                        ),
                    )
                )
                not_included_article_count += 1
                progress.processed_article_count += 1
                progress.current_article = None
                continue
            relevance_failures.append(
                RelevanceFailure(summary=summary, invalid_outputs=invalid_outputs)
            )
            archived_files.append(source_file)
            not_included_article_count += 1
            progress.processed_article_count += 1
            progress.current_article = None
            continue

        if not relevance_decision.include:
            articles_not_selected.append(
                ArticleNotSelected(
                    source_file=article.source_file,
                    title=article.title,
                    url=article.url,
                    reason=relevance_decision.reason,
                )
            )
            archived_files.append(source_file)
            not_included_article_count += 1
            progress.processed_article_count += 1
            progress.current_article = None
            continue

        try:
            summaries.append(summarize_article_with_retry(summarizer, article))
        except OllamaError as error:
            warning = f"{source_file.name} could not be summarized after 2 attempts: {error}"
            warnings.append(warning)
            articles_not_selected.append(
                ArticleNotSelected(
                    source_file=article.source_file,
                    title=article.title,
                    url=article.url,
                    reason=warning,
                )
            )
            not_included_article_count += 1
            progress.processed_article_count += 1
            progress.current_article = None
            continue
        archived_files.append(source_file)
        progress.processed_article_count += 1
        progress.current_article = None

    overview = (
        summarize_briefing_with_retry(summarizer, summaries)
        if summaries
        else BriefingOverview(
            overview="No article summaries are available for the combined briefing.",
            major_themes=[],
            watch_for="Review a later collection cycle for new relevant stories.",
        )
    )

    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H%M%S%z")
    briefing_path = settings.briefing_dir / timestamp
    archive_path = settings.archive_dir / timestamp
    if briefing_path.exists() or archive_path.exists():
        raise ProcessingError(
            f"Briefing timestamp collision at {timestamp}. Rerun the command."
        )

    generation_duration_seconds = perf_counter() - started_at
    briefing = Briefing(
        created_at=timestamp,
        model=settings.ollama_model,
        overview=overview,
        articles=summaries,
        skipped_files=warnings,
        generation_duration_seconds=generation_duration_seconds,
        relevance_failures=relevance_failures,
        articles_not_selected=articles_not_selected,
        included_article_count=len(summaries),
        not_included_article_count=not_included_article_count,
    )
    create_briefing_directory(briefing_path)
    write_briefing(briefing_path, briefing)
    email_delivery_duration_seconds = sender.send(briefing) if sender else None
    if archived_files:
        archive_files(archived_files, archive_path, progress)
    completed_briefing = Briefing(
        created_at=briefing.created_at,
        model=briefing.model,
        overview=briefing.overview,
        articles=briefing.articles,
        skipped_files=briefing.skipped_files,
        generation_duration_seconds=briefing.generation_duration_seconds,
        relevance_failures=briefing.relevance_failures,
        articles_not_selected=briefing.articles_not_selected,
        included_article_count=briefing.included_article_count,
        not_included_article_count=briefing.not_included_article_count,
        total_duration_seconds=perf_counter() - started_at,
        email_delivery_duration_seconds=email_delivery_duration_seconds,
    )
    write_briefing(briefing_path, completed_briefing)
    return ProcessingResult(
        briefing_path=briefing_path,
        warnings=tuple(warnings),
        generation_duration_seconds=completed_briefing.generation_duration_seconds,
        total_duration_seconds=completed_briefing.total_duration_seconds,
        email_delivery_duration_seconds=email_delivery_duration_seconds,
        included_article_count=completed_briefing.included_article_count,
        not_included_article_count=completed_briefing.not_included_article_count,
    )


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

    url, text_start_line = optional_url(lines, first_content_line, path)
    text = "\n".join(lines[text_start_line:]).strip()
    if not text:
        raise ProcessingError(f"{path.name} has no article text.")

    was_truncated = len(text) > maximum_characters
    return Article(
        source_file=path.name,
        title=path.stem.replace("_", " ").replace("-", " ").strip(),
        url=url,
        text=text[:maximum_characters],
        was_truncated=was_truncated,
    )


def summarize_article_with_retry(summarizer: Summarizer, article: Article) -> ArticleSummary:
    last_error: OllamaError | None = None
    for _ in range(2):
        try:
            return summarizer.summarize_article(article)
        except OllamaError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def assess_relevance_with_retry(
    summarizer: Summarizer, article: Article
) -> tuple[RelevanceDecision | None, list[str]]:
    invalid_outputs: list[str] = []
    for _ in range(3):
        try:
            return summarizer.assess_relevance(article), invalid_outputs
        except OllamaOutputError as error:
            invalid_outputs.append(error.raw_output)
    return None, invalid_outputs


def summarize_briefing_with_retry(
    summarizer: Summarizer, summaries: list[ArticleSummary]
) -> BriefingOverview:
    last_error: OllamaError | None = None
    for _ in range(2):
        try:
            return summarizer.summarize_briefing(summaries)
        except OllamaError as error:
            last_error = error
    assert last_error is not None
    return BriefingOverview(
        overview="The combined briefing could not be created.",
        major_themes=[],
        watch_for="Review the individual article summaries below.",
        error=str(last_error),
    )


def optional_url(lines: list[str], first_content_line: int, path: Path) -> tuple[str | None, int]:
    first_line = lines[first_content_line].strip()
    if first_line.lower().startswith("url:"):
        url = first_line[4:].strip()
        if not is_complete_url(url):
            raise ProcessingError(
                f"{path.name} has an invalid URL: header. Use a complete http or https URL."
            )
        return url, first_content_line + 1
    if is_complete_url(first_line):
        return first_line, first_content_line + 1
    return None, first_content_line


def is_complete_url(value: str) -> bool:
    parsed_url = urlparse(value)
    return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)


def create_briefing_directory(briefing_path: Path) -> None:
    try:
        briefing_path.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ProcessingError(f"Could not create briefing output: {error}") from error


def write_briefing(briefing_path: Path, briefing: Briefing) -> None:
    payload = briefing.as_dict()
    try:
        write_text(
            briefing_path / "briefing.json",
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )
        write_text(briefing_path / "briefing.md", render_markdown(payload))
        write_article_summaries(briefing_path / "article-summaries", briefing.articles)
    except OSError as error:
        raise ProcessingError(f"Could not write briefing output: {error}") from error


def archive_files(
    landing_files: list[Path], archive_path: Path, progress: ProcessingProgress
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    moved_files: list[tuple[Path, Path]] = []
    try:
        archive_path.mkdir()
        for source_path in landing_files:
            progress.current_article = source_path.name
            destination_path = archive_path / source_path.name
            source_path.replace(destination_path)
            moved_files.append((source_path, destination_path))
        progress.current_article = None
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
    summaries_path.mkdir(exist_ok=True)
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
        f"Articles included: {payload['included_article_count']}",
        f"Articles not included: {payload['not_included_article_count']}",
        "These counts refer to the main briefing; excluded articles are listed below.",
        f"Briefing generation time: {format_seconds(payload['generation_duration_seconds'])}",
        (
            f"Total process time: {format_seconds(payload['total_duration_seconds'])}"
            if payload["total_duration_seconds"] is not None
            else ""
        ),
        (
            "SMTP delivery time: "
            f"{format_seconds(payload['email_delivery_duration_seconds'])}"
            if payload["email_delivery_duration_seconds"] is not None
            else ""
        ),
        "",
        "## Overview",
        "",
        str(overview["overview"]),
    ]
    overview_error = overview["error"]
    assert overview_error is None or isinstance(overview_error, str)
    if overview_error:
        lines.extend(
            [
                "",
                "## Briefing overview error",
                "",
                f"The combined briefing could not be created: {overview_error}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Major themes",
                "",
                *[f"- {theme}" for theme in overview["major_themes"]],
                "",
                f"**Watch for:** {overview['watch_for']}",
            ]
        )
    lines.extend(
        [
            "",
            "Individual article summaries are in [`article-summaries/`](article-summaries/).",
        ]
    )
    articles_not_selected = payload["articles_not_selected"]
    assert isinstance(articles_not_selected, list)
    if articles_not_selected:
        lines.extend(["", "## Articles not selected", ""])
        for article in articles_not_selected:
            assert isinstance(article, dict)
            title = article["title"]
            url = article["url"]
            lines.extend(
                [
                    f"- [{title}]({url})" if url else f"- {title} (original link unavailable)",
                    f"  - Reason: {article['reason']}",
                ]
            )
    skipped_files = payload["skipped_files"]
    assert isinstance(skipped_files, list)
    if skipped_files:
        lines.extend(
            [
                "",
                "## Articles not included",
                "",
                *[f"- {warning}" for warning in skipped_files],
            ]
        )
    relevance_failures = payload["relevance_failures"]
    assert isinstance(relevance_failures, list)
    if relevance_failures:
        lines.extend(
            [
                "",
                "## Relevance triage failures",
                "",
                "These articles are excluded from the combined briefing because the "
                "model did not return valid relevance output after 3 attempts.",
            ]
        )
        for failure in relevance_failures:
            assert isinstance(failure, dict)
            summary = failure["summary"]
            invalid_outputs = failure["invalid_outputs"]
            assert isinstance(summary, dict)
            assert isinstance(invalid_outputs, list)
            lines.extend(
                [
                    "",
                    f"### {summary['title']}",
                    "",
                    str(summary["summary"]),
                    "",
                    "**Invalid relevance outputs**",
                    *[
                        f"{index}. `{output}`"
                        for index, output in enumerate(invalid_outputs, 1)
                    ],
                ]
            )
    return "\n".join(lines) + "\n"


def render_article_summary(summary: ArticleSummary) -> str:
    title = (
        f"[{summary.title}]({summary.url})"
        if summary.url
        else summary.title
    )
    original_article = (
        f"**Original article:** {summary.url}"
        if summary.url
        else "**Original article:** unavailable"
    )
    lines = [
        f"# {title}",
        "",
        original_article,
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
        (
            "**Summarization time:** "
            f"{format_seconds(summary.summarization_duration_seconds)}"
        ),
        "",
        f"*{summary.input_note}*",
    ]
    return "\n".join(lines) + "\n"


def format_seconds(duration_seconds: object) -> str:
    assert isinstance(duration_seconds, (int, float))
    return f"{duration_seconds:.1f} seconds"
