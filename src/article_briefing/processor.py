from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol
from urllib.parse import urlparse

from article_briefing.config import Settings
from article_briefing.history import ArticleHistory, DEFAULT_HISTORY_PATH, HistoryError
from article_briefing.models import (
    Article,
    ArticleNotSelected,
    ArticleSummary,
    Briefing,
    FailureReport,
    RelevanceDecision,
    RelevanceFailure,
    format_relevance_score,
)
from article_briefing.ollama import OllamaError, OllamaOutputError

logger = logging.getLogger(__name__)


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
        logger.info("Sending failure notification email.")
        sender.send_failure(failure)
        logger.info("Failure notification email sent.")
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
        logger.info(
            "Briefing failed after %d processed article(s); active article: %s; error: %s",
            progress.processed_article_count,
            progress.current_article or "none",
            error,
        )
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
    logger.info("Found %d landing article(s) in %s.", len(landing_files), settings.landing_dir)
    if not landing_files:
        return ProcessingResult(briefing_path=None, warnings=())

    history = ArticleHistory(settings.history_path or settings.project_root / DEFAULT_HISTORY_PATH)
    articles: list[Article] = []
    processed_files: list[Path] = []
    failed_files: list[Path] = []
    failed_urls: list[tuple[str, bool | None, str]] = []
    warnings: list[str] = []
    articles_not_selected: list[ArticleNotSelected] = []
    for path in landing_files:
        logger.info("Loading article: %s", path.name)
        progress.current_article = path.name
        try:
            article = load_article(
                path,
                None if settings.chunked_summarization else settings.max_article_characters,
            )
        except ProcessingError as error:
            logger.info("Skipping invalid article %s: %s", path.name, error)
            warnings.append(str(error))
            failed_files.append(path)
            url = history.url_for_landing_file(path) or recover_article_url(path)
            if url:
                failed_urls.append((url, None, "Invalid landing file"))
            articles_not_selected.append(
                ArticleNotSelected(
                    source_file=path.name,
                    title=path.stem.replace("_", " ").replace("-", " ").strip(),
                    url=url,
                    reason=str(error),
                )
            )
            progress.processed_article_count += 1
            progress.current_article = None
            continue
        articles.append(article)
        if article.url and not history.contains(article.url):
            history.record_collected(article.url, article.url, path)
        logger.info(
            "Loaded %s: %d characters%s.",
            path.name,
            len(article.text),
            " (truncated)" if article.was_truncated else "",
        )
        processed_files.append(path)
        progress.current_article = None

    summaries: list[ArticleSummary] = []
    relevance_failures: list[RelevanceFailure] = []
    archived_files: list[Path] = []
    processed_urls: list[tuple[str, bool | None]] = []
    not_included_article_count = len(articles_not_selected)
    for article, source_file in zip(articles, processed_files, strict=True):
        progress.current_article = source_file.name
        relevance_decision, invalid_outputs = assess_relevance_with_retry(
            summarizer, article
        )
        if relevance_decision is None:
            logger.info(
                "Relevance check failed for %s; attempting a fallback summary.",
                source_file.name,
            )
            try:
                summary = summarize_article_with_retry(summarizer, article)
            except OllamaError as error:
                warning = f"{source_file.name} could not be summarized after 2 attempts: {error}"
                warnings.append(warning)
                logger.info("%s", warning)
                failed_files.append(source_file)
                if article.url:
                    failed_urls.append((article.url, None, "Relevance and summary attempts failed"))
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
            logger.info("Saved fallback summary for %s outside the main briefing.", source_file.name)
            archived_files.append(source_file)
            if article.url:
                processed_urls.append((article.url, None))
            not_included_article_count += 1
            progress.processed_article_count += 1
            progress.current_article = None
            continue

        if not relevance_decision.include:
            logger.info("Rejected %s: %s", source_file.name, relevance_decision.reason)
            articles_not_selected.append(
                ArticleNotSelected(
                    source_file=article.source_file,
                    title=article.title,
                    url=article.url,
                    reason=relevance_decision.reason,
                )
            )
            archived_files.append(source_file)
            if article.url:
                processed_urls.append((article.url, False))
            not_included_article_count += 1
            progress.processed_article_count += 1
            progress.current_article = None
            continue

        try:
            logger.info("Selected %s: %s", source_file.name, relevance_decision.reason)
            summaries.append(summarize_article_with_retry(summarizer, article))
        except OllamaError as error:
            warning = f"{source_file.name} could not be summarized after 2 attempts: {error}"
            warnings.append(warning)
            logger.info("%s", warning)
            failed_files.append(source_file)
            if article.url:
                failed_urls.append((article.url, True, "Summary failed after 2 attempts"))
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
        if article.url:
            processed_urls.append((article.url, True))
        logger.info("Summary accepted for %s.", source_file.name)
        progress.processed_article_count += 1
        progress.current_article = None

    summaries.sort(key=lambda summary: -(summary.relevance_score or 0))

    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H%M%S%z")
    briefing_path = settings.briefing_dir / timestamp
    archive_path = settings.archive_dir / timestamp
    failed_path = settings.archive_dir / "failed" / timestamp
    if briefing_path.exists() or archive_path.exists() or (failed_files and failed_path.exists()):
        raise ProcessingError(
            f"Briefing timestamp collision at {timestamp}. Rerun the command."
        )

    generation_duration_seconds = perf_counter() - started_at
    briefing = Briefing(
        created_at=timestamp,
        model=settings.ollama_model,
        articles=summaries,
        skipped_files=warnings,
        generation_duration_seconds=generation_duration_seconds,
        relevance_failures=relevance_failures,
        articles_not_selected=articles_not_selected,
        included_article_count=len(summaries),
        not_included_article_count=not_included_article_count,
    )
    create_briefing_directory(briefing_path)
    logger.info("Writing briefing output to %s.", briefing_path)
    write_briefing(briefing_path, briefing)
    if sender:
        logger.info("Sending briefing email.")
        email_delivery_duration_seconds = sender.send(briefing)
        logger.info("Briefing email sent in %.1f seconds.", email_delivery_duration_seconds)
    else:
        email_delivery_duration_seconds = None
        logger.info("Skipping email delivery because SMTP is disabled.")
    moved_files, created_directories = archive_files(
        [(archived_files, archive_path), (failed_files, failed_path)], progress
    )
    completed_briefing = Briefing(
        created_at=briefing.created_at,
        model=briefing.model,
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
    try:
        logger.info("Updating briefing output with completion timings.")
        write_briefing(briefing_path, completed_briefing)
        history.record_outcomes(processed_urls, failed_urls)
    except Exception as error:
        failures = restore_archived_files(moved_files, created_directories)
        if failures:
            raise ProcessingError(
                f"Batch completion failed: {error}. Could not restore all files: {'; '.join(failures)}."
            ) from error
        if isinstance(error, HistoryError):
            raise ProcessingError(f"Could not save article history: {error}") from error
        raise
    logger.info("Briefing processing completed.")
    return ProcessingResult(
        briefing_path=briefing_path,
        warnings=tuple(warnings),
        generation_duration_seconds=completed_briefing.generation_duration_seconds,
        total_duration_seconds=completed_briefing.total_duration_seconds,
        email_delivery_duration_seconds=email_delivery_duration_seconds,
        included_article_count=completed_briefing.included_article_count,
        not_included_article_count=completed_briefing.not_included_article_count,
    )


def load_article(path: Path, maximum_characters: int | None) -> Article:
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

    was_truncated = maximum_characters is not None and len(text) > maximum_characters
    return Article(
        source_file=path.name,
        title=path.stem.replace("_", " ").replace("-", " ").strip(),
        url=url,
        text=text[:maximum_characters],
        was_truncated=was_truncated,
    )


def recover_article_url(path: Path) -> str | None:
    try:
        with path.open("rb") as source:
            raw_line = source.readline().removeprefix(b"\xef\xbb\xbf")
            while raw_line and not raw_line.strip():
                raw_line = source.readline()
            first_line = raw_line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if first_line.lower().startswith("url:"):
        first_line = first_line[4:].strip()
    return first_line if is_complete_url(first_line) else None


def summarize_article_with_retry(summarizer: Summarizer, article: Article) -> ArticleSummary:
    last_error: OllamaError | None = None
    for attempt in range(1, 3):
        logger.info("Summarizing %s (attempt %d/2).", article.source_file, attempt)
        try:
            summary = summarizer.summarize_article(article)
            logger.info("Summary succeeded for %s (attempt %d/2).", article.source_file, attempt)
            return summary
        except OllamaError as error:
            logger.info("Summary attempt %d/2 failed for %s: %s", attempt, article.source_file, error)
            last_error = error
    assert last_error is not None
    raise last_error


def assess_relevance_with_retry(
    summarizer: Summarizer, article: Article
) -> tuple[RelevanceDecision | None, list[str]]:
    invalid_outputs: list[str] = []
    for attempt in range(1, 4):
        logger.info("Checking relevance of %s (attempt %d/3).", article.source_file, attempt)
        try:
            decision = summarizer.assess_relevance(article)
            logger.info("Relevance check succeeded for %s (attempt %d/3).", article.source_file, attempt)
            return decision, invalid_outputs
        except OllamaOutputError as error:
            logger.info("Relevance attempt %d/3 failed for %s: %s", attempt, article.source_file, error)
            invalid_outputs.append(error.raw_output)
    return None, invalid_outputs


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
    try:
        parsed_url = urlparse(value)
    except ValueError:
        return False
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
    groups: list[tuple[list[Path], Path]], progress: ProcessingProgress
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    moved_files: list[tuple[Path, Path]] = []
    created_directories: list[Path] = []
    try:
        for landing_files, archive_path in groups:
            if not landing_files:
                continue
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.mkdir()
            created_directories.append(archive_path)
            for source_path in landing_files:
                progress.current_article = source_path.name
                logger.info("Archiving article: %s", source_path.name)
                destination_path = archive_path / source_path.name
                source_path.replace(destination_path)
                moved_files.append((source_path, destination_path))
        progress.current_article = None
    except OSError as error:
        failures = restore_archived_files(moved_files, created_directories)
        detail = f" Could not restore all files: {'; '.join(failures)}." if failures else ""
        raise ProcessingError(f"Could not archive landing files: {error}.{detail}") from error
    return moved_files, created_directories


def restore_archived_files(
    moved_files: list[tuple[Path, Path]], created_directories: list[Path]
) -> list[str]:
    failures: list[str] = []
    for source_path, destination_path in reversed(moved_files):
        try:
            if source_path.exists():
                raise OSError(f"Cannot restore {source_path}: file already exists")
            destination_path.replace(source_path)
        except OSError as error:
            failures.append(str(error))
    for directory in reversed(created_directories):
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
        except OSError as error:
            failures.append(str(error))
    return failures


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
    ]
    articles = payload["articles"]
    assert isinstance(articles, list)
    lines.extend(["", "## Article summaries", ""])
    if articles:
        for article in articles:
            assert isinstance(article, dict)
            score = article["relevance_score"]
            assert score is None or isinstance(score, int)
            filename = f"{Path(article['source_file']).stem}.md"
            lines.append(
                f"- [{article['title']}](<article-summaries/{filename}>) "
                f"— Relevance: {format_relevance_score(score)}"
            )
    else:
        lines.append("No article summaries are available.")
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
                "These articles are excluded from the main briefing because the "
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
                    f"**Relevance score:** {format_relevance_score(summary['relevance_score'])}",
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
        f"**Relevance score:** {format_relevance_score(summary.relevance_score)}",
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
