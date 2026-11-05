from __future__ import annotations

import html
import smtplib
import ssl
from email.message import EmailMessage
from time import perf_counter

import certifi

from article_briefing.config import SmtpSettings
from article_briefing.models import (
    ArticleSummary,
    Briefing,
    FailureReport,
    RelevanceFailure,
    format_relevance_score,
)


class EmailDeliveryError(RuntimeError):
    """Raised when SMTP delivery cannot be completed."""


class SmtpBriefingSender:
    def __init__(self, settings: SmtpSettings) -> None:
        if not settings.enabled:
            raise ValueError("SMTP delivery cannot be created when SMTP_ENABLED is false.")
        self._settings = settings

    def send(self, briefing: Briefing) -> float:
        message = EmailMessage()
        message["Subject"] = f"Article Briefing — {briefing.created_at}"
        message["From"] = self._settings.from_address
        message["To"] = ", ".join(self._settings.to_addresses)
        message.set_content(render_plain_text(briefing))
        message.add_alternative(render_html(briefing), subtype="html")
        return self._send_message(message)

    def send_failure(self, failure: FailureReport) -> float:
        message = EmailMessage()
        message["Subject"] = "Article Briefing — process failed"
        message["From"] = self._settings.from_address
        message["To"] = ", ".join(self._settings.to_addresses)
        message.set_content(
            "ARTICLE BRIEFING PROCESS FAILED\n\n"
            f"Error: {failure.error}\n"
            f"Process runtime: {format_seconds(failure.duration_seconds)}\n"
            f"Articles processed: {failure.processed_article_count}\n"
            f"Article being processed: {failure.current_article or 'None (failure outside article processing)'}\n"
        )
        return self._send_message(message)

    def _send_message(self, message: EmailMessage) -> float:
        started_at = perf_counter()
        smtp_class = smtplib.SMTP_SSL if self._settings.use_ssl else smtplib.SMTP
        tls_context = ssl.create_default_context(cafile=certifi.where())
        try:
            with smtp_class(
                self._settings.host,
                self._settings.port,
                timeout=self._settings.timeout_seconds,
            ) as client:
                if self._settings.use_starttls:
                    client.starttls(context=tls_context)
                client.login(self._settings.username, self._settings.password)
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise EmailDeliveryError(f"Could not deliver briefing email: {error}") from error
        return perf_counter() - started_at


def render_plain_text(briefing: Briefing) -> str:
    lines = [
        "ARTICLE BRIEFING",
        f"Created: {briefing.created_at}",
        f"Model: {briefing.model}",
        f"Articles included: {briefing.included_article_count}",
        f"Articles not included: {briefing.not_included_article_count}",
        "These counts refer to the main briefing; excluded articles are listed below.",
        f"Briefing generation time: {format_seconds(briefing.generation_duration_seconds)}",
        "",
        "ARTICLE SUMMARIES",
    ]
    if any(article.chunk_limit_reached for article in briefing.articles) or any(
        failure.summary.chunk_limit_reached for failure in briefing.relevance_failures
    ):
        lines.extend(["", "* Chunk limit reached; remaining article text was not summarized."])
    for article in briefing.articles:
        lines.extend(
            [
                "",
                f"{article.title}{' *' if article.chunk_limit_reached else ''}",
                f"Original article: {article.url or 'unavailable'}",
                f"Relevance score: {format_relevance_score(article.relevance_score)}",
                f"Summarization time: {format_seconds(article.summarization_duration_seconds)}",
                article.summary,
                "Key points:",
                *[f"- {point}" for point in article.key_points],
                f"Why it matters: {article.why_it_matters}",
                f"Read recommendation: {article.read_recommendation}",
                article.input_note,
            ]
        )
    if briefing.articles_not_selected:
        lines.extend(["", "ARTICLES NOT SELECTED"])
        for article in briefing.articles_not_selected:
            lines.extend(
                [
                    "",
                    article.title,
                    f"Original article: {article.url or 'unavailable'}",
                    f"Reason: {article.reason}",
                ]
            )
    if briefing.relevance_failures:
        lines.extend(
            [
                "",
                "RELEVANCE TRIAGE FAILURES",
                (
                    "These articles are excluded from the main briefing because "
                    "the model returned invalid relevance output after 3 attempts."
                ),
            ]
        )
        for failure in briefing.relevance_failures:
            lines.extend(render_relevance_failure_plain_text(failure))
    if briefing.skipped_files:
        lines.extend(
            [
                "",
                "ARTICLES NOT INCLUDED",
                *[f"- {warning}" for warning in briefing.skipped_files],
            ]
        )
    return "\n".join(lines) + "\n"


def render_html(briefing: Briefing) -> str:
    article_sections = "".join(render_article_html(article) for article in briefing.articles)
    chunk_limit_note = (
        "<p>* Chunk limit reached; remaining article text was not summarized.</p>"
        if any(article.chunk_limit_reached for article in briefing.articles)
        or any(failure.summary.chunk_limit_reached for failure in briefing.relevance_failures)
        else ""
    )
    relevance_failure_sections = "".join(
        render_relevance_failure_html(failure)
        for failure in briefing.relevance_failures
    )
    not_selected_sections = (
        "<h2>Articles not selected</h2>"
        + "".join(
            "<section>"
            f"<h3>{html.escape(article.title)}</h3>"
            + (
                f'<p><a href="{html.escape(article.url, quote=True)}">Original article</a></p>'
                if article.url
                else "<p>Original article unavailable</p>"
            )
            + f"<p><strong>Reason:</strong> {html.escape(article.reason)}</p>"
            "</section>"
            for article in briefing.articles_not_selected
        )
        if briefing.articles_not_selected
        else ""
    )
    skipped_sections = (
        "<h2>Articles not included</h2><ul>"
        + "".join(f"<li>{html.escape(warning)}</li>" for warning in briefing.skipped_files)
        + "</ul>"
        if briefing.skipped_files
        else ""
    )
    return f"""\
<!doctype html>
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.5; color: #1f2937;">
    <h1>Article Briefing</h1>
    <p><strong>Created:</strong> {html.escape(briefing.created_at)}<br>
    <strong>Model:</strong> {html.escape(briefing.model)}<br>
    <strong>Articles included:</strong> {briefing.included_article_count}<br>
    <strong>Articles not included:</strong> {briefing.not_included_article_count}<br>
    These counts refer to the main briefing; excluded articles are listed below.<br>
    <strong>Briefing generation time:</strong>
    {html.escape(format_seconds(briefing.generation_duration_seconds))}</p>
    <h2>Article summaries</h2>
    {chunk_limit_note}
    {article_sections}
    {not_selected_sections}
    {relevance_failure_sections}
    {skipped_sections}
  </body>
</html>
"""


def render_article_html(article: ArticleSummary) -> str:
    article_title = html.escape(article.title) + (" *" if article.chunk_limit_reached else "")
    source = (
        f'<a href="{html.escape(article.url, quote=True)}">Original article</a>'
        if article.url
        else "Original article unavailable"
    )
    key_points = "".join(
        f"<li>{html.escape(point)}</li>" for point in article.key_points
    )
    return f"""\
<section style="margin-top: 24px;">
  <h3>{article_title}</h3>
  <p>{source}<br>
  <strong>Relevance score:</strong> {format_relevance_score(article.relevance_score)}<br>
  <strong>Summarization time:</strong>
  {html.escape(format_seconds(article.summarization_duration_seconds))}</p>
  <p>{html.escape(article.summary)}</p>
  <strong>Key points</strong>
  <ul>{key_points}</ul>
  <p><strong>Why it matters:</strong> {html.escape(article.why_it_matters)}<br>
  <strong>Read recommendation:</strong>
  {html.escape(article.read_recommendation)}</p>
  <p><em>{html.escape(article.input_note)}</em></p>
</section>
"""


def render_relevance_failure_plain_text(failure: RelevanceFailure) -> list[str]:
    summary = failure.summary
    lines = [
        "",
        f"WARNING: Relevance triage failed — {summary.title}"
        f"{' *' if summary.chunk_limit_reached else ''}",
        f"Original article: {summary.url or 'unavailable'}",
        f"Relevance score: {format_relevance_score(summary.relevance_score)}",
        f"Summarization time: {format_seconds(summary.summarization_duration_seconds)}",
        summary.summary,
        "Invalid relevance outputs:",
    ]
    lines.extend(
        f"{index}. {output}"
        for index, output in enumerate(failure.invalid_outputs, 1)
    )
    return lines


def render_relevance_failure_html(failure: RelevanceFailure) -> str:
    summary = failure.summary
    source = (
        f'<a href="{html.escape(summary.url, quote=True)}">Original article</a>'
        if summary.url
        else "Original article unavailable"
    )
    invalid_outputs = "".join(
        f"<li><pre>{html.escape(output)}</pre></li>"
        for output in failure.invalid_outputs
    )
    return f"""\
<section style="margin-top: 24px; border-top: 2px solid #b91c1c; padding-top: 16px;">
  <h2>Relevance triage warning: {html.escape(summary.title)}{' *' if summary.chunk_limit_reached else ''}</h2>
  <p>This article is excluded from the main briefing because the model returned
  invalid relevance output after 3 attempts.</p>
  <p>{source}<br>
  <strong>Relevance score:</strong> {format_relevance_score(summary.relevance_score)}<br>
  <strong>Summarization time:</strong>
  {html.escape(format_seconds(summary.summarization_duration_seconds))}</p>
  <p>{html.escape(summary.summary)}</p>
  <strong>Invalid relevance outputs</strong>
  <ol>{invalid_outputs}</ol>
</section>
"""


def format_seconds(duration_seconds: float) -> str:
    minutes, seconds = divmod(duration_seconds, 60)
    minute_label = "minute" if int(minutes) == 1 else "minutes"
    return f"{int(minutes)} {minute_label} {seconds:.1f} seconds"
