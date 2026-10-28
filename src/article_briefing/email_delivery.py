from __future__ import annotations

import html
import smtplib
import ssl
from email.message import EmailMessage
from time import perf_counter

import certifi

from article_briefing.config import SmtpSettings
from article_briefing.models import ArticleSummary, Briefing, RelevanceFailure


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
        f"Briefing generation time: {format_seconds(briefing.generation_duration_seconds)}",
        "",
        "OVERVIEW",
        briefing.overview.overview,
    ]
    if briefing.overview.error:
        lines.extend(
            [
                "",
                "BRIEFING OVERVIEW ERROR",
                f"The combined briefing could not be created: {briefing.overview.error}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "MAJOR THEMES",
                *[f"- {theme}" for theme in briefing.overview.major_themes],
                "",
                f"WATCH FOR: {briefing.overview.watch_for}",
                (
                    "Overview summarization time: "
                    f"{format_seconds(briefing.overview.summarization_duration_seconds)}"
                ),
            ]
        )
    lines.extend(["", "ARTICLE SUMMARIES"])
    for article in briefing.articles:
        lines.extend(
            [
                "",
                article.title,
                f"Original article: {article.url or 'unavailable'}",
                f"Summarization time: {format_seconds(article.summarization_duration_seconds)}",
                article.summary,
                "Key points:",
                *[f"- {point}" for point in article.key_points],
                f"Why it matters: {article.why_it_matters}",
                f"Read recommendation: {article.read_recommendation}",
                article.input_note,
            ]
        )
    if briefing.relevance_failures:
        lines.extend(
            [
                "",
                "RELEVANCE TRIAGE FAILURES",
                (
                    "These articles are excluded from the combined briefing because "
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
    theme_items = "".join(
        f"<li>{html.escape(theme)}</li>" for theme in briefing.overview.major_themes
    )
    article_sections = "".join(render_article_html(article) for article in briefing.articles)
    relevance_failure_sections = "".join(
        render_relevance_failure_html(failure)
        for failure in briefing.relevance_failures
    )
    overview_details = (
        "<h2>Briefing overview error</h2>"
        "<p>The combined briefing could not be created: "
        f"{html.escape(briefing.overview.error)}</p>"
        if briefing.overview.error
        else (
            "<h3>Major themes</h3>"
            f"<ul>{theme_items}</ul>"
            f"<p><strong>Watch for:</strong> {html.escape(briefing.overview.watch_for)}<br>"
            "<strong>Overview summarization time:</strong> "
            f"{html.escape(format_seconds(briefing.overview.summarization_duration_seconds))}</p>"
        )
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
    <strong>Briefing generation time:</strong>
    {html.escape(format_seconds(briefing.generation_duration_seconds))}</p>
    <h2>Overview</h2>
    <p>{html.escape(briefing.overview.overview)}</p>
    {overview_details}
    <h2>Article summaries</h2>
    {article_sections}
    {relevance_failure_sections}
    {skipped_sections}
  </body>
</html>
"""


def render_article_html(article: ArticleSummary) -> str:
    article_title = html.escape(article.title)
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
        f"WARNING: Relevance triage failed — {summary.title}",
        f"Original article: {summary.url or 'unavailable'}",
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
  <h2>Relevance triage warning: {html.escape(summary.title)}</h2>
  <p>This article is excluded from the combined briefing because the model returned
  invalid relevance output after 3 attempts.</p>
  <p>{source}<br>
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
