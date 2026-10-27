from __future__ import annotations

import html
import smtplib
import ssl
from email.message import EmailMessage
from time import perf_counter

import certifi

from article_briefing.config import SmtpSettings
from article_briefing.models import ArticleSummary, Briefing


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
        f"Briefing generation time: {format_seconds(briefing.generation_duration_seconds)}",
        "",
        "OVERVIEW",
        briefing.overview.overview,
        "",
        "MAJOR THEMES",
        *[f"- {theme}" for theme in briefing.overview.major_themes],
        "",
        f"WATCH FOR: {briefing.overview.watch_for}",
        (
            "Overview summarization time: "
            f"{format_seconds(briefing.overview.summarization_duration_seconds)}"
        ),
        "",
        "ARTICLE SUMMARIES",
    ]
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
    return "\n".join(lines) + "\n"


def render_html(briefing: Briefing) -> str:
    theme_items = "".join(
        f"<li>{html.escape(theme)}</li>" for theme in briefing.overview.major_themes
    )
    article_sections = "".join(render_article_html(article) for article in briefing.articles)
    return f"""\
<!doctype html>
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.5; color: #1f2937;">
    <h1>Article Briefing</h1>
    <p><strong>Created:</strong> {html.escape(briefing.created_at)}<br>
    <strong>Model:</strong> {html.escape(briefing.model)}<br>
    <strong>Briefing generation time:</strong>
    {html.escape(format_seconds(briefing.generation_duration_seconds))}</p>
    <h2>Overview</h2>
    <p>{html.escape(briefing.overview.overview)}</p>
    <h3>Major themes</h3>
    <ul>{theme_items}</ul>
    <p><strong>Watch for:</strong> {html.escape(briefing.overview.watch_for)}<br>
    <strong>Overview summarization time:</strong>
    {html.escape(format_seconds(briefing.overview.summarization_duration_seconds))}</p>
    <h2>Article summaries</h2>
    {article_sections}
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


def format_seconds(duration_seconds: float) -> str:
    minutes, seconds = divmod(duration_seconds, 60)
    minute_label = "minute" if int(minutes) == 1 else "minutes"
    return f"{int(minutes)} {minute_label} {seconds:.1f} seconds"
