from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from article_briefing.config import SmtpSettings
from article_briefing.email_delivery import SmtpBriefingSender, render_plain_text
from article_briefing.models import (
    ArticleSummary,
    ArticleNotSelected,
    Briefing,
    FailureReport,
    RelevanceFailure,
)


class SmtpBriefingSenderTests(unittest.TestCase):
    @patch("article_briefing.email_delivery.smtplib.SMTP")
    def test_sends_failure_notification_with_progress(self, smtp_class: MagicMock) -> None:
        smtp_client = smtp_class.return_value.__enter__.return_value
        sender = SmtpBriefingSender(make_smtp_settings())
        sender.send_failure(FailureReport(
            error="OllamaError: model unavailable",
            duration_seconds=125.4,
            processed_article_count=3,
            current_article="failed.txt",
        ))
        message = smtp_client.send_message.call_args.args[0]
        self.assertIn("process failed", message["Subject"])
        body = message.get_body("plain").get_content()
        self.assertIn("OllamaError: model unavailable", body)
        self.assertIn("2 minutes 5.4 seconds", body)
        self.assertIn("Articles processed: 3", body)
        self.assertIn("failed.txt", body)

    @patch("article_briefing.email_delivery.smtplib.SMTP")
    def test_sends_plain_text_and_html_briefing(self, smtp_class: MagicMock) -> None:
        smtp_client = smtp_class.return_value.__enter__.return_value
        sender = SmtpBriefingSender(make_smtp_settings())

        duration = sender.send(make_briefing())

        self.assertGreaterEqual(duration, 0)
        smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=30)
        smtp_client.starttls.assert_called_once()
        smtp_client.login.assert_called_once_with("user@example.com", "app-password")
        smtp_client.send_message.assert_called_once()
        message = smtp_client.send_message.call_args.args[0]
        self.assertEqual(message["To"], "reader@example.com")
        self.assertIn("Article Briefing", message["Subject"])
        self.assertIn(
            "Briefing generation time: 0 minutes 12.5 seconds",
            message.get_body("plain").get_content(),
        )
        self.assertIn("<html>", message.get_body("html").get_content())

    def test_plain_text_includes_per_article_and_overall_timings(self) -> None:
        body = render_plain_text(make_briefing())

        self.assertIn("Briefing generation time: 0 minutes 12.5 seconds", body)
        self.assertNotIn("OVERVIEW", body)
        self.assertIn("Summarization time: 0 minutes 3.2 seconds", body)
        self.assertIn("Relevance score: 8/10", body)

    def test_plain_text_lists_articles_not_included(self) -> None:
        briefing = make_briefing()
        briefing = Briefing(
            created_at=briefing.created_at,
            model=briefing.model,
            articles=briefing.articles,
            skipped_files=["failed.txt could not be summarized after 2 attempts."],
            generation_duration_seconds=briefing.generation_duration_seconds,
        )

        body = render_plain_text(briefing)

        self.assertIn("ARTICLES NOT INCLUDED", body)
        self.assertIn("failed.txt could not be summarized after 2 attempts.", body)

    def test_plain_text_includes_counts_and_all_invalid_relevance_outputs(self) -> None:
        briefing = make_briefing()
        briefing = Briefing(
            created_at=briefing.created_at,
            model=briefing.model,
            articles=briefing.articles,
            skipped_files=[],
            generation_duration_seconds=briefing.generation_duration_seconds,
            relevance_failures=[
                RelevanceFailure(
                    summary=briefing.articles[0],
                    invalid_outputs=["Invalid output one", "Invalid output two"],
                )
            ],
            included_article_count=1,
            not_included_article_count=1,
        )

        body = render_plain_text(briefing)

        self.assertIn("Articles included: 1", body)
        self.assertIn("Articles not included: 1", body)
        self.assertIn("RELEVANCE TRIAGE FAILURES", body)
        self.assertIn("Invalid output one", body)
        self.assertIn("Invalid output two", body)

    def test_not_selected_articles_render_links_reasons_and_escape_html(self) -> None:
        from dataclasses import replace

        from article_briefing.email_delivery import render_html

        briefing = replace(
            make_briefing(),
            articles_not_selected=[
                ArticleNotSelected(
                    source_file="travel.txt",
                    title="Travel <story>",
                    url="https://example.com/story?a=1&b=2",
                    reason="Not relevant & outside your topics.",
                ),
                ArticleNotSelected(
                    source_file="missing.txt",
                    title="No URL",
                    url=None,
                    reason="Missing article text.",
                ),
            ],
        )
        text = render_plain_text(briefing)
        html = render_html(briefing)

        self.assertIn("https://example.com/story?a=1&b=2", text)
        self.assertIn("Reason: Not relevant & outside your topics.", text)
        self.assertIn("Original article: unavailable", text)
        self.assertIn('href="https://example.com/story?a=1&amp;b=2"', html)
        self.assertIn("Travel &lt;story&gt;", html)
        self.assertIn("Not relevant &amp; outside your topics.", html)
        self.assertIn("Original article unavailable", html)


def make_smtp_settings() -> SmtpSettings:
    return SmtpSettings(
        enabled=True,
        host="smtp.example.com",
        port=587,
        username="user@example.com",
        password="app-password",
        from_address="sender@example.com",
        to_addresses=("reader@example.com",),
        use_starttls=True,
        timeout_seconds=30,
    )


def make_briefing() -> Briefing:
    article = ArticleSummary(
        title="Example Article",
        url="https://example.com/article",
        source_file="example.txt",
        summary="An article summary.",
        key_points=["First point.", "Second point."],
        why_it_matters="It provides useful context.",
        read_recommendation="Read the original article.",
        input_note="The summary is based on the supplied article text.",
        summarization_duration_seconds=3.2,
        relevance_score=8,
    )
    return Briefing(
        created_at="2026-10-26_180000-0400",
        model="test-model",
        articles=[article],
        skipped_files=[],
        generation_duration_seconds=12.5,
        included_article_count=1,
        not_included_article_count=0,
    )


if __name__ == "__main__":
    unittest.main()
