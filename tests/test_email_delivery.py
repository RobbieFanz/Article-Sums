from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from article_briefing.config import SmtpSettings
from article_briefing.email_delivery import SmtpBriefingSender, render_plain_text
from article_briefing.models import ArticleSummary, Briefing, BriefingOverview


class SmtpBriefingSenderTests(unittest.TestCase):
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
        self.assertIn("12.5 seconds", message.get_body("plain").get_content())
        self.assertIn("<html>", message.get_body("html").get_content())

    def test_plain_text_includes_per_article_and_overall_timings(self) -> None:
        body = render_plain_text(make_briefing())

        self.assertIn("Briefing generation time: 12.5 seconds", body)
        self.assertIn("Overview summarization time: 1.5 seconds", body)
        self.assertIn("Summarization time: 3.2 seconds", body)


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
    )
    overview = BriefingOverview(
        overview="A concise overview.",
        major_themes=["Theme one.", "Theme two."],
        watch_for="Watch for follow-up.",
        summarization_duration_seconds=1.5,
    )
    return Briefing(
        created_at="2026-10-26_180000-0400",
        model="test-model",
        overview=overview,
        articles=[article],
        skipped_files=[],
        generation_duration_seconds=12.5,
    )


if __name__ == "__main__":
    unittest.main()
