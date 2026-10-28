from __future__ import annotations

import sys
from pathlib import Path

from article_briefing.config import ConfigurationError, Settings
from article_briefing.email_delivery import EmailDeliveryError, SmtpBriefingSender
from article_briefing.ollama import OllamaClient, OllamaError
from article_briefing.processor import ProcessingError, process_landing_folder


def main() -> None:
    project_root = Path.cwd()
    try:
        settings = Settings.from_environment(project_root)
        sender = SmtpBriefingSender(settings.smtp) if settings.smtp.enabled else None
        result = process_landing_folder(settings, OllamaClient(settings), sender=sender)
    except (ConfigurationError, EmailDeliveryError, OllamaError, ProcessingError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    if result.briefing_path is None:
        if result.warnings:
            print(
                f"No articles were summarized from {settings.landing_dir}. "
                "Review the warnings and run the command again."
            )
            return
        print(
            f"No .txt files found in {settings.landing_dir}. "
            "Add landing files and run the command again."
        )
        return
    print(f"Created briefing: {result.briefing_path}")
    print(
        "Articles included: "
        f"{result.included_article_count} | "
        f"not included: {result.not_included_article_count}"
    )
    print(f"Briefing generation time: {result.generation_duration_seconds:.1f} seconds")
    print(f"Total process time: {result.total_duration_seconds:.1f} seconds")
    if result.email_delivery_duration_seconds is not None:
        print(f"SMTP delivery time: {result.email_delivery_duration_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
