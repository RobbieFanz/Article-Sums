from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

from article_briefing.config import ConfigurationError, Settings, SmtpSettings, load_dotenv
from article_briefing.email_delivery import EmailDeliveryError, SmtpBriefingSender
from article_briefing.ollama import OllamaClient, OllamaError
from article_briefing.processor import (
    ProcessingError,
    ProcessingProgress,
    notify_failure,
    process_landing_folder,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting article briefing; loading configuration.")
    project_root = Path.cwd()
    started_at = perf_counter()
    try:
        load_dotenv(project_root / ".env")
        smtp = SmtpSettings.from_environment()
        logger.info("SMTP delivery is %s.", "enabled" if smtp.enabled else "disabled")
        sender = SmtpBriefingSender(smtp) if smtp.enabled else None
        try:
            settings = Settings.from_loaded_environment(project_root, smtp=smtp)
            logger.info("Configured Ollama model: %s", settings.ollama_model)
            summarizer = OllamaClient(settings)
        except Exception as error:
            if sender is not None:
                notify_failure(sender, error, started_at, ProcessingProgress())
            raise
        result = process_landing_folder(settings, summarizer, sender=sender)
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
