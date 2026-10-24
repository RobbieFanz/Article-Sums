from __future__ import annotations

import sys
from pathlib import Path

from article_briefing.config import ConfigurationError, Settings
from article_briefing.ollama import OllamaClient, OllamaError
from article_briefing.processor import ProcessingError, process_landing_folder


def main() -> None:
    project_root = Path.cwd()
    try:
        settings = Settings.from_environment(project_root)
        result = process_landing_folder(settings, OllamaClient(settings))
    except (ConfigurationError, OllamaError, ProcessingError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    if result.briefing_path is None:
        if result.warnings:
            print(
                f"No valid .txt files found in {settings.landing_dir}. "
                "Correct the warned files and run the command again."
            )
            return
        print(
            f"No .txt files found in {settings.landing_dir}. "
            "Add landing files and run the command again."
        )
        return
    print(f"Created briefing: {result.briefing_path}")


if __name__ == "__main__":
    main()
