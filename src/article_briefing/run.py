from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from article_briefing.config import (
    ConfigurationError,
    load_dotenv,
    positive_integer,
    resolve_project_path,
)
from article_briefing.hacker_news import DEFAULT_STORY_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Hacker News articles and create a briefing."
    )
    parser.parse_args()

    try:
        project_root = Path.cwd()
        load_dotenv(project_root / ".env")
        limit = positive_integer("HN_ARTICLE_LIMIT", DEFAULT_STORY_LIMIT)
        landing_dir = resolve_project_path(
            project_root, os.getenv("LANDING_DIR", "data/landing")
        )
    except (ConfigurationError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    stages = (
        (
            "Collecting Hacker News articles",
            [
                sys.executable, "-m", "article_briefing.hacker_news_collect",
                "--limit", str(limit), "--landing-dir", str(landing_dir),
            ],
        ),
        ("Creating article briefing", [sys.executable, "-m", "article_briefing"]),
    )
    for label, command in stages:
        print(f"{label}...", flush=True)
        try:
            result = subprocess.run(command, check=False)
        except OSError as error:
            print(f"Error: Could not start {label.lower()}: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        if result.returncode:
            print(f"Error: {label} failed (exit code {result.returncode}).", file=sys.stderr)
            raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)


if __name__ == "__main__":
    main()
