from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from article_briefing import __main__ as briefing_command
from article_briefing import hacker_news_collect, run
from article_briefing.hacker_news import HackerNewsStory
from article_briefing.hacker_news_collect import DownloadedPage
from article_briefing.models import Article, ArticleSummary, RelevanceDecision


class CombinedRunTests(unittest.TestCase):
    def test_runs_collection_then_briefing_using_dotenv_and_custom_landing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "HN_ARTICLE_LIMIT=23\nLANDING_DIR=custom/landing\n", encoding="utf-8"
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(run.Path, "cwd", return_value=root),
                patch.object(sys, "argv", ["article-briefing-run"]),
                patch("article_briefing.run.subprocess.run") as execute,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                execute.return_value = subprocess.CompletedProcess([], 0)
                run.main()

            self.assertEqual(execute.call_count, 2)
            self.assertEqual(
                execute.call_args_list[0].args[0],
                [
                    sys.executable, "-m", "article_briefing.hacker_news_collect",
                    "--limit", "23", "--landing-dir", str(root / "custom/landing"),
                ],
            )
            self.assertEqual(
                execute.call_args_list[1].args[0], [sys.executable, "-m", "article_briefing"]
            )

    def test_default_and_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("HN_ARTICLE_LIMIT=19\n", encoding="utf-8")
            for environment, expected in (({}, "19"), ({"HN_ARTICLE_LIMIT": "5"}, "5")):
                with (
                    self.subTest(environment=environment),
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(run.Path, "cwd", return_value=root),
                    patch.object(sys, "argv", ["run"]),
                    patch("article_briefing.run.subprocess.run") as execute,
                    patch("sys.stdout", new_callable=io.StringIO),
                ):
                    execute.return_value = subprocess.CompletedProcess([], 0)
                    run.main()
                    self.assertEqual(execute.call_args_list[0].args[0][4], expected)
            (root / ".env").unlink()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(run.Path, "cwd", return_value=root),
                patch.object(sys, "argv", ["run"]),
                patch("article_briefing.run.subprocess.run") as execute,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                execute.return_value = subprocess.CompletedProcess([], 0)
                run.main()
                self.assertEqual(execute.call_args_list[0].args[0][4], "10")

    def test_invalid_limit_fails_before_starting_any_stage(self) -> None:
        for value in ("0", "-1", "abc", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with (
                    patch.dict(os.environ, {"HN_ARTICLE_LIMIT": value}, clear=True),
                    patch.object(run.Path, "cwd", return_value=Path(directory)),
                    patch.object(sys, "argv", ["run"]),
                    patch("article_briefing.run.subprocess.run") as execute,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises(SystemExit) as exit_result,
                ):
                    run.main()
                self.assertEqual(exit_result.exception.code, 1)
                self.assertIn("HN_ARTICLE_LIMIT", stderr.getvalue())
                execute.assert_not_called()

    def test_failed_stage_stops_and_propagates_exit_status(self) -> None:
        for codes in ((2,), (0, 3)):
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as directory:
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(run.Path, "cwd", return_value=Path(directory)),
                    patch.object(sys, "argv", ["run"]),
                    patch("article_briefing.run.subprocess.run") as execute,
                    patch("sys.stdout", new_callable=io.StringIO),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises(SystemExit) as exit_result,
                ):
                    execute.side_effect = [subprocess.CompletedProcess([], code) for code in codes]
                    run.main()
                self.assertEqual(exit_result.exception.code, codes[-1])
                self.assertEqual(execute.call_count, len(codes))
                self.assertIn(f"exit code {codes[-1]}", stderr.getvalue())

    def test_launch_error_is_reported(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch.object(run.Path, "cwd", return_value=Path(directory)),
            patch.object(sys, "argv", ["run"]),
            patch("article_briefing.run.subprocess.run", side_effect=OSError("cannot launch")),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as exit_result,
        ):
            run.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertIn("cannot launch", stderr.getvalue())

    def test_real_collection_and_processing_share_configured_paths(self) -> None:
        class FakeClient:
            def __init__(self, timeout_seconds: float) -> None:
                self.limit: int | None = None

            def top_stories(self, limit: int) -> list[HackerNewsStory]:
                self.limit = limit
                return [HackerNewsStory(
                    id=1, title="Example", score=1, descendants=0, by="author",
                    url="https://example.com/article", story_type="Linked article",
                )]

        class FakeSummarizer:
            def assess_relevance(self, article: Article) -> RelevanceDecision:
                return RelevanceDecision(True, "Relevant", 0.0)

            def summarize_article(self, article: Article) -> ArticleSummary:
                return ArticleSummary(
                    title=article.title, url=article.url, source_file=article.source_file,
                    summary="Summary", key_points=["Point"], why_it_matters="Reason",
                    read_recommendation="Read", input_note="Input",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "criteria.txt").write_text("Technology", encoding="utf-8")
            (root / ".env").write_text(
                "HN_ARTICLE_LIMIT=3\nLANDING_DIR=custom/landing\n"
                "ARTICLE_HISTORY_PATH=custom/history.sqlite3\n"
                "OLLAMA_MODEL=test\nRELEVANCE_CRITERIA_FILE=criteria.txt\n",
                encoding="utf-8",
            )
            client = FakeClient(timeout_seconds=15)

            def execute(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
                with patch.object(sys, "argv", [command[2], *command[3:]]):
                    if command[2] == "article_briefing.hacker_news_collect":
                        hacker_news_collect.main()
                    else:
                        briefing_command.main()
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(run.Path, "cwd", return_value=root),
                patch.object(sys, "argv", ["run"]),
                patch("article_briefing.run.subprocess.run", side_effect=execute),
                patch("article_briefing.hacker_news_collect.HackerNewsClient", return_value=client),
                patch("article_briefing.hacker_news_collect.download_page",
                      return_value=DownloadedPage("https://example.com/article", "<html>text</html>")),
                patch("article_briefing.hacker_news_collect.extract_article_text", return_value="x" * 500),
                patch("article_briefing.__main__.OllamaClient", return_value=FakeSummarizer()),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                run.main()

            self.assertEqual(client.limit, 3)
            self.assertTrue((root / "custom" / "history.sqlite3").is_file())
            self.assertEqual(len(list((root / "data/archive").glob("*/*.txt"))), 1)
            self.assertEqual(len(list((root / "data/briefings").glob("*/briefing.md"))), 1)
            self.assertEqual(list((root / "custom/landing").glob("*.txt")), [])


if __name__ == "__main__":
    unittest.main()
