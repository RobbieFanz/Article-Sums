from __future__ import annotations

import argparse
import json
import ssl
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


HACKER_NEWS_API_URL = "https://hacker-news.firebaseio.com/v0"
DEFAULT_STORY_LIMIT = 10
DEFAULT_TIMEOUT_SECONDS = 15


class HackerNewsError(RuntimeError):
    """Raised when the Hacker News API cannot return usable data."""


@dataclass(frozen=True)
class HackerNewsStory:
    id: int
    title: str
    score: int
    descendants: int
    by: str
    url: str | None
    story_type: str


class HackerNewsClient:
    def __init__(
        self,
        api_url: str = HACKER_NEWS_API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def top_stories(self, limit: int) -> list[HackerNewsStory]:
        story_ids = self._get_json("/topstories.json")
        if not isinstance(story_ids, list) or not all(
            isinstance(story_id, int) for story_id in story_ids
        ):
            raise HackerNewsError("Hacker News returned invalid top-story IDs.")

        stories: list[HackerNewsStory] = []
        for story_id in story_ids:
            try:
                story = self._get_json(f"/item/{story_id}.json")
            except HackerNewsError:
                continue
            parsed_story = parse_story(story)
            if parsed_story is None:
                continue
            stories.append(parsed_story)
            if len(stories) == limit:
                break

        return stories

    def _get_json(self, path: str) -> Any:
        request = Request(
            f"{self._api_url}{path}",
            headers={"Accept": "application/json", "User-Agent": "ArticleBriefing/0.1"},
        )
        try:
            with urlopen(
                request,
                timeout=self._timeout_seconds,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")[:500]
            raise HackerNewsError(
                f"Hacker News returned HTTP {error.code}: {details}"
            ) from error
        except URLError as error:
            raise HackerNewsError(
                f"Could not reach the Hacker News API: {error.reason}"
            ) from error
        except UnicodeDecodeError as error:
            raise HackerNewsError(
                "Hacker News returned data that was not valid UTF-8."
            ) from error

        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise HackerNewsError("Hacker News returned invalid JSON.") from error


def parse_story(payload: Any) -> HackerNewsStory | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("deleted") is True or payload.get("dead") is True:
        return None
    if payload.get("type") != "story":
        return None

    story_id = payload.get("id")
    title = payload.get("title")
    if not isinstance(story_id, int) or not isinstance(title, str) or not title.strip():
        return None

    url = payload.get("url")
    return HackerNewsStory(
        id=story_id,
        title=title.strip(),
        score=nonnegative_integer(payload.get("score")),
        descendants=nonnegative_integer(payload.get("descendants")),
        by=string_or_default(payload.get("by"), "unknown"),
        url=url if isinstance(url, str) and url.strip() else None,
        story_type="Ask HN" if url is None else "Linked article",
    )


def nonnegative_integer(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def string_or_default(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def render_stories(stories: list[HackerNewsStory]) -> str:
    if not stories:
        return "No readable Hacker News top stories were returned."

    lines = ["Hacker News top stories", ""]
    for rank, story in enumerate(stories, 1):
        lines.extend(
            [
                f"{rank}. {story.title}",
                f"   {story.score} points | {story.descendants} comments | by {story.by}",
                f"   {story.story_type}: {story.url or f'https://news.ycombinator.com/item?id={story.id}'}",
                f"   Hacker News: https://news.ycombinator.com/item?id={story.id}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print current Hacker News top stories without storing or summarizing them."
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        default=DEFAULT_STORY_LIMIT,
        help=f"Number of stories to print (default: {DEFAULT_STORY_LIMIT}).",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    arguments = parse_arguments()
    try:
        stories = HackerNewsClient(timeout_seconds=arguments.timeout).top_stories(
            arguments.limit
        )
    except HackerNewsError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(render_stories(stories))


if __name__ == "__main__":
    main()
