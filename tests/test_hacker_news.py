from __future__ import annotations

import unittest
from typing import Any

from article_briefing.hacker_news import HackerNewsClient, parse_story, render_stories


class FakeHackerNewsClient(HackerNewsClient):
    def __init__(self, responses: dict[str, Any]) -> None:
        super().__init__()
        self.responses = responses

    def _get_json(self, path: str) -> Any:
        return self.responses[path]


class HackerNewsTests(unittest.TestCase):
    def test_top_stories_skips_deleted_and_non_story_items(self) -> None:
        client = FakeHackerNewsClient(
            {
                "/topstories.json": [1, 2, 3, 4],
                "/item/1.json": {"id": 1, "type": "story", "deleted": True},
                "/item/2.json": {"id": 2, "type": "job", "title": "Ignored job"},
                "/item/3.json": {
                    "id": 3,
                    "type": "story",
                    "title": "External story",
                    "score": 50,
                    "descendants": 20,
                    "by": "alice",
                    "url": "https://example.com/story",
                },
                "/item/4.json": {
                    "id": 4,
                    "type": "story",
                    "title": "Ask HN: A question",
                    "score": 40,
                    "descendants": 10,
                    "by": "bob",
                },
            }
        )

        stories = client.top_stories(limit=2)

        self.assertEqual([story.id for story in stories], [3, 4])
        self.assertEqual(stories[1].story_type, "Ask HN")

    def test_rendered_stories_include_hacker_news_and_article_links(self) -> None:
        story = parse_story(
            {
                "id": 42,
                "type": "story",
                "title": "A story",
                "score": 25,
                "descendants": 5,
                "by": "carol",
                "url": "https://example.com/article",
            }
        )

        assert story is not None
        rendered = render_stories([story])

        self.assertIn("1. A story", rendered)
        self.assertIn("https://example.com/article", rendered)
        self.assertIn("https://news.ycombinator.com/item?id=42", rendered)

    def test_parse_story_rejects_missing_title(self) -> None:
        self.assertIsNone(parse_story({"id": 1, "type": "story"}))


if __name__ == "__main__":
    unittest.main()
