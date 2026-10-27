from __future__ import annotations

import unittest

from article_briefing.ollama import key_points_or_fallback


class OllamaResponseTests(unittest.TestCase):
    def test_missing_key_points_uses_fallback(self) -> None:
        self.assertEqual(
            key_points_or_fallback({}),
            ["No key points were found given."],
        )

    def test_empty_or_invalid_key_points_uses_fallback(self) -> None:
        self.assertEqual(
            key_points_or_fallback({"key_points": [None, "", 42]}),
            ["No key points were found given."],
        )

    def test_key_points_keeps_valid_strings_and_discards_invalid_values(self) -> None:
        self.assertEqual(
            key_points_or_fallback(
                {"key_points": [" First point ", None, "", "Second point"]}
            ),
            ["First point", "Second point"],
        )


if __name__ == "__main__":
    unittest.main()
