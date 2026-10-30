from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Generator


DEFAULT_HISTORY_PATH = Path("data/article-history.sqlite3")


class HistoryError(RuntimeError):
    """Raised when article history cannot be read or updated."""


class ArticleHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('collected', 'missing', 'processed')),
                    relevance INTEGER CHECK (relevance IN (0, 1) OR relevance IS NULL),
                    landing_path TEXT,
                    collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS article_urls (
                    url TEXT PRIMARY KEY,
                    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE
                );
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as connection:
                with connection:
                    connection.execute("PRAGMA foreign_keys = ON")
                    yield connection
        except (OSError, sqlite3.Error) as error:
            raise HistoryError(f"Could not access article history {self.path}: {error}") from error

    def contains(self, url: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT articles.id, articles.status, articles.landing_path
                   FROM articles JOIN article_urls ON articles.id = article_urls.article_id
                   WHERE article_urls.url = ?""",
                (url,),
            ).fetchone()
            if row is None:
                return False
            article_id, status, landing_path = row
            if status == "collected" and (not landing_path or not Path(landing_path).is_file()):
                connection.execute(
                    "UPDATE articles SET status = 'missing', landing_path = NULL WHERE id = ?",
                    (article_id,),
                )
                return False
            return status != "missing"

    def add_alias(self, known_url: str, alias: str) -> None:
        with self._connection() as connection:
            known = connection.execute(
                "SELECT article_id FROM article_urls WHERE url = ?", (known_url,)
            ).fetchone()
            existing = connection.execute(
                """SELECT articles.id, articles.status FROM articles
                   JOIN article_urls ON articles.id = article_urls.article_id
                   WHERE article_urls.url = ?""",
                (alias,),
            ).fetchone()
            if known is None:
                raise HistoryError(f"Cannot add alias: article URL {known_url} is not in history.")
            if existing is not None and existing[0] != known[0]:
                if existing[1] != "missing":
                    raise HistoryError(f"Cannot associate {alias} with two different articles.")
                connection.execute(
                    """UPDATE articles SET collected_at = MIN(collected_at,
                       (SELECT collected_at FROM articles WHERE id = ?)) WHERE id = ?""",
                    (existing[0], known[0]),
                )
                connection.execute(
                    "UPDATE article_urls SET article_id = ? WHERE article_id = ?",
                    (known[0], existing[0]),
                )
                connection.execute("DELETE FROM articles WHERE id = ?", (existing[0],))
            connection.execute(
                """INSERT OR IGNORE INTO article_urls (url, article_id)
                   SELECT ?, article_id FROM article_urls WHERE url = ?""",
                (alias, known_url),
            )

    def record_collected(self, original_url: str, final_url: str, landing_path: Path) -> None:
        with self._connection() as connection:
            existing = connection.execute(
                """SELECT DISTINCT articles.id, articles.status FROM articles
                   JOIN article_urls ON articles.id = article_urls.article_id
                   WHERE article_urls.url IN (?, ?)""",
                (original_url, final_url),
            ).fetchall()
            if len(existing) > 1 or (existing and existing[0][1] != "missing"):
                raise HistoryError("Cannot record collection: URL already belongs to another article.")
            if existing:
                article_id = existing[0][0]
                connection.execute(
                    "UPDATE articles SET status = 'collected', landing_path = ? WHERE id = ?",
                    (str(landing_path.resolve()), article_id),
                )
            else:
                cursor = connection.execute(
                    "INSERT INTO articles (status, landing_path) VALUES ('collected', ?)",
                    (str(landing_path.resolve()),),
                )
                article_id = cursor.lastrowid
            for url in {original_url, final_url}:
                connection.execute(
                    "INSERT OR IGNORE INTO article_urls (url, article_id) VALUES (?, ?)",
                    (url, article_id),
                )

    def record_processed(self, articles: list[tuple[str, bool | None]]) -> None:
        with self._connection() as connection:
            for url, relevance in articles:
                row = connection.execute(
                    "SELECT article_id FROM article_urls WHERE url = ?", (url,)
                ).fetchone()
                if row is None:
                    cursor = connection.execute(
                        """INSERT INTO articles (status, relevance, processed_at)
                           VALUES ('processed', ?, CURRENT_TIMESTAMP)""",
                        (relevance,),
                    )
                    connection.execute(
                        "INSERT INTO article_urls (url, article_id) VALUES (?, ?)",
                        (url, cursor.lastrowid),
                    )
                else:
                    connection.execute(
                        """UPDATE articles SET status = 'processed', relevance = ?,
                           landing_path = NULL, processed_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (relevance, row[0]),
                    )
