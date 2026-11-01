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
        with self._connection(foreign_keys=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'articles'"
            ).fetchone()
            if existing is None:
                self._create_articles_table(connection, "articles")
            else:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(articles)")
                }
                if "failure_reason" not in columns and "failed_at" not in columns:
                    self._create_articles_table(connection, "articles_migrated")
                    connection.execute(
                        """INSERT INTO articles_migrated
                           (id, status, relevance, landing_path, collected_at, processed_at)
                           SELECT id, status, relevance, landing_path, collected_at, processed_at
                           FROM articles"""
                    )
                    connection.execute("DROP TABLE articles")
                    connection.execute("ALTER TABLE articles_migrated RENAME TO articles")
                elif not {"failure_reason", "failed_at"}.issubset(columns):
                    raise HistoryError("Article history has an incomplete failure schema.")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS article_urls (
                    url TEXT PRIMARY KEY,
                    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE
                )"""
            )
            if connection.execute("PRAGMA foreign_key_check").fetchone():
                raise HistoryError("Article history contains invalid URL references.")

    @staticmethod
    def _create_articles_table(connection: sqlite3.Connection, name: str) -> None:
        if name not in {"articles", "articles_migrated"}:
            raise ValueError("Invalid article table name.")
        connection.execute(
            f"""CREATE TABLE {name} (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('collected', 'missing', 'processed', 'failed')),
                relevance INTEGER CHECK (relevance IN (0, 1) OR relevance IS NULL),
                landing_path TEXT,
                collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                processed_at TEXT,
                failed_at TEXT,
                failure_reason TEXT
            )"""
        )

    @contextmanager
    def _connection(self, *, foreign_keys: bool = True) -> Generator[sqlite3.Connection, None, None]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as connection:
                with connection:
                    connection.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")
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
        self.record_outcomes(articles, [])

    def url_for_landing_file(self, path: Path) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT article_urls.url FROM articles JOIN article_urls
                   ON articles.id = article_urls.article_id
                   WHERE articles.status = 'collected' AND articles.landing_path = ?
                   ORDER BY article_urls.url LIMIT 1""",
                (str(path.resolve()),),
            ).fetchone()
            return row[0] if row else None

    def record_outcomes(
        self,
        processed: list[tuple[str, bool | None]],
        failed: list[tuple[str, bool | None, str]],
    ) -> None:
        with self._connection() as connection:
            for url, relevance, reason in failed:
                row = connection.execute(
                    "SELECT article_id FROM article_urls WHERE url = ?", (url,)
                ).fetchone()
                if row is None:
                    cursor = connection.execute(
                        """INSERT INTO articles (status, relevance, failed_at, failure_reason)
                           VALUES ('failed', ?, CURRENT_TIMESTAMP, ?)""",
                        (relevance, reason),
                    )
                    connection.execute(
                        "INSERT INTO article_urls (url, article_id) VALUES (?, ?)",
                        (url, cursor.lastrowid),
                    )
                else:
                    connection.execute(
                        """UPDATE articles SET status = 'failed', relevance = ?,
                           landing_path = NULL, processed_at = NULL, failed_at = CURRENT_TIMESTAMP,
                           failure_reason = ? WHERE id = ?""",
                        (relevance, reason, row[0]),
                    )
            for url, relevance in processed:
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
                           landing_path = NULL, processed_at = CURRENT_TIMESTAMP,
                           failed_at = NULL, failure_reason = NULL WHERE id = ?""",
                        (relevance, row[0]),
                    )
