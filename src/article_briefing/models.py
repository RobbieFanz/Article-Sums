from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Article:
    source_file: str
    title: str
    url: str | None
    text: str
    was_truncated: bool


@dataclass(frozen=True)
class ArticleSummary:
    title: str
    url: str | None
    source_file: str
    summary: str
    key_points: list[str]
    why_it_matters: str
    read_recommendation: str
    input_note: str
    summarization_duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BriefingOverview:
    overview: str
    major_themes: list[str]
    watch_for: str
    summarization_duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Briefing:
    created_at: str
    model: str
    overview: BriefingOverview
    articles: list[ArticleSummary]
    skipped_files: list[str]
    generation_duration_seconds: float
    total_duration_seconds: float | None = None
    email_delivery_duration_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
