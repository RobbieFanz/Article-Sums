from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
class RelevanceDecision:
    include: bool
    reason: str
    duration_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RelevanceFailure:
    summary: ArticleSummary
    invalid_outputs: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArticleNotSelected:
    source_file: str
    title: str
    url: str | None
    reason: str


@dataclass(frozen=True)
class FailureReport:
    error: str
    duration_seconds: float
    processed_article_count: int
    current_article: str | None


@dataclass(frozen=True)
class BriefingOverview:
    overview: str
    major_themes: list[str]
    watch_for: str
    summarization_duration_seconds: float = 0.0
    error: str | None = None

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
    relevance_failures: list[RelevanceFailure] = field(default_factory=list)
    articles_not_selected: list[ArticleNotSelected] = field(default_factory=list)
    included_article_count: int = 0
    not_included_article_count: int = 0
    total_duration_seconds: float | None = None
    email_delivery_duration_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
