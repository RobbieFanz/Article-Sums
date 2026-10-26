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

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BriefingOverview:
    overview: str
    major_themes: list[str]
    watch_for: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
