from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    ollama_model: str
    ollama_base_url: str
    ollama_timeout_seconds: float
    landing_dir: Path
    archive_dir: Path
    briefing_dir: Path
    max_article_characters: int

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        load_dotenv(project_root / ".env")

        model = required_environment("OLLAMA_MODEL")
        timeout = positive_float("OLLAMA_TIMEOUT_SECONDS", 180)
        maximum_characters = positive_integer("MAX_ARTICLE_CHARACTERS", 24000)

        return cls(
            project_root=project_root,
            ollama_model=model,
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_timeout_seconds=timeout,
            landing_dir=resolve_project_path(
                project_root, os.getenv("LANDING_DIR", "data/landing")
            ),
            archive_dir=resolve_project_path(
                project_root, os.getenv("ARCHIVE_DIR", "data/archive")
            ),
            briefing_dir=resolve_project_path(
                project_root, os.getenv("BRIEFING_DIR", "data/briefings")
            ),
            max_article_characters=maximum_characters,
        )


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"{path}:{line_number} must use KEY=VALUE format.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigurationError(f"{path}:{line_number} has an empty variable name.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"{name} is required. Copy .env.example to .env and set a local Ollama model."
        )
    return value


def positive_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a positive number.") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return parsed


def positive_integer(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a positive integer.") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return parsed


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path
