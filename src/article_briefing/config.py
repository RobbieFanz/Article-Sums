from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when configuration is missing or invalid."""


DEFAULT_RELEVANCE_CRITERIA = (
    "Include substantive world news, AI research and development, significant "
    "software engineering news, significant technology, significant cybersecurity, "
    "and US/global finance. Reject food, travel, lifestyle, entertainment, generic "
    "personal stories, and low-information social posts unless they have a direct "
    "technical, business, finance, or world-news connection."
)


@dataclass(frozen=True)
class SmtpSettings:
    enabled: bool
    host: str | None = None
    port: int = 587
    username: str | None = None
    password: str | None = None
    from_address: str | None = None
    to_addresses: tuple[str, ...] = ()
    use_starttls: bool = True
    use_ssl: bool = False
    timeout_seconds: float = 30

    @classmethod
    def from_environment(cls) -> "SmtpSettings":
        enabled = boolean_environment("SMTP_ENABLED", False)
        if not enabled:
            return cls(enabled=False)

        recipients = tuple(
            address.strip()
            for address in required_environment("SMTP_TO").split(",")
            if address.strip()
        )
        if not recipients:
            raise ConfigurationError("SMTP_TO must include at least one recipient address.")

        use_starttls = boolean_environment("SMTP_USE_STARTTLS", True)
        use_ssl = boolean_environment("SMTP_USE_SSL", False)
        if use_starttls and use_ssl:
            raise ConfigurationError(
                "SMTP_USE_STARTTLS and SMTP_USE_SSL cannot both be true."
            )

        return cls(
            enabled=True,
            host=required_environment("SMTP_HOST"),
            port=positive_integer("SMTP_PORT", 587),
            username=required_environment("SMTP_USERNAME"),
            password=required_environment("SMTP_PASSWORD"),
            from_address=required_environment("SMTP_FROM"),
            to_addresses=recipients,
            use_starttls=use_starttls,
            use_ssl=use_ssl,
            timeout_seconds=positive_float("SMTP_TIMEOUT_SECONDS", 30),
        )


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
    relevance_criteria: str = DEFAULT_RELEVANCE_CRITERIA
    relevance_max_article_characters: int = 6000
    smtp: SmtpSettings = field(default_factory=lambda: SmtpSettings(enabled=False))

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        load_dotenv(project_root / ".env")

        model = required_environment("OLLAMA_MODEL")
        timeout = positive_float("OLLAMA_TIMEOUT_SECONDS", 180)
        maximum_characters = positive_integer("MAX_ARTICLE_CHARACTERS", 24000)
        relevance_maximum_characters = positive_integer(
            "RELEVANCE_MAX_ARTICLE_CHARACTERS", 6000
        )
        relevance_criteria = load_relevance_criteria(
            resolve_project_path(
                project_root,
                os.getenv("RELEVANCE_CRITERIA_FILE", "config/relevance_criteria.txt"),
            )
        )

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
            relevance_criteria=relevance_criteria,
            relevance_max_article_characters=relevance_maximum_characters,
            smtp=SmtpSettings.from_environment(),
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


def boolean_environment(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, or off."
    )


def load_relevance_criteria(path: Path) -> str:
    try:
        criteria = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ConfigurationError(
            f"Could not read relevance criteria file {path}: {error}"
        ) from error
    if not criteria:
        raise ConfigurationError(f"Relevance criteria file {path} is empty.")
    return criteria


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path
