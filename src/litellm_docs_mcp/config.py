"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPOSITORY_URL = "https://github.com/BerriAI/litellm-docs.git"
DEFAULT_SOURCE_REF = "main"
DEFAULT_DATABASE_PATH = Path("data/docs.sqlite3")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8000


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Configuration required by the standalone ingestion process."""

    openai_api_key: str = field(repr=False)
    database_path: Path = DEFAULT_DATABASE_PATH
    repository_url: str = DEFAULT_REPOSITORY_URL
    source_ref: str = DEFAULT_SOURCE_REF
    source_dir: Path | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if not self.openai_api_key.strip():
            raise ConfigurationError("OPENAI_API_KEY is required for ingestion")
        if not self.repository_url.strip():
            raise ConfigurationError("DOCS_MCP_REPO_URL must not be empty")
        if not self.source_ref.strip():
            raise ConfigurationError("DOCS_MCP_SOURCE_REF must not be empty")
        if not self.embedding_model.strip():
            raise ConfigurationError("DOCS_MCP_EMBEDDING_MODEL must not be empty")
        if self.embedding_batch_size <= 0:
            raise ConfigurationError("DOCS_MCP_EMBEDDING_BATCH_SIZE must be greater than zero")
        if self.chunk_size <= 0:
            raise ConfigurationError("DOCS_MCP_CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0:
            raise ConfigurationError("DOCS_MCP_CHUNK_OVERLAP must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "DOCS_MCP_CHUNK_OVERLAP must be smaller than DOCS_MCP_CHUNK_SIZE"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> IngestionConfig:
        """Create ingestion configuration from environment variables."""

        values = os.environ if env is None else env
        api_key = values.get("OPENAI_API_KEY", "")

        return cls(
            openai_api_key=api_key,
            database_path=Path(
                values.get("DOCS_MCP_DATABASE_PATH", str(DEFAULT_DATABASE_PATH))
            ).expanduser(),
            repository_url=values.get("DOCS_MCP_REPO_URL", DEFAULT_REPOSITORY_URL),
            source_ref=values.get("DOCS_MCP_SOURCE_REF", DEFAULT_SOURCE_REF),
            source_dir=_optional_path(values.get("DOCS_MCP_SOURCE_DIR")),
            embedding_model=values.get("DOCS_MCP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_batch_size=_positive_integer(
                values,
                "DOCS_MCP_EMBEDDING_BATCH_SIZE",
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
            chunk_size=_positive_integer(
                values,
                "DOCS_MCP_CHUNK_SIZE",
                DEFAULT_CHUNK_SIZE,
            ),
            chunk_overlap=_non_negative_integer(
                values,
                "DOCS_MCP_CHUNK_OVERLAP",
                DEFAULT_CHUNK_OVERLAP,
            ),
        )


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Configuration required by the local MCP server."""

    openai_api_key: str = field(repr=False)
    database_path: Path = DEFAULT_DATABASE_PATH
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT

    def __post_init__(self) -> None:
        if not self.openai_api_key.strip():
            raise ConfigurationError("OPENAI_API_KEY is required for search")
        if not self.host.strip():
            raise ConfigurationError("DOCS_MCP_HOST must not be empty")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("DOCS_MCP_PORT must be from 1 to 65535")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerConfig:
        """Create server configuration from environment variables."""

        values = os.environ if env is None else env
        return cls(
            openai_api_key=values.get("OPENAI_API_KEY", ""),
            database_path=Path(
                values.get("DOCS_MCP_DATABASE_PATH", str(DEFAULT_DATABASE_PATH))
            ).expanduser(),
            host=values.get("DOCS_MCP_HOST", DEFAULT_SERVER_HOST),
            port=_integer(values, "DOCS_MCP_PORT", DEFAULT_SERVER_PORT),
        )


def _optional_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def _positive_integer(values: Mapping[str, str], name: str, default: int) -> int:
    value = _integer(values, name, default)
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _non_negative_integer(values: Mapping[str, str], name: str, default: int) -> int:
    value = _integer(values, name, default)
    if value < 0:
        raise ConfigurationError(f"{name} must not be negative")
    return value


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw_value = values.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
