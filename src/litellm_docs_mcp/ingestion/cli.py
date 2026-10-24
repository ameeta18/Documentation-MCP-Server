"""Command-line entry point for documentation ingestion."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from litellm_docs_mcp.config import ConfigurationError, IngestionConfig
from litellm_docs_mcp.embeddings import EmbeddingError
from litellm_docs_mcp.ingestion.pipeline import IngestionError, IngestionSummary, run_ingestion
from litellm_docs_mcp.ingestion.source import SourceError
from litellm_docs_mcp.persistence import PersistenceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest LiteLLM Markdown documentation into a local SQLite database."
    )
    parser.add_argument(
        "--repo-url",
        help="Git repository URL; defaults to DOCS_MCP_REPO_URL.",
    )
    parser.add_argument(
        "--ref",
        dest="source_ref",
        help="Git branch or tag; defaults to DOCS_MCP_SOURCE_REF.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Local repository root containing docs/; skips Git acquisition.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Output SQLite path; defaults to DOCS_MCP_DATABASE_PATH.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ingestion and return a process exit code."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        config = _config_with_overrides(arguments)
        summary = run_ingestion(config)
    except (
        ConfigurationError,
        EmbeddingError,
        IngestionError,
        OSError,
        PersistenceError,
        SourceError,
        sqlite3.Error,
    ) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


def _config_with_overrides(arguments: argparse.Namespace) -> IngestionConfig:
    config = IngestionConfig.from_env()
    updates: dict[str, object] = {}
    if arguments.repo_url is not None:
        updates["repository_url"] = arguments.repo_url
    if arguments.source_ref is not None:
        updates["source_ref"] = arguments.source_ref
    if arguments.source_dir is not None:
        updates["source_dir"] = arguments.source_dir
    if arguments.database is not None:
        updates["database_path"] = arguments.database
    return replace(config, **updates)


def _print_summary(summary: IngestionSummary) -> None:
    source_commit = summary.source_commit or "local source without Git commit"
    print("Ingestion completed successfully.")
    print(f"Database: {summary.database_path}")
    print(f"Source commit: {source_commit}")
    print(f"Documents: {summary.document_count}")
    print(f"Sections: {summary.section_count}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Embedding prompt tokens: {summary.prompt_tokens}")


if __name__ == "__main__":
    raise SystemExit(main())
