"""Orchestrate documentation ingestion from source to SQLite."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from litellm_docs_mcp.config import IngestionConfig
from litellm_docs_mcp.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    OpenAIEmbeddingClient,
)
from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.ingestion.source import acquire_source, discover_documents
from litellm_docs_mcp.markdown import section_document
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore


class IngestionError(RuntimeError):
    """Raised when an ingestion result is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Non-secret information produced by a completed ingestion."""

    database_path: Path
    source_commit: str | None
    document_count: int
    section_count: int
    embedding_dimension: int
    prompt_tokens: int


def run_ingestion(
    config: IngestionConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestionSummary:
    """Build and atomically publish a complete documentation database."""

    provider = embedding_provider or OpenAIEmbeddingClient(
        api_key=config.openai_api_key,
        model=config.embedding_model,
        batch_size=config.embedding_batch_size,
    )

    with acquire_source(
        repository_url=config.repository_url,
        source_ref=config.source_ref,
        source_dir=config.source_dir,
    ) as snapshot:
        discovered_documents = discover_documents(snapshot)
        if not discovered_documents:
            raise IngestionError("No Markdown documents were found under docs/")

        documents, sections = _section_documents(discovered_documents, config)
        embedding_result = provider.embed_texts([section.embedding_text for section in sections])
        _validate_embeddings(embedding_result, len(sections))

        metadata = IngestionMetadata(
            source_repository=config.repository_url,
            source_ref=config.source_ref,
            source_commit=snapshot.commit_sha,
            embedding_model=config.embedding_model,
            embedding_dimension=embedding_result.dimension,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        _build_database(
            database_path=config.database_path,
            metadata=metadata,
            documents=documents,
            sections=sections,
            embeddings=embedding_result.vectors,
        )

    return IngestionSummary(
        database_path=config.database_path,
        source_commit=snapshot.commit_sha,
        document_count=len(documents),
        section_count=len(sections),
        embedding_dimension=embedding_result.dimension,
        prompt_tokens=embedding_result.prompt_tokens,
    )


def _section_documents(
    documents: list[Document],
    config: IngestionConfig,
) -> tuple[list[Document], list[DocumentSection]]:
    processed_documents: list[Document] = []
    all_sections: list[DocumentSection] = []
    for document in documents:
        processed_document, sections = section_document(
            document,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        processed_documents.append(processed_document)
        all_sections.extend(sections)
    return processed_documents, all_sections


def _validate_embeddings(result: EmbeddingResult, expected_count: int) -> None:
    if len(result.vectors) != expected_count:
        raise IngestionError(
            f"Expected {expected_count} embeddings but received {len(result.vectors)}"
        )
    if result.dimension <= 0:
        raise IngestionError("Embedding vectors must have a positive dimension")


def _build_database(
    *,
    database_path: Path,
    metadata: IngestionMetadata,
    documents: list[Document],
    sections: list[DocumentSection],
    embeddings: tuple[tuple[float, ...], ...],
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{database_path.name}-",
        dir=database_path.parent,
    ) as staging_directory:
        staged_path = Path(staging_directory) / database_path.name
        with SQLiteStore.create(staged_path) as store:
            with store.transaction():
                store.save_ingestion(metadata)
                for document in documents:
                    store.save_document(document)
                for section, embedding in zip(sections, embeddings, strict=True):
                    store.save_section(section, embedding)
            _validate_database(store, metadata, documents, len(sections))
        os.replace(staged_path, database_path)


def _validate_database(
    store: SQLiteStore,
    metadata: IngestionMetadata,
    documents: list[Document],
    expected_section_count: int,
) -> None:
    if store.get_ingestion() != metadata:
        raise IngestionError("Stored ingestion metadata does not match the completed run")
    for document in documents:
        if store.get_document(document.doc_id) != document:
            raise IngestionError(f"Stored document does not match source: {document.path}")
    if len(store.load_sections()) != expected_section_count:
        raise IngestionError("Stored section count does not match the completed run")
