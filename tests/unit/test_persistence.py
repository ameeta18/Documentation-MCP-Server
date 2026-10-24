from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.persistence import IngestionMetadata, PersistenceError, SQLiteStore


def _metadata(dimension: int = 3) -> IngestionMetadata:
    return IngestionMetadata(
        source_repository="https://github.com/BerriAI/litellm-docs.git",
        source_ref="main",
        source_commit="abc123",
        embedding_model="test-embedding-model",
        embedding_dimension=dimension,
        chunk_size=700,
        chunk_overlap=100,
    )


def _document() -> Document:
    return Document(
        doc_id="doc_test",
        path="docs/guide.md",
        title="Guide",
        markdown="---\ntitle: Guide\n---\n# Guide\nOriginal text.\n",
    )


def _section() -> DocumentSection:
    return DocumentSection(
        section_id="doc_test:0000",
        doc_id="doc_test",
        ordinal=0,
        path="docs/guide.md",
        heading_path=("Guide", "Setup"),
        markdown="## Setup\nSection text.\n",
        embedding_text="Guide > Setup\n\nSetup Section text.",
    )


def test_database_round_trip_preserves_documents_sections_and_vectors(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "docs.sqlite3"
    metadata = _metadata()
    document = _document()
    section = _section()

    with SQLiteStore.create(database_path) as store:
        with store.transaction():
            store.save_ingestion(metadata)
            store.save_document(document)
            store.save_section(section, [0.1, 0.2, 0.3])

        loaded_sections = store.load_sections()
        assert store.get_ingestion() == metadata
        assert store.get_document(document.doc_id) == document

    assert database_path.is_file()
    assert len(loaded_sections) == 1
    assert loaded_sections[0].section == section
    assert loaded_sections[0].document_title == document.title
    np.testing.assert_allclose(
        loaded_sections[0].embedding,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_transaction_rolls_back_all_writes_after_failure(tmp_path: Path) -> None:
    with SQLiteStore.create(tmp_path / "docs.sqlite3") as store:
        with pytest.raises(RuntimeError, match="stop ingestion"), store.transaction():
            store.save_ingestion(_metadata())
            store.save_document(_document())
            raise RuntimeError("stop ingestion")

        assert store.get_ingestion() is None
        assert store.get_document("doc_test") is None


def test_section_embedding_must_match_ingestion_dimension(tmp_path: Path) -> None:
    with SQLiteStore.create(tmp_path / "docs.sqlite3") as store, store.transaction():
        store.save_ingestion(_metadata(dimension=3))
        store.save_document(_document())
        with pytest.raises(PersistenceError, match="does not match"):
            store.save_section(_section(), [0.1, 0.2])


def test_read_only_store_allows_reads_but_rejects_writes(tmp_path: Path) -> None:
    database_path = tmp_path / "docs.sqlite3"
    document = _document()
    with SQLiteStore.create(database_path) as store, store.transaction():
        store.save_ingestion(_metadata())
        store.save_document(document)

    with SQLiteStore.open_read_only(database_path) as store:
        assert store.get_document(document.doc_id) == document
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.save_document(
                Document(
                    doc_id="another",
                    path="docs/another.md",
                    title="Another",
                    markdown="# Another\n",
                )
            )
