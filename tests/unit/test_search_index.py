from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore
from litellm_docs_mcp.retrieval import SearchIndexError, load_search_index


def _metadata() -> IngestionMetadata:
    return IngestionMetadata(
        source_repository="https://github.com/BerriAI/litellm-docs.git",
        source_ref="main",
        source_commit="abc123",
        embedding_model="test-embedding-model",
        embedding_dimension=3,
        chunk_size=700,
        chunk_overlap=100,
    )


def _write_search_database(database_path: Path, *, include_section: bool = True) -> None:
    document = Document(
        doc_id="doc_guide",
        path="docs/guide.md",
        title="Configuration Guide",
        markdown="# Configuration Guide\n\n## Setup\nUse this setting.\n",
    )
    section = DocumentSection(
        section_id="doc_guide:0000",
        doc_id=document.doc_id,
        ordinal=0,
        path=document.path,
        heading_path=("Configuration Guide", "Setup"),
        markdown="## Setup\nUse this setting.\n",
        embedding_text="Configuration Guide > Setup\n\nUse this setting.",
    )

    with SQLiteStore.create(database_path) as store, store.transaction():
        store.save_ingestion(_metadata())
        store.save_document(document)
        if include_section:
            store.save_section(section, [0.1, 0.2, 0.3])


def test_load_search_index_keeps_metadata_sections_and_matrix_aligned(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "docs.sqlite3"
    _write_search_database(database_path)

    index = load_search_index(database_path)

    assert index.metadata == _metadata()
    assert len(index.sections) == 1
    assert index.sections[0].document_title == "Configuration Guide"
    assert index.sections[0].section.heading_path == ("Configuration Guide", "Setup")
    assert index.embeddings.shape == (1, 3)
    assert index.embeddings.dtype == np.float32
    np.testing.assert_allclose(index.embeddings[0], [0.1, 0.2, 0.3])


def test_load_search_index_rejects_dimension_mismatch(tmp_path: Path) -> None:
    database_path = tmp_path / "docs.sqlite3"
    _write_search_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE ingestions SET embedding_dimension = 2 WHERE id = 1")

    with pytest.raises(SearchIndexError, match="dimensions"):
        load_search_index(database_path)
