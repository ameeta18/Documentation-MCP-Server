from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from litellm_docs_mcp.config import IngestionConfig
from litellm_docs_mcp.embeddings import EmbeddingError, EmbeddingResult
from litellm_docs_mcp.ingestion.pipeline import run_ingestion
from litellm_docs_mcp.persistence import SQLiteStore


@dataclass
class _FakeEmbeddingProvider:
    dimension: int = 3
    prompt_tokens_per_text: int = 2

    def __post_init__(self) -> None:
        self.inputs: list[str] = []

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.inputs = list(texts)
        vectors = tuple(
            tuple(float(index + offset) for offset in range(self.dimension))
            for index, _ in enumerate(texts)
        )
        return EmbeddingResult(
            vectors=vectors,
            prompt_tokens=len(texts) * self.prompt_tokens_per_text,
        )


class _FailingEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        raise EmbeddingError("simulated embedding failure")


def _source_tree(root: Path) -> dict[str, str]:
    documents = {
        "docs/index.md": "---\ntitle: Home\n---\n# Welcome\nOriginal home text.\n",
        "docs/guides/setup.md": "# Setup\nInstall the package.\n",
    }
    for relative_path, content in documents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "docs" / "ignored.txt").write_text("Not Markdown", encoding="utf-8")
    return documents


def _config(source_dir: Path, database_path: Path) -> IngestionConfig:
    return IngestionConfig(
        openai_api_key="test-key",
        source_dir=source_dir,
        database_path=database_path,
        chunk_size=100,
        chunk_overlap=10,
    )


def test_pipeline_builds_complete_database_with_fake_embeddings(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_documents = _source_tree(source_dir)
    database_path = tmp_path / "output" / "docs.sqlite3"
    provider = _FakeEmbeddingProvider()

    summary = run_ingestion(_config(source_dir, database_path), provider)

    assert summary.database_path == database_path
    assert summary.source_commit is None
    assert summary.document_count == 2
    assert summary.section_count == 2
    assert summary.embedding_dimension == 3
    assert summary.prompt_tokens == 4
    assert len(provider.inputs) == 2

    with SQLiteStore.open(database_path) as store:
        metadata = store.get_ingestion()
        stored_sections = store.load_sections()
        stored_documents = {
            section.section.path: store.get_document(section.section.doc_id)
            for section in stored_sections
        }

    assert metadata is not None
    assert metadata.embedding_dimension == 3
    assert len(stored_sections) == 2
    assert stored_documents["docs/index.md"].markdown == source_documents["docs/index.md"]
    assert (
        stored_documents["docs/guides/setup.md"].markdown
        == source_documents["docs/guides/setup.md"]
    )


def test_failed_ingestion_does_not_replace_existing_database(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _source_tree(source_dir)
    database_path = tmp_path / "docs.sqlite3"
    config = _config(source_dir, database_path)
    run_ingestion(config, _FakeEmbeddingProvider())
    valid_database = database_path.read_bytes()

    with pytest.raises(EmbeddingError, match="simulated embedding failure"):
        run_ingestion(config, _FailingEmbeddingProvider())

    assert database_path.read_bytes() == valid_database
    with SQLiteStore.open(database_path) as store:
        assert store.get_ingestion() is not None
        assert len(store.load_sections()) == 2
