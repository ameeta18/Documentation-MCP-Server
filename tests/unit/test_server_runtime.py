from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import litellm_docs_mcp.mcp_server.runtime as runtime_module
from litellm_docs_mcp.config import ServerConfig
from litellm_docs_mcp.embeddings import EmbeddingResult
from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.mcp_server import create_server_runtime
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore


@dataclass
class _FakeEmbeddingProvider:
    vector: tuple[float, ...] = (1.0, 0.0)
    inputs: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.inputs.append(list(texts))
        return EmbeddingResult(vectors=(self.vector,), prompt_tokens=1)


def _write_database(database_path: Path) -> Document:
    document = Document(
        doc_id="doc_setup",
        path="docs/setup.md",
        title="Setup Guide",
        markdown="# Setup Guide\n\nInstall the package.\n",
    )
    section = DocumentSection(
        section_id="doc_setup:0000",
        doc_id=document.doc_id,
        ordinal=0,
        path=document.path,
        heading_path=("Setup Guide",),
        markdown=document.markdown,
        embedding_text="Setup Guide Install the package.",
    )
    metadata = IngestionMetadata(
        source_repository="https://github.com/BerriAI/litellm-docs.git",
        source_ref="main",
        source_commit="abc123",
        embedding_model="stored-model",
        embedding_dimension=2,
        chunk_size=700,
        chunk_overlap=100,
    )
    with SQLiteStore.create(database_path) as store, store.transaction():
        store.save_ingestion(metadata)
        store.save_document(document)
        store.save_section(section, [1.0, 0.0])
    return document


def test_runtime_configures_query_embeddings_from_stored_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "docs.sqlite3"
    _write_database(database_path)
    provider = _FakeEmbeddingProvider()
    captured_arguments: dict[str, object] = {}

    def fake_openai_client(**arguments: object) -> _FakeEmbeddingProvider:
        captured_arguments.update(arguments)
        return provider

    monkeypatch.setattr(runtime_module, "OpenAIEmbeddingClient", fake_openai_client)
    config = ServerConfig(openai_api_key="secret-test-value", database_path=database_path)

    with create_server_runtime(config):
        pass

    assert captured_arguments == {
        "api_key": "secret-test-value",
        "model": "stored-model",
        "batch_size": 1,
    }
