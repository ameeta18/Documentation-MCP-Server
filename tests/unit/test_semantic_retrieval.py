from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from litellm_docs_mcp.embeddings import EmbeddingResult
from litellm_docs_mcp.ingestion.models import DocumentSection
from litellm_docs_mcp.persistence import IngestionMetadata, StoredSection
from litellm_docs_mcp.retrieval import (
    SearchIndex,
    SemanticRetriever,
    SemanticSearchError,
)


@dataclass
class _FakeEmbeddingProvider:
    vector: tuple[float, ...]
    inputs: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.inputs.append(list(texts))
        return EmbeddingResult(vectors=(self.vector,), prompt_tokens=1)


def _stored_section(
    section_id: str,
    path: str,
    ordinal: int,
    embedding: tuple[float, ...],
) -> StoredSection:
    return StoredSection(
        section=DocumentSection(
            section_id=section_id,
            doc_id=f"doc_{section_id}",
            ordinal=ordinal,
            path=path,
            heading_path=("Guide",),
            markdown=f"Content for {section_id}",
            embedding_text=f"Guide Content for {section_id}",
        ),
        document_title="Guide",
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def _index(*sections: StoredSection) -> SearchIndex:
    dimension = int(sections[0].embedding.size)
    return SearchIndex(
        metadata=IngestionMetadata(
            source_repository="https://github.com/BerriAI/litellm-docs.git",
            source_ref="main",
            source_commit="abc123",
            embedding_model="test-model",
            embedding_dimension=dimension,
            chunk_size=700,
            chunk_overlap=100,
        ),
        sections=sections,
        embeddings=np.stack([section.embedding for section in sections]),
    )


def test_semantic_retrieval_embeds_query_and_returns_best_candidates() -> None:
    index = _index(
        _stored_section("first", "docs/first.md", 0, (10.0, 0.0)),
        _stored_section("second", "docs/second.md", 0, (1.0, 1.0)),
        _stored_section("third", "docs/third.md", 0, (0.0, 1.0)),
    )
    provider = _FakeEmbeddingProvider((1.0, 0.0))

    candidates = SemanticRetriever(index, provider).search("configuration", candidate_k=2)

    assert provider.inputs == [["configuration"]]
    assert [candidate.stored_section.section.section_id for candidate in candidates] == [
        "first",
        "second",
    ]
    assert candidates[0].semantic_score == pytest.approx(1.0)
    assert candidates[1].semantic_score == pytest.approx(2**-0.5)


def test_semantic_retrieval_uses_stable_metadata_order_for_equal_scores() -> None:
    index = _index(
        _stored_section("second", "docs/b.md", 0, (1.0, 0.0)),
        _stored_section("first", "docs/a.md", 0, (1.0, 0.0)),
    )

    candidates = SemanticRetriever(index, _FakeEmbeddingProvider((1.0, 0.0))).search(
        "same score",
        candidate_k=10,
    )

    assert [candidate.stored_section.section.section_id for candidate in candidates] == [
        "first",
        "second",
    ]


def test_semantic_retrieval_rejects_query_dimension_mismatch() -> None:
    index = _index(_stored_section("first", "docs/first.md", 0, (1.0, 0.0)))

    with pytest.raises(SemanticSearchError, match="dimension"):
        SemanticRetriever(index, _FakeEmbeddingProvider((1.0, 0.0, 0.0))).search("query", 1)
