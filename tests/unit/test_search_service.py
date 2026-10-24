from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from litellm_docs_mcp.ingestion.models import DocumentSection
from litellm_docs_mcp.persistence import IngestionMetadata, StoredSection
from litellm_docs_mcp.retrieval import SemanticCandidate
from litellm_docs_mcp.retrieval.search import (
    SearchService,
    SearchServiceError,
)


@dataclass
class _FakeRetriever:
    candidates: tuple[SemanticCandidate, ...]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, candidate_k: int) -> tuple[SemanticCandidate, ...]:
        self.calls.append((query, candidate_k))
        return self.candidates[:candidate_k]


def _metadata(
    *,
    commit: str | None = "abc123",
    repository: str = "https://github.com/BerriAI/litellm-docs.git",
) -> IngestionMetadata:
    return IngestionMetadata(
        source_repository=repository,
        source_ref="main",
        source_commit=commit,
        embedding_model="test-model",
        embedding_dimension=2,
        chunk_size=700,
        chunk_overlap=100,
    )


def _candidate(
    section_id: str,
    semantic_score: float,
    *,
    title: str = "Guide",
    headings: tuple[str, ...] = ("Overview",),
    markdown: str = "General documentation.",
    path: str | None = None,
) -> SemanticCandidate:
    section = DocumentSection(
        section_id=section_id,
        doc_id=f"doc_{section_id}",
        ordinal=0,
        path=path or f"docs/{section_id}.md",
        heading_path=headings,
        markdown=markdown,
        embedding_text=markdown,
    )
    return SemanticCandidate(
        stored_section=StoredSection(
            section=section,
            document_title=title,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        semantic_score=semantic_score,
    )


def test_search_retrieves_extra_candidates_and_applies_reranker() -> None:
    semantic_first = _candidate("semantic", 0.60)
    lexical_first = _candidate(
        "lexical",
        0.55,
        title="Configure retries",
        headings=("Configure retries",),
        markdown="Configure retries for failed requests.",
    )
    retriever = _FakeRetriever((semantic_first, lexical_first))

    results = SearchService(retriever, _metadata()).search("configure retries", top_k=1)

    assert retriever.calls == [("configure retries", 25)]
    assert len(results) == 1
    assert results[0].section_id == "lexical"
    assert results[0].final_score > results[0].semantic_score


def test_search_result_contains_section_context_and_compact_excerpt() -> None:
    markdown = "## Setup\n\n" + "word " * 150
    candidate = _candidate(
        "setup",
        0.8,
        title="Configuration Guide",
        headings=("Configuration", "Setup"),
        markdown=markdown,
        path="docs/configuration.md",
    )

    result = SearchService(_FakeRetriever((candidate,)), _metadata()).search(
        "setup",
        top_k=5,
    )[0]

    assert result.doc_id == "doc_setup"
    assert result.path == "docs/configuration.md"
    assert result.document_title == "Configuration Guide"
    assert result.heading_path == ("Configuration", "Setup")
    assert "\n" not in result.excerpt
    assert len(result.excerpt) <= 500
    assert result.excerpt.endswith("…")
    assert result.semantic_score == 0.8
    assert result.source.path == "docs/configuration.md"
    assert result.source.url == (
        "https://github.com/BerriAI/litellm-docs/blob/abc123/docs/configuration.md"
    )


def test_search_rejects_invalid_top_k() -> None:
    service = SearchService(_FakeRetriever(()), _metadata())

    with pytest.raises(SearchServiceError, match="top_k"):
        service.search("query", 21)


def test_search_rejects_blank_query_before_retrieval() -> None:
    retriever = _FakeRetriever(())

    with pytest.raises(SearchServiceError, match="must not be empty"):
        SearchService(retriever, _metadata()).search("   ", top_k=5)

    assert retriever.calls == []
