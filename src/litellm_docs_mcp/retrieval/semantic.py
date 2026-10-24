"""Load stored documentation embeddings into an in-memory search index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from litellm_docs_mcp.embeddings import EmbeddingProvider
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore, StoredSection


class SearchIndexError(RuntimeError):
    """Raised when a stored documentation database cannot be searched safely."""


class SemanticSearchError(ValueError):
    """Raised when a semantic search request or query embedding is invalid."""


@dataclass(frozen=True, slots=True)
class SearchIndex:
    """Stored sections and their embedding vectors in matching row order."""

    metadata: IngestionMetadata
    sections: tuple[StoredSection, ...]
    embeddings: NDArray[np.float32] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """A stored section and its cosine-similarity score."""

    stored_section: StoredSection
    semantic_score: float


class SemanticRetriever:
    """Retrieve the sections closest to an embedded query."""

    def __init__(self, index: SearchIndex, embedding_provider: EmbeddingProvider) -> None:
        norms = np.linalg.norm(index.embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise SearchIndexError("Stored embeddings must have non-zero magnitude")

        self._index = index
        self._embedding_provider = embedding_provider
        self._normalized_embeddings = index.embeddings / norms

    def search(self, query: str, candidate_k: int) -> tuple[SemanticCandidate, ...]:
        """Embed a query and return the strongest semantic candidates."""

        if not query.strip():
            raise SemanticSearchError("Search query must not be empty")
        if candidate_k <= 0:
            raise SemanticSearchError("candidate_k must be greater than zero")

        result = self._embedding_provider.embed_texts([query])
        if len(result.vectors) != 1:
            raise SemanticSearchError("Expected exactly one query embedding")

        query_vector = np.asarray(result.vectors[0], dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.size != self._index.metadata.embedding_dimension:
            raise SemanticSearchError(
                "Query embedding dimension does not match the stored embeddings"
            )
        if not np.isfinite(query_vector).all():
            raise SemanticSearchError("Query embedding must contain only finite values")

        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0:
            raise SemanticSearchError("Query embedding must have non-zero magnitude")

        scores = self._normalized_embeddings @ (query_vector / query_norm)
        ranked_indexes = sorted(
            range(len(self._index.sections)),
            key=lambda index: (
                -float(scores[index]),
                self._index.sections[index].section.path,
                self._index.sections[index].section.ordinal,
                self._index.sections[index].section.section_id,
            ),
        )
        return tuple(
            SemanticCandidate(
                stored_section=self._index.sections[index],
                semantic_score=float(scores[index]),
            )
            for index in ranked_indexes[:candidate_k]
        )


def load_search_index(database_path: Path) -> SearchIndex:
    """Load and validate one completed ingestion for semantic search."""

    with SQLiteStore.open_read_only(database_path) as store:
        metadata = store.get_ingestion()
        sections = tuple(store.load_sections())

    if metadata is None:
        raise SearchIndexError("Database does not contain ingestion metadata")
    if not sections:
        raise SearchIndexError("Database does not contain searchable sections")

    dimensions = {int(stored.embedding.size) for stored in sections}
    if dimensions != {metadata.embedding_dimension}:
        raise SearchIndexError("Stored embedding dimensions do not match the ingestion metadata")

    embeddings = np.stack([stored.embedding for stored in sections]).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(embeddings).all():
        raise SearchIndexError("Stored embeddings must contain only finite values")

    return SearchIndex(
        metadata=metadata,
        sections=sections,
        embeddings=embeddings,
    )
