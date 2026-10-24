"""Application-level documentation search orchestration."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol
from urllib.parse import quote, urlsplit

from litellm_docs_mcp.persistence import IngestionMetadata
from litellm_docs_mcp.reranking import rerank_candidates
from litellm_docs_mcp.retrieval.semantic import SemanticCandidate

MAX_TOP_K = 20
MIN_CANDIDATE_COUNT = 25
CANDIDATE_MULTIPLIER = 5
MAX_EXCERPT_CHARACTERS = 500


class SearchServiceError(ValueError):
    """Raised when a documentation search request is invalid."""


class CandidateRetriever(Protocol):
    """Interface required by the search orchestration layer."""

    def search(self, query: str, candidate_k: int) -> Sequence[SemanticCandidate]: ...


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Stored provenance and an optional commit-pinned GitHub URL."""

    repository: str
    commit: str | None
    path: str
    url: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A reranked documentation section ready for result formatting."""

    section_id: str
    doc_id: str
    path: str
    document_title: str
    heading_path: tuple[str, ...]
    excerpt: str
    semantic_score: float
    final_score: float
    source: SourceReference


class SearchService:
    """Run semantic retrieval followed by the custom reranker."""

    def __init__(self, retriever: CandidateRetriever, metadata: IngestionMetadata) -> None:
        self._retriever = retriever
        self._metadata = metadata

    def search(self, query: str, top_k: int) -> tuple[SearchResult, ...]:
        """Return at most top_k reranked documentation sections."""

        if not query.strip():
            raise SearchServiceError("Search query must not be empty")
        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise SearchServiceError(f"top_k must be an integer from 1 to {MAX_TOP_K}")

        candidate_k = max(MIN_CANDIDATE_COUNT, top_k * CANDIDATE_MULTIPLIER)
        semantic_candidates = self._retriever.search(query, candidate_k)
        reranked = rerank_candidates(query, semantic_candidates)

        results: list[SearchResult] = []
        for candidate in reranked[:top_k]:
            semantic = candidate.semantic_candidate
            stored = semantic.stored_section
            section = stored.section
            results.append(
                SearchResult(
                    section_id=section.section_id,
                    doc_id=section.doc_id,
                    path=section.path,
                    document_title=stored.document_title,
                    heading_path=section.heading_path,
                    excerpt=_excerpt(section.markdown),
                    semantic_score=semantic.semantic_score,
                    final_score=candidate.final_score,
                    source=build_source_reference(self._metadata, section.path),
                )
            )
        return tuple(results)


def build_source_reference(metadata: IngestionMetadata, path: str) -> SourceReference:
    """Create source provenance and a pinned URL when possible."""

    normalized_path = PurePosixPath(path.replace("\\", "/"))
    if normalized_path.is_absolute() or ".." in normalized_path.parts:
        raise SearchServiceError("Source path must be repository-relative")
    path_text = normalized_path.as_posix()

    repository_path = _github_repository_path(metadata.source_repository)
    url = None
    if metadata.source_commit and repository_path:
        url = (
            f"https://github.com/{quote(repository_path, safe='/')}"
            f"/blob/{quote(metadata.source_commit, safe='')}"
            f"/{quote(path_text, safe='/')}"
        )

    return SourceReference(
        repository=metadata.source_repository,
        commit=metadata.source_commit,
        path=path_text,
        url=url,
    )


def _github_repository_path(repository_url: str) -> str | None:
    repository = repository_url.strip().rstrip("/")
    if repository.startswith("git@github.com:"):
        repository_path = repository.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(repository)
        if parsed.hostname is None or parsed.hostname.casefold() != "github.com":
            return None
        repository_path = parsed.path.lstrip("/")

    repository_path = repository_path.removesuffix(".git").strip("/")
    return repository_path or None


def _excerpt(markdown: str) -> str:
    compact_text = " ".join(markdown.split())
    return textwrap.shorten(
        compact_text,
        width=MAX_EXCERPT_CHARACTERS,
        placeholder="…",
    )
