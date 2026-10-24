"""Small, explainable scoring function for documentation search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from litellm_docs_mcp.retrieval.semantic import SemanticCandidate

SEMANTIC_WEIGHT = 0.70
CONTENT_TERM_WEIGHT = 0.15
CONTEXT_TERM_WEIGHT = 0.10
EXACT_PHRASE_WEIGHT = 0.05

_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "with",
    }
)


class RerankingError(ValueError):
    """Raised when candidates cannot be reranked for the supplied query."""


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """A semantic candidate with transparent reranking components."""

    semantic_candidate: SemanticCandidate
    final_score: float
    content_term_coverage: float
    context_term_coverage: float
    exact_phrase_match: bool


def rerank_candidates(
    query: str,
    candidates: Sequence[SemanticCandidate],
) -> tuple[RerankedCandidate, ...]:
    """Score and deterministically reorder semantic candidates."""

    if not query.strip():
        raise RerankingError("Search query must not be empty")

    query_terms = _meaningful_terms(query)
    normalized_query = _normalized_phrase(query)
    reranked: list[RerankedCandidate] = []

    for candidate in candidates:
        stored = candidate.stored_section
        section = stored.section
        context = " ".join([stored.document_title, *section.heading_path])
        searchable_text = f"{context}\n{section.markdown}"

        content_coverage = _term_coverage(query_terms, section.markdown)
        context_coverage = _term_coverage(query_terms, context)
        phrase_match = bool(normalized_query) and normalized_query in _normalized_phrase(
            searchable_text
        )
        normalized_semantic = _normalized_semantic_score(candidate.semantic_score)
        final_score = (
            SEMANTIC_WEIGHT * normalized_semantic
            + CONTENT_TERM_WEIGHT * content_coverage
            + CONTEXT_TERM_WEIGHT * context_coverage
            + EXACT_PHRASE_WEIGHT * float(phrase_match)
        )
        reranked.append(
            RerankedCandidate(
                semantic_candidate=candidate,
                final_score=final_score,
                content_term_coverage=content_coverage,
                context_term_coverage=context_coverage,
                exact_phrase_match=phrase_match,
            )
        )

    return tuple(
        sorted(
            reranked,
            key=lambda result: (
                -result.final_score,
                -result.semantic_candidate.semantic_score,
                result.semantic_candidate.stored_section.section.path,
                result.semantic_candidate.stored_section.section.ordinal,
                result.semantic_candidate.stored_section.section.section_id,
            ),
        )
    )


def _normalized_semantic_score(score: float) -> float:
    return min(1.0, max(0.0, (score + 1.0) / 2.0))


def _meaningful_terms(text: str) -> set[str]:
    return {term for term in _tokens(text) if term not in _STOP_WORDS}


def _term_coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    matched_terms = query_terms.intersection(_tokens(text))
    return len(matched_terms) / len(query_terms)


def _tokens(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.casefold()))


def _normalized_phrase(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.casefold()))
