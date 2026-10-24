from __future__ import annotations

import numpy as np

from litellm_docs_mcp.ingestion.models import DocumentSection
from litellm_docs_mcp.persistence import StoredSection
from litellm_docs_mcp.reranking import rerank_candidates
from litellm_docs_mcp.retrieval import SemanticCandidate


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


def test_semantic_similarity_remains_the_main_signal() -> None:
    stronger_semantic = _candidate("semantic", 0.9)
    stronger_lexical = _candidate(
        "lexical",
        -0.5,
        title="Configure retries",
        headings=("Configure retries",),
        markdown="Configure retries for requests.",
    )

    results = rerank_candidates("configure retries", [stronger_lexical, stronger_semantic])

    assert results[0].semantic_candidate == stronger_semantic


def test_content_term_coverage_can_break_equal_semantic_scores() -> None:
    no_match = _candidate("no-match", 0.5)
    content_match = _candidate(
        "content-match",
        0.5,
        markdown="Configure retry behavior for failed requests.",
    )

    results = rerank_candidates(
        "how to configure configure retry",
        [no_match, content_match],
    )

    assert results[0].semantic_candidate == content_match
    assert results[0].content_term_coverage == 1.0
    assert results[1].content_term_coverage == 0.0


def test_title_and_heading_matches_receive_context_boost() -> None:
    no_match = _candidate("no-match", 0.5)
    context_match = _candidate(
        "context-match",
        0.5,
        title="Proxy Configuration",
        headings=("Retry Policy",),
    )

    results = rerank_candidates("proxy retry", [no_match, context_match])

    assert results[0].semantic_candidate == context_match
    assert results[0].context_term_coverage == 1.0


def test_normalized_exact_phrase_receives_small_boost() -> None:
    separate_terms = _candidate(
        "separate",
        0.5,
        markdown="Retry other requests and configure settings.",
    )
    phrase_match = _candidate(
        "phrase",
        0.5,
        markdown="Use CONFIGURE-retry when handling requests.",
    )

    results = rerank_candidates("configure retry", [separate_terms, phrase_match])

    assert results[0].semantic_candidate == phrase_match
    assert results[0].exact_phrase_match is True
    assert results[1].exact_phrase_match is False


def test_equal_scores_use_stable_path_order() -> None:
    second = _candidate("second", 0.5, path="docs/b.md")
    first = _candidate("first", 0.5, path="docs/a.md")

    results = rerank_candidates("unmatched query", [second, first])

    assert [result.semantic_candidate for result in results] == [first, second]
