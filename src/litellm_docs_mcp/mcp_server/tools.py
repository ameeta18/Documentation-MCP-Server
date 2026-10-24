"""Python handlers for the two required documentation tools."""

from __future__ import annotations

from litellm_docs_mcp.mcp_server.runtime import ServerRuntime
from litellm_docs_mcp.retrieval.search import SourceReference, build_source_reference


class ToolInputError(ValueError):
    """Raised when a tool receives an invalid argument."""


class DocumentNotFoundError(LookupError):
    """Raised when get_doc cannot find the requested document."""


class DocumentationTools:
    """The application handlers that will be registered with MCP."""

    def __init__(self, runtime: ServerRuntime) -> None:
        self._runtime = runtime

    def search_docs(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return reranked documentation sections with source references."""

        results = self._runtime.search_service.search(query, top_k)
        return [
            {
                "section_id": result.section_id,
                "doc_id": result.doc_id,
                "path": result.path,
                "document_title": result.document_title,
                "heading_path": list(result.heading_path),
                "excerpt": result.excerpt,
                "semantic_score": result.semantic_score,
                "score": result.final_score,
                "source": _source_dict(result.source),
            }
            for result in results
        ]

    def get_doc(self, doc_id: str) -> dict[str, object]:
        """Return one complete original Markdown document by its ingestion ID."""

        if not doc_id.strip():
            raise ToolInputError("doc_id must not be empty")

        document = self._runtime.store.get_document(doc_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        source = build_source_reference(self._runtime.metadata, document.path)
        return {
            "doc_id": document.doc_id,
            "path": document.path,
            "title": document.title,
            "markdown": document.markdown,
            "source": _source_dict(source),
        }


def _source_dict(source: SourceReference) -> dict[str, str | None]:
    return {
        "repository": source.repository,
        "commit": source.commit,
        "path": source.path,
        "url": source.url,
    }
