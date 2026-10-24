"""Assemble the storage and search components used by the MCP server."""

from __future__ import annotations

from dataclasses import dataclass

from litellm_docs_mcp.config import ServerConfig
from litellm_docs_mcp.embeddings import EmbeddingProvider, OpenAIEmbeddingClient
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore
from litellm_docs_mcp.retrieval import SemanticRetriever, load_search_index
from litellm_docs_mcp.retrieval.search import SearchService


@dataclass(slots=True)
class ServerRuntime:
    """Long-lived services owned by one MCP server process."""

    config: ServerConfig
    metadata: IngestionMetadata
    store: SQLiteStore
    search_service: SearchService

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> ServerRuntime:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def create_server_runtime(
    config: ServerConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> ServerRuntime:
    """Load a completed database and connect the search components."""

    index = load_search_index(config.database_path)
    provider = embedding_provider or OpenAIEmbeddingClient(
        api_key=config.openai_api_key,
        model=index.metadata.embedding_model,
        batch_size=1,
    )
    retriever = SemanticRetriever(index, provider)
    search_service = SearchService(retriever, index.metadata)
    store = SQLiteStore.open_read_only(config.database_path)
    return ServerRuntime(
        config=config,
        metadata=index.metadata,
        store=store,
        search_service=search_service,
    )
