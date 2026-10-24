"""Embedding generation interfaces."""

from litellm_docs_mcp.embeddings.openai_client import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
    OpenAIEmbeddingClient,
)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingResult",
    "OpenAIEmbeddingClient",
]
