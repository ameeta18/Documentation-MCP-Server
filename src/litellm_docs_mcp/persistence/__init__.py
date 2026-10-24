"""SQLite persistence for documents, sections, and embeddings."""

from litellm_docs_mcp.persistence.sqlite_store import (
    IngestionMetadata,
    PersistenceError,
    SQLiteStore,
    StoredSection,
)

__all__ = ["IngestionMetadata", "PersistenceError", "SQLiteStore", "StoredSection"]
