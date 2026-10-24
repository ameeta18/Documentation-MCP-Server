"""Small SQLite storage layer for the ingested documentation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from litellm_docs_mcp.ingestion.models import Document, DocumentSection


class PersistenceError(RuntimeError):
    """Raised when stored data is missing or invalid."""


@dataclass(frozen=True, slots=True)
class IngestionMetadata:
    """Information needed to reproduce and validate one ingestion."""

    source_repository: str
    source_ref: str
    source_commit: str | None
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True, slots=True)
class StoredSection:
    """A section loaded together with its embedding vector."""

    section: DocumentSection
    document_title: str
    embedding: NDArray[np.float32] = field(repr=False)


class SQLiteStore:
    """Read and write one completed documentation snapshot."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(cls, database_path: Path) -> SQLiteStore:
        """Create a database and initialize its schema."""

        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, isolation_level=None)
        store = cls(connection)
        schema = (
            resources.files("litellm_docs_mcp.persistence")
            .joinpath("schema.sql")
            .read_text(encoding="utf-8")
        )
        connection.executescript(schema)
        return store

    @classmethod
    def open(cls, database_path: Path) -> SQLiteStore:
        """Open an existing documentation database."""

        if not database_path.is_file():
            raise PersistenceError(f"Database does not exist: {database_path}")
        return cls(sqlite3.connect(database_path, isolation_level=None))

    @classmethod
    def open_read_only(cls, database_path: Path) -> SQLiteStore:
        """Open an existing documentation database without write access."""

        if not database_path.is_file():
            raise PersistenceError(f"Database does not exist: {database_path}")
        database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
        store = cls(
            sqlite3.connect(
                database_uri,
                uri=True,
                isolation_level=None,
                check_same_thread=False,
            )
        )
        store._connection.execute("PRAGMA query_only = ON")
        return store

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit all writes together or roll all of them back."""

        self._connection.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def save_ingestion(self, metadata: IngestionMetadata) -> None:
        self._connection.execute(
            """
            INSERT INTO ingestions (
                id,
                source_repository,
                source_ref,
                source_commit,
                embedding_model,
                embedding_dimension,
                chunk_size,
                chunk_overlap
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.source_repository,
                metadata.source_ref,
                metadata.source_commit,
                metadata.embedding_model,
                metadata.embedding_dimension,
                metadata.chunk_size,
                metadata.chunk_overlap,
            ),
        )

    def get_ingestion(self) -> IngestionMetadata | None:
        row = self._connection.execute(
            """
            SELECT
                source_repository,
                source_ref,
                source_commit,
                embedding_model,
                embedding_dimension,
                chunk_size,
                chunk_overlap
            FROM ingestions
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return None
        return IngestionMetadata(
            source_repository=row["source_repository"],
            source_ref=row["source_ref"],
            source_commit=row["source_commit"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
        )

    def save_document(self, document: Document) -> None:
        self._connection.execute(
            """
            INSERT INTO documents (
                doc_id,
                path,
                title,
                original_markdown,
                content_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.path,
                document.title,
                document.markdown,
                _sha256(document.markdown),
            ),
        )

    def get_document(self, doc_id: str) -> Document | None:
        row = self._connection.execute(
            """
            SELECT doc_id, path, title, original_markdown, content_sha256
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        if _sha256(row["original_markdown"]) != row["content_sha256"]:
            raise PersistenceError(f"Stored document failed integrity check: {doc_id}")
        return Document(
            doc_id=row["doc_id"],
            path=row["path"],
            title=row["title"],
            markdown=row["original_markdown"],
        )

    def save_section(self, section: DocumentSection, embedding: Sequence[float]) -> None:
        expected_dimension = self._embedding_dimension()
        vector = _vector_to_bytes(embedding)
        if vector.dimension != expected_dimension:
            raise PersistenceError(
                f"Embedding dimension {vector.dimension} does not match expected dimension "
                f"{expected_dimension}"
            )
        self._connection.execute(
            """
            INSERT INTO sections (
                section_id,
                doc_id,
                ordinal,
                heading_path,
                markdown,
                embedding_text,
                content_sha256,
                embedding,
                embedding_dimension
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                section.section_id,
                section.doc_id,
                section.ordinal,
                json.dumps(section.heading_path, ensure_ascii=False, separators=(",", ":")),
                section.markdown,
                section.embedding_text,
                _sha256(section.markdown),
                vector.data,
                vector.dimension,
            ),
        )

    def load_sections(self) -> list[StoredSection]:
        rows = self._connection.execute(
            """
            SELECT
                sections.section_id,
                sections.doc_id,
                sections.ordinal,
                documents.path,
                documents.title,
                sections.heading_path,
                sections.markdown,
                sections.embedding_text,
                sections.content_sha256,
                sections.embedding,
                sections.embedding_dimension
            FROM sections
            JOIN documents ON documents.doc_id = sections.doc_id
            ORDER BY documents.path, sections.ordinal
            """
        ).fetchall()
        stored_sections: list[StoredSection] = []
        for row in rows:
            if _sha256(row["markdown"]) != row["content_sha256"]:
                raise PersistenceError(
                    f"Stored section failed integrity check: {row['section_id']}"
                )
            embedding = _bytes_to_vector(row["embedding"], row["embedding_dimension"])
            stored_sections.append(
                StoredSection(
                    section=DocumentSection(
                        section_id=row["section_id"],
                        doc_id=row["doc_id"],
                        ordinal=row["ordinal"],
                        path=row["path"],
                        heading_path=tuple(json.loads(row["heading_path"])),
                        markdown=row["markdown"],
                        embedding_text=row["embedding_text"],
                    ),
                    document_title=row["title"],
                    embedding=embedding,
                )
            )
        return stored_sections

    def _embedding_dimension(self) -> int:
        row = self._connection.execute(
            "SELECT embedding_dimension FROM ingestions WHERE id = 1"
        ).fetchone()
        if row is None:
            raise PersistenceError("Ingestion metadata must be stored before sections")
        return row["embedding_dimension"]


@dataclass(frozen=True, slots=True)
class _SerializedVector:
    data: bytes
    dimension: int


def _vector_to_bytes(embedding: Sequence[float]) -> _SerializedVector:
    vector = np.asarray(embedding, dtype="<f4")
    if vector.ndim != 1 or vector.size == 0:
        raise PersistenceError("Embedding must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise PersistenceError("Embedding must contain only finite values")
    return _SerializedVector(data=vector.tobytes(), dimension=int(vector.size))


def _bytes_to_vector(data: bytes, dimension: int) -> NDArray[np.float32]:
    vector = np.frombuffer(data, dtype="<f4")
    if vector.size != dimension:
        raise PersistenceError("Stored embedding size does not match its dimension")
    return vector.copy()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
