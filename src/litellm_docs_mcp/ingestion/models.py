"""Data passed between ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """A source document whose original Markdown must be preserved."""

    doc_id: str
    path: str
    title: str
    markdown: str


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A searchable section derived from one source document."""

    section_id: str
    doc_id: str
    ordinal: int
    path: str
    heading_path: tuple[str, ...]
    markdown: str
    embedding_text: str
