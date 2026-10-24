PRAGMA foreign_keys = ON;

CREATE TABLE ingestions (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    source_repository TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_commit TEXT,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap INTEGER NOT NULL CHECK (
        chunk_overlap >= 0 AND chunk_overlap < chunk_size
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    ingestion_id INTEGER NOT NULL DEFAULT 1 REFERENCES ingestions(id),
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    original_markdown TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
);

CREATE TABLE sections (
    section_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    heading_path TEXT NOT NULL,
    markdown TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX sections_document_order ON sections(doc_id, ordinal);

PRAGMA user_version = 1;
