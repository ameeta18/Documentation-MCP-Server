"""Acquire and discover LiteLLM Markdown source documents."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from litellm_docs_mcp.ingestion.models import Document


class SourceError(RuntimeError):
    """Raised when documentation source acquisition or discovery fails."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A local documentation tree and its resolved source commit."""

    repository_root: Path
    docs_root: Path
    commit_sha: str | None


@contextmanager
def acquire_source(
    repository_url: str,
    source_ref: str,
    source_dir: Path | None = None,
) -> Iterator[SourceSnapshot]:
    """Open a local source tree or create a temporary sparse Git checkout."""

    if source_dir is not None:
        yield _open_local_source(source_dir)
        return

    with tempfile.TemporaryDirectory(prefix="litellm-docs-") as temp_dir:
        repository_root = Path(temp_dir) / "repository"
        _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--single-branch",
                "--branch",
                source_ref,
                "--",
                repository_url,
                str(repository_root),
            ],
            operation="clone documentation repository",
        )
        _run_git(
            [
                "-C",
                str(repository_root),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/docs/",
            ],
            operation="select the docs directory",
        )
        commit_sha = _run_git(
            ["-C", str(repository_root), "rev-parse", "HEAD"],
            operation="resolve source commit",
        )
        yield _snapshot(repository_root, commit_sha)


def discover_documents(snapshot: SourceSnapshot) -> list[Document]:
    """Read every Markdown document below docs in deterministic path order."""

    paths = sorted(
        (path for path in snapshot.docs_root.rglob("*.md") if path.is_file()),
        key=lambda path: path.relative_to(snapshot.repository_root).as_posix(),
    )

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for path in paths:
        relative_path = path.relative_to(snapshot.repository_root).as_posix()
        doc_id = document_id_for_path(relative_path)
        if doc_id in seen_ids:
            raise SourceError(f"Document ID collision for repository path: {relative_path}")
        seen_ids.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                path=relative_path,
                title=path.stem,
                markdown=path.read_text(encoding="utf-8"),
            )
        )
    return documents


def document_id_for_path(relative_path: str) -> str:
    """Return a stable ID based only on a normalized repository-relative path."""

    normalized_path = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized_path.is_absolute() or ".." in normalized_path.parts:
        raise ValueError("Document path must be repository-relative")
    path_text = normalized_path.as_posix()
    digest = hashlib.sha256(path_text.encode("utf-8")).hexdigest()[:20]
    return f"doc_{digest}"


def _open_local_source(source_dir: Path) -> SourceSnapshot:
    repository_root = source_dir.expanduser().resolve()
    commit_sha = _try_resolve_commit(repository_root)
    return _snapshot(repository_root, commit_sha)


def _snapshot(repository_root: Path, commit_sha: str | None) -> SourceSnapshot:
    docs_root = repository_root / "docs"
    if not docs_root.is_dir():
        raise SourceError(f"Documentation directory does not exist: {docs_root}")
    return SourceSnapshot(
        repository_root=repository_root,
        docs_root=docs_root,
        commit_sha=commit_sha,
    )


def _try_resolve_commit(repository_root: Path) -> str | None:
    result = _run_git(
        ["-C", str(repository_root), "rev-parse", "HEAD"],
        operation="resolve local source commit",
        check=False,
    )
    return result or None


def _run_git(arguments: list[str], operation: str, *, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise SourceError("Git is required to acquire the documentation source") from exc

    if check and result.returncode != 0:
        raise SourceError(f"Could not {operation} (Git exit code {result.returncode})")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
