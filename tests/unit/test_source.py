from __future__ import annotations

import subprocess
from pathlib import Path

from litellm_docs_mcp.ingestion.source import (
    acquire_source,
    discover_documents,
    document_id_for_path,
)


def test_local_source_discovers_nested_markdown_in_path_order(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "nested").mkdir(parents=True)
    (docs_root / "z-last.md").write_text("# Last\n", encoding="utf-8")
    (docs_root / "a-first.md").write_text("# First\n", encoding="utf-8")
    (docs_root / "nested" / "middle.md").write_text("# Middle\n", encoding="utf-8")
    (docs_root / "ignored.txt").write_text("Not Markdown", encoding="utf-8")

    with acquire_source("unused", "unused", source_dir=tmp_path) as snapshot:
        documents = discover_documents(snapshot)

    assert [document.path for document in documents] == [
        "docs/a-first.md",
        "docs/nested/middle.md",
        "docs/z-last.md",
    ]
    assert [document.markdown for document in documents] == [
        "# First\n",
        "# Middle\n",
        "# Last\n",
    ]


def test_document_id_is_stable_for_the_same_normalized_path() -> None:
    posix_id = document_id_for_path("docs/providers/openai.md")
    windows_id = document_id_for_path(r"docs\providers\openai.md")

    assert posix_id == windows_id
    assert posix_id.startswith("doc_")
    assert len(posix_id) == 24


def test_git_source_uses_sparse_checkout_and_records_commit(tmp_path: Path) -> None:
    source_repository = tmp_path / "source"
    source_repository.mkdir()
    _git(source_repository, "init", "-b", "main")
    (source_repository / "docs").mkdir()
    (source_repository / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (source_repository / "outside.txt").write_text("Not required", encoding="utf-8")
    _git(source_repository, "add", ".")
    _git(
        source_repository,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "Test documentation",
    )
    expected_commit = _git(source_repository, "rev-parse", "HEAD")

    with acquire_source(str(source_repository), "main") as snapshot:
        assert snapshot.commit_sha == expected_commit
        assert (snapshot.docs_root / "guide.md").is_file()
        assert not (snapshot.repository_root / "outside.txt").exists()
        assert [document.path for document in discover_documents(snapshot)] == ["docs/guide.md"]


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()
