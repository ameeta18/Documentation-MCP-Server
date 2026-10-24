from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from litellm_docs_mcp.ingestion import cli
from litellm_docs_mcp.ingestion.pipeline import IngestionSummary


def test_cli_applies_arguments_and_prints_safe_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    captured_config = None

    def fake_run_ingestion(config):
        nonlocal captured_config
        captured_config = config
        return IngestionSummary(
            database_path=config.database_path,
            source_commit="abc123",
            document_count=2,
            section_count=3,
            embedding_dimension=4,
            prompt_tokens=25,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-test-value")
    monkeypatch.setattr(cli, "run_ingestion", fake_run_ingestion)
    database_path = tmp_path / "docs.sqlite3"
    source_dir = tmp_path / "source"

    exit_code = cli.main(
        [
            "--repo-url",
            "https://example.test/docs.git",
            "--ref",
            "release",
            "--source-dir",
            str(source_dir),
            "--database",
            str(database_path),
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert captured_config.repository_url == "https://example.test/docs.git"
    assert captured_config.source_ref == "release"
    assert captured_config.source_dir == source_dir
    assert captured_config.database_path == database_path
    assert "Documents: 2" in output.out
    assert "Sections: 3" in output.out
    assert "Embedding prompt tokens: 25" in output.out
    assert "sensitive-test-value" not in output.out
    assert output.err == ""


def test_cli_module_is_independently_executable() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "litellm_docs_mcp.ingestion.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert "--repo-url" in result.stdout
    assert "--source-dir" in result.stdout
    assert result.stderr == ""
