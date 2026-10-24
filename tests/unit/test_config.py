from pathlib import Path

import pytest

from litellm_docs_mcp.config import ConfigurationError, IngestionConfig, ServerConfig


def test_ingestion_config_uses_safe_defaults() -> None:
    config = IngestionConfig.from_env({"OPENAI_API_KEY": "test-key"})

    assert config.database_path == Path("data/docs.sqlite3")
    assert config.repository_url == "https://github.com/BerriAI/litellm-docs.git"
    assert config.source_ref == "main"
    assert config.source_dir is None
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_batch_size == 64
    assert config.chunk_size == 700
    assert config.chunk_overlap == 100


def test_ingestion_config_accepts_environment_overrides() -> None:
    config = IngestionConfig.from_env(
        {
            "OPENAI_API_KEY": "test-key",
            "DOCS_MCP_DATABASE_PATH": "build/docs.db",
            "DOCS_MCP_REPO_URL": "https://example.test/docs.git",
            "DOCS_MCP_SOURCE_REF": "release",
            "DOCS_MCP_SOURCE_DIR": "fixtures/docs",
            "DOCS_MCP_EMBEDDING_MODEL": "test-model",
            "DOCS_MCP_EMBEDDING_BATCH_SIZE": "16",
            "DOCS_MCP_CHUNK_SIZE": "500",
            "DOCS_MCP_CHUNK_OVERLAP": "50",
        }
    )

    assert config.database_path == Path("build/docs.db")
    assert config.repository_url == "https://example.test/docs.git"
    assert config.source_ref == "release"
    assert config.source_dir == Path("fixtures/docs")
    assert config.embedding_model == "test-model"
    assert config.embedding_batch_size == 16
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50


def test_ingestion_config_requires_api_key() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        IngestionConfig.from_env({})


def test_ingestion_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ConfigurationError, match="must be greater than zero"):
        IngestionConfig.from_env(
            {"OPENAI_API_KEY": "test-key", "DOCS_MCP_EMBEDDING_BATCH_SIZE": "0"}
        )


def test_ingestion_config_requires_overlap_smaller_than_chunk_size() -> None:
    with pytest.raises(ConfigurationError, match="must be smaller"):
        IngestionConfig.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "DOCS_MCP_CHUNK_SIZE": "100",
                "DOCS_MCP_CHUNK_OVERLAP": "100",
            }
        )


def test_ingestion_config_repr_does_not_reveal_api_key() -> None:
    config = IngestionConfig.from_env({"OPENAI_API_KEY": "sensitive-test-value"})

    assert "sensitive-test-value" not in repr(config)


def test_server_config_uses_safe_defaults() -> None:
    config = ServerConfig.from_env({"OPENAI_API_KEY": "test-key"})

    assert config.database_path == Path("data/docs.sqlite3")
    assert config.host == "0.0.0.0"
    assert config.port == 8000


def test_server_config_accepts_environment_overrides() -> None:
    config = ServerConfig.from_env(
        {
            "OPENAI_API_KEY": "test-key",
            "DOCS_MCP_DATABASE_PATH": "build/docs.db",
            "DOCS_MCP_HOST": "127.0.0.1",
            "DOCS_MCP_PORT": "9000",
        }
    )

    assert config.database_path == Path("build/docs.db")
    assert config.host == "127.0.0.1"
    assert config.port == 9000


def test_server_config_requires_api_key() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        ServerConfig.from_env({})


def test_server_config_rejects_invalid_port() -> None:
    with pytest.raises(ConfigurationError, match="DOCS_MCP_PORT"):
        ServerConfig.from_env({"OPENAI_API_KEY": "test-key", "DOCS_MCP_PORT": "0"})
