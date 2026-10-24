from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from litellm_docs_mcp.config import ServerConfig
from litellm_docs_mcp.embeddings import EmbeddingResult
from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.mcp_server.app import MCP_PATH, create_mcp_server
from litellm_docs_mcp.mcp_server.runtime import create_server_runtime
from litellm_docs_mcp.mcp_server.tools import DocumentationTools
from litellm_docs_mcp.persistence import IngestionMetadata, SQLiteStore


@dataclass
class _FakeQueryEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=((1.0, 0.0),), prompt_tokens=1)


def _write_database(database_path: Path) -> Document:
    document = Document(
        doc_id="doc_setup",
        path="docs/setup.md",
        title="Setup Guide",
        markdown="# Setup Guide\n\nInstall the package.\n",
    )
    section = DocumentSection(
        section_id="doc_setup:0000",
        doc_id=document.doc_id,
        ordinal=0,
        path=document.path,
        heading_path=("Setup Guide",),
        markdown=document.markdown,
        embedding_text="Setup Guide Install the package.",
    )
    metadata = IngestionMetadata(
        source_repository="https://github.com/BerriAI/litellm-docs.git",
        source_ref="main",
        source_commit="abc123",
        embedding_model="test-model",
        embedding_dimension=2,
        chunk_size=700,
        chunk_overlap=100,
    )
    with SQLiteStore.create(database_path) as store, store.transaction():
        store.save_ingestion(metadata)
        store.save_document(document)
        store.save_section(section, [1.0, 0.0])
    return document


@asynccontextmanager
async def _running_server(app: object) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False, lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("Test MCP server did not start")
        yield f"http://127.0.0.1:{port}{MCP_PATH}"
    finally:
        server.should_exit = True
        await task
        listener.close()


@pytest.mark.asyncio
async def test_streamable_http_exposes_only_the_required_tool_contract(tmp_path: Path) -> None:
    database_path = tmp_path / "docs.sqlite3"
    expected_document = _write_database(database_path)
    config = ServerConfig(openai_api_key="test-key", database_path=database_path)

    with create_server_runtime(config, _FakeQueryEmbeddingProvider()) as runtime:
        mcp_server = create_mcp_server(DocumentationTools(runtime))
        app = mcp_server.streamable_http_app(
            streamable_http_path=MCP_PATH,
            host="127.0.0.1",
        )

        async with (
            _running_server(app) as endpoint,
            streamable_http_client(endpoint) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            listed = await session.list_tools()
            assert [tool.name for tool in listed.tools] == ["search_docs", "get_doc"]

            search_result = await session.call_tool(
                "search_docs",
                {"query": "install package", "top_k": 1},
            )
            assert not search_result.is_error
            assert search_result.structured_content is not None
            results = search_result.structured_content["result"]
            assert len(results) == 1
            assert results[0]["doc_id"] == expected_document.doc_id
            assert results[0]["source"]["url"].endswith("/blob/abc123/docs/setup.md")

            document_result = await session.call_tool(
                "get_doc",
                {"doc_id": expected_document.doc_id},
            )
            assert not document_result.is_error
            assert document_result.structured_content is not None
            assert document_result.structured_content["markdown"] == expected_document.markdown

            invalid_search = await session.call_tool(
                "search_docs",
                {"query": "install", "top_k": 0},
            )
            assert invalid_search.is_error

            missing_document = await session.call_tool(
                "get_doc",
                {"doc_id": "doc_missing"},
            )
            assert missing_document.is_error
            assert "Document not found" in missing_document.content[0].text
