"""Create and run the local Streamable HTTP MCP server."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from litellm_docs_mcp.config import ServerConfig
from litellm_docs_mcp.mcp_server.runtime import create_server_runtime
from litellm_docs_mcp.mcp_server.tools import (
    DocumentationTools,
    DocumentNotFoundError,
    ToolInputError,
)
from litellm_docs_mcp.retrieval import SemanticSearchError
from litellm_docs_mcp.retrieval.search import SearchServiceError

MCP_PATH = "/mcp"


def create_mcp_server(tools: DocumentationTools) -> MCPServer:
    """Register exactly the two documentation tools required by the assignment."""

    server = MCPServer(
        name="litellm-docs-mcp",
        title="LiteLLM Documentation MCP Server",
        description="Search and retrieve the ingested LiteLLM Markdown documentation.",
    )

    @server.tool(name="search_docs")
    def search_docs(query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Search documentation sections and return reranked results with sources."""

        try:
            return tools.search_docs(query, top_k)
        except (ToolInputError, SearchServiceError, SemanticSearchError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(name="get_doc")
    def get_doc(doc_id: str) -> dict[str, object]:
        """Return the complete original Markdown document for a document ID."""

        try:
            return tools.get_doc(doc_id)
        except (ToolInputError, DocumentNotFoundError) as exc:
            raise ToolError(str(exc)) from exc

    return server


def main() -> None:
    """Load configuration and run the MCP server until it is stopped."""

    config = ServerConfig.from_env()
    with create_server_runtime(config) as runtime:
        server = create_mcp_server(DocumentationTools(runtime))
        server.run(
            transport="streamable-http",
            host=config.host,
            port=config.port,
            streamable_http_path=MCP_PATH,
        )


if __name__ == "__main__":
    main()
