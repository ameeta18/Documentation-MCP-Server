# LiteLLM Documentation MCP Server

> **Current status:** Planning and repository setup. The ingestion pipeline, search implementation, MCP server, tests, and container have not been implemented yet. Commands in this README are proposed interfaces and have not been verified.

## Project description

This project will provide a local Python MCP server for searching the Markdown documentation in [`BerriAI/litellm-docs/docs`](https://github.com/BerriAI/litellm-docs/tree/main/docs).

The finished system will:

- recursively process every `.md` file under `docs/`, including subdirectories;
- preserve the complete original Markdown documents;
- create searchable sections and store their generated embeddings;
- run ingestion independently from the server;
- implement semantic search followed by a small custom reranker; and
- run locally in a container using the official Model Context Protocol Python SDK.

The MCP server will expose exactly two tools:

- `search_docs(query, top_k)` searches documentation sections, applies the custom reranker, and returns source references.
- `get_doc(doc_id)` returns the complete original Markdown document associated with an ingestion-assigned document ID.

No ingestion, health, statistics, or debugging tools will be exposed through MCP.

## Planned architecture

Ingestion and serving are separate processes.

The planned ingestion command will create a temporary shallow Git checkout with sparse checkout configured for the `docs/` directory. It will record the resolved commit, discover `docs/**/*.md`, preserve each original document, split documents into searchable sections, generate OpenAI embeddings, and store the results in SQLite.

The planned server will load stored vectors into NumPy and use cosine similarity for semantic retrieval. A deterministic local reranker will then apply smaller boosts for useful lexical signals such as query-term, title, heading, and exact-phrase matches. Search results will include document IDs, paths, heading context, and commit-pinned source URLs.

The proposed embedding model is `text-embedding-3-small`. This choice and the final chunking and reranking settings still need to be validated during implementation.

Detailed design decisions and the implementation schedule are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Current repository contents

The repository currently contains:

- the project plan and this README;
- Python dependency metadata in `pyproject.toml`;
- safe environment-variable documentation in `.env.example`;
- Git and Docker exclusion rules; and
- the initial Python package directory.

## Proposed installation

Python 3.12 is the proposed runtime.

Linux and macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The declared dependencies include the official MCP Python SDK, the OpenAI Python library, NumPy, Markdown parsing, and test tooling. SQLite support comes from the Python standard library. Dependency versions will be validated and pinned before submission.

## API key handling

The OpenAI API key must be supplied only through the `OPENAI_API_KEY` environment variable. Do not put it in source code, `.env.example`, a real `.env` file, command output, logs, Docker build arguments, or Docker layers.

Linux and macOS:

```bash
export OPENAI_API_KEY="<provided-key>"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "<provided-key>"
```

The example values above are placeholders. `.gitignore` and `.dockerignore` provide additional protection for local secret files and generated data.

## Proposed ingestion

The intended command-line interface is:

```bash
python -m litellm_docs_mcp.ingestion.cli \
  --repo-url https://github.com/BerriAI/litellm-docs.git \
  --ref main \
  --database ./data/docs.sqlite3
```

The command will acquire the selected Git revision, process every Markdown file recursively under `docs/`, generate embeddings, and create the local SQLite database. The generated database will be excluded from Git.

## Proposed startup

The intended local development command is:

```bash
python -m litellm_docs_mcp.mcp_server.app
```

The planned transport is Streamable HTTP through a published local container port. Port `8000` is currently a configuration default, not a verified endpoint.

The expected container workflow is:

```bash
docker build -t litellm-docs-mcp .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY \
  -e DOCS_MCP_DATABASE_PATH=/app/data/docs.sqlite3 \
  -v "$(pwd)/data:/app/data:ro" \
  litellm-docs-mcp
```

The final command and MCP endpoint will be confirmed during container testing.

## Proposed testing

The test suite will use small Markdown fixtures and fake deterministic embeddings for normal automated tests. Planned coverage includes recursive discovery, original-document preservation, stable document IDs, section relationships, semantic retrieval, reranking, source references, unknown document IDs, the exact two-tool MCP contract, secret exclusion, and a container smoke test.

## Roadmap

- **October 19:** finalize the plan and repository scaffold, create the private GitHub repository, and grant reviewer access.
- **October 20:** implement recursive ingestion, Markdown processing, embedding generation, and SQLite persistence.
- **October 21:** implement semantic retrieval and custom reranking.
- **October 22:** implement the two MCP tools and local container execution.
- **October 23:** run full-corpus ingestion, automated tests, retrieval review, documentation checks, and secret checks.
- **October 24:** complete clean-clone verification, final fixes, and submission.
