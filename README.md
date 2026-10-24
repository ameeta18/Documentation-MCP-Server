# LiteLLM Documentation MCP Server

## Project description

This project provides a local Python MCP server for searching the Markdown documentation
under [`BerriAI/litellm-docs/docs`](https://github.com/BerriAI/litellm-docs/tree/main/docs).

It recursively ingests every `.md` file under `docs/`, preserves each complete original
document, creates searchable sections, generates embeddings, and stores the results in a
local SQLite database. Ingestion runs independently from the MCP server.

The server uses the official Model Context Protocol Python SDK and exposes exactly two tools:

- `search_docs(query, top_k)` performs semantic retrieval, applies the custom reranker, and
  returns documentation sections with source references. `top_k` defaults to 5 and accepts
  values from 1 to 20.
- `get_doc(doc_id)` returns the complete original Markdown document stored during ingestion.

## Architecture

Ingestion and serving are separate processes connected by the SQLite database:

```text
INGESTION (run separately)

LiteLLM docs/ repository
        |
        v
Discover all Markdown files recursively
        |
        v
Store each original document and create searchable sections
        |
        v
Generate section embeddings
        |
        v
SQLite database
  - complete original documents
  - searchable sections and metadata
  - embedding vectors


MCP SERVER (reads the completed database)

search_docs(query, top_k)
        |
        v
Embed query -> cosine similarity -> custom reranker -> ranked sections and sources

get_doc(doc_id)
        |
        v
SQLite lookup -> complete original Markdown document
```

The ingestion command creates a shallow Git checkout of `docs/`, records the resolved
commit, and discovers Markdown files recursively. A local source-directory option is also
available for tests and offline development.

Markdown is split into heading-aware, size-bounded sections and embedded with
`text-embedding-3-small`. The complete original documents remain unchanged.

SQLite stores the documents, sections, metadata, and embedding vectors. The server loads
the vectors into NumPy for exact cosine search, then applies a deterministic reranker using
semantic similarity and smaller lexical signals.

Detailed design decisions are recorded in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Installation

Python 3.12 is the target runtime.

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

## API key handling

The OpenAI API key is read from the `OPENAI_API_KEY` environment variable. Never commit the
key or include it in the Docker image. Local environment files and generated databases are
excluded through `.gitignore` and `.dockerignore`.

Linux and macOS:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = Read-Host "OpenAI API key" -MaskInput
```

## Ingestion

After exporting `OPENAI_API_KEY`, run:

```bash
python -m litellm_docs_mcp.ingestion.cli \
  --repo-url https://github.com/BerriAI/litellm-docs.git \
  --ref main \
  --database ./data/docs.sqlite3
```

For an existing local checkout containing a `docs/` directory:

```bash
python -m litellm_docs_mcp.ingestion.cli \
  --source-dir /path/to/local/litellm-docs \
  --database ./data/docs.sqlite3
```

The command prints the resolved commit, document and section counts, embedding dimension,
and prompt-token usage. It does not print the API key. The generated database is excluded
from Git.

The full-corpus ingestion produced the following results:

| Measurement | Result |
|---|---:|
| Source commit | `245dd496cca7d0bcd4bb662d5a2e6a4326b1c3eb` |
| Markdown documents | 953 |
| Searchable sections | 13,193 |
| Embedding dimensions | 1,536 |
| Ingestion duration | 213.47 seconds |
| Database size | 120.48 MiB |

These measurements describe one ingestion run. They may change with the source commit,
chunking settings, embedding model, or development machine.

## Start the MCP server

The server needs a completed database and `OPENAI_API_KEY`. Query text is embedded at search
time, so `search_docs` requires network access to the OpenAI embeddings API.

```bash
export DOCS_MCP_DATABASE_PATH=./data/docs.sqlite3
python -m litellm_docs_mcp.mcp_server.app
```

PowerShell:

```powershell
$env:DOCS_MCP_DATABASE_PATH = ".\data\docs.sqlite3"
python -m litellm_docs_mcp.mcp_server.app
```

The default MCP endpoint is `http://127.0.0.1:8000/mcp`. The host and port can be changed
with `DOCS_MCP_HOST` and `DOCS_MCP_PORT`.

## Run in Docker

Build the image:

```bash
docker build --tag litellm-docs-mcp .
```

Linux and macOS:

```bash
docker run --rm \
  --publish 127.0.0.1:8000:8000 \
  --env OPENAI_API_KEY \
  --volume "$(pwd)/data:/app/data:ro" \
  litellm-docs-mcp
```

Windows PowerShell:

```powershell
docker run --rm `
  --publish 127.0.0.1:8000:8000 `
  --env OPENAI_API_KEY `
  --volume "${PWD}/data:/app/data:ro" `
  litellm-docs-mcp
```

The image runs as a non-root user. The database is mounted read-only at
`/app/data/docs.sqlite3`, and the API key is passed from the current shell at container
startup. It is not included in the image.

## Testing

Run the automated suite:

```bash
python -m pytest
```

Run the style checks:

```bash
python -m ruff check .
python -m ruff format --check .
```

Automated tests run locally without making external API requests. Coverage includes
ingestion, Markdown processing, persistence, semantic retrieval, reranking, source
references, and the two-tool MCP contract over Streamable HTTP. All unit and integration
tests pass locally, and the Docker image was smoke-tested with the official MCP client.

### Retrieval review

A manual review used representative questions about model fallbacks, Azure OpenAI, virtual
keys, and logging callbacks. All four searches returned relevant documentation within the
top five results.

| Question | Relevant result | Rank |
|---|---|---:|
| Model fallbacks | `docs/tutorials/model_fallbacks.md` | 1 |
| Azure OpenAI | `docs/providers/azure/azure.md` | 1 |
| Virtual keys | `docs/proxy/virtual_keys.md` | 5 |
| Logging callbacks | `docs/observability/custom_callback.md` | 2 |

The dedicated fallback and Azure OpenAI documentation ranked first. For virtual keys and
logging callbacks, relevant general documentation appeared within the top five, although
more specialized pages sometimes ranked higher. Source URLs were pinned to the ingested
commit, and a returned document ID was successfully passed to `get_doc` to retrieve the
complete original Markdown.

This was a small practical review, not a formal retrieval benchmark.

## Roadmap

- Maintain compatibility with future LiteLLM documentation and MCP SDK changes.
- Build a small labelled query set if more systematic retrieval-quality measurement is needed.
- Consider incremental ingestion only if documentation refreshes become frequent enough to
  justify the added complexity.
