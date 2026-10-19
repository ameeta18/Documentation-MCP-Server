# Project Plan

## Objective and Scope

The goal is to build a Python MCP server that makes the LiteLLM documentation searchable. The system will recursively ingest every Markdown file under the repository's `docs/` directory, including all subdirectories. It will preserve each complete original Markdown document, create smaller searchable sections, generate embeddings for those sections, and store the documents, sections, and embeddings locally.

The server will expose exactly two MCP tools:

- `search_docs(query, top_k)` retrieves relevant documentation sections using semantic search, applies a small custom reranker, and returns the best results with source references.
- `get_doc(doc_id)` returns the complete original Markdown text for the document ID assigned during ingestion.

The project will use the official Model Context Protocol Python SDK and will run locally in a container. Ingestion will be a separate command so that the documentation database can be created or refreshed without starting the MCP server.

The project does not include a hosted deployment, web interface, authentication system, scheduled synchronization, answer generation, or ingestion of non-Markdown assets. The generated database is a local artifact and will not be committed to Git.

The OpenAI API key will be read only from the `OPENAI_API_KEY` environment variable. It will not be stored in source code, local data files, logs, Docker build arguments, Docker layers, or Git history. `.env.example` will document variable names without containing a secret.


## Work Packages and Timeline

| Date | Work package | Planned result |
|---|---|---|
| Oct 19 | Planning and repository setup | Final project plan, private repository access, safe scaffold, dependency configuration, and README outline |
| Oct 20 | Ingestion and persistence | Recursive discovery, stable IDs, Markdown sectioning, embedding generation, SQLite schema, and independently executable ingestion command |
| Oct 21 | Search and reranking | Query embeddings, NumPy cosine retrieval, custom reranker, and source-reference formatting |
| Oct 22 | MCP server and container | Exactly two MCP tools available through Streamable HTTP in a local container |
| Oct 23 | Full-corpus validation | Complete ingestion, automated tests, retrieval review, README instructions, and secret checks |
| Oct 24 | Final verification and submission | Clean-clone installation, ingestion and container smoke test, final fixes, and submission |

## Proposed Architecture

The design separates ingestion from serving.

### Ingestion

The ingestion command will obtain the LiteLLM documentation at a selected Git reference and record the resolved commit. It will find all `docs/**/*.md` files in deterministic path order. A local source-directory option will also be supported for tests.

The documentation is spread across nested directories and includes front matter, tables, code blocks, MDX components, and several large documents. The ingestion summary will record the final document and section counts.

For each document, ingestion will:

1. derive a stable document ID from its repository-relative path;
2. store the complete original Markdown text and source information;
3. parse the document into heading-aware sections;
4. split large sections into size-bounded chunks while preserving heading context and avoiding unnecessary splits inside fenced code blocks;
5. create normalized embedding text without changing the stored original document;
6. generate embeddings in batches; and
7. store sections, metadata, and embeddings in SQLite.

Initial chunk-size and overlap settings will be chosen conservatively and validated with representative documentation queries. Presentation-only content, such as MDX imports or inline base64 image data, can be removed from embedding input while remaining untouched in the stored original Markdown.

### Search and document retrieval

At server startup, stored section vectors will be loaded from SQLite into a NumPy matrix. `search_docs` will embed the query, calculate cosine similarity against the stored vectors, and pass a larger candidate set to the custom reranker. The final results will include the document ID, repository-relative path, heading context, content excerpt, relevance score, and a commit-pinned GitHub URL.

`get_doc` will query SQLite by `doc_id` and return the complete original Markdown text together with basic source metadata. An unknown ID will produce a clear not-found response.

### MCP server and container

The server will use Streamable HTTP. It provides a straightforward way to access the MCP server through a published local container port. The server will open an existing database and will not run ingestion automatically. Only `search_docs` and `get_doc` will be registered as MCP tools.

## Key Technical Decisions

### Python and MCP SDK

Python 3.12 is the proposed runtime. The implementation will use a pinned and tested version of the official MCP Python SDK. The validated version will be recorded in `pyproject.toml`.

### Documentation acquisition

A shallow sparse Git clone is the preferred source method because it can fetch the `docs/` tree and provide the exact commit identifier without making hundreds of GitHub API requests. The ingestion command will also accept a local documentation directory, which keeps automated tests independent of the network.

### Markdown processing

`markdown-it-py` will provide Markdown-aware block and line information. This is more reliable than splitting with regular expressions, especially when code blocks contain heading-like text. YAML front matter will be handled separately. MDX components will be treated as content rather than executed or rendered.

### Document IDs and source information

A stable ID will be derived from each repository-relative document path. The original path will remain available in stored metadata and tool responses. Each search result will include enough information to locate the source document at the commit used for ingestion.

### Embeddings

OpenAI `text-embedding-3-small` is the proposed embedding model, subject to validation with the supplied key. It offers a suitable balance of retrieval quality, storage size, and API cost for this corpus. Document sections and queries must use the same model and vector dimensions. Embedding requests will be batched and monitored to remain within the supplied API key's $20 usage limit. Retries will be limited to transient failures.

### Storage and retrieval

SQLite will store ingestion information, complete documents, sections, and embedding vectors in one portable database. The server will load the stored vectors into NumPy and use exact cosine similarity. This avoids adding a separate vector database for a corpus of this size.

### Custom reranking

The reranker will be implemented locally with transparent scoring logic. Semantic similarity will remain the main signal. Smaller boosts may be applied for query-term coverage, matches in document titles or headings, and exact-phrase matches. The final weights will be selected after testing representative queries and will be documented in the code.

The reranker will be deterministic: identical inputs and stored data will produce the same ordering. It will not call an LLM or use a third-party reranking service.

## Repository Structure

```text
.
├── .env.example
├── .gitignore
├── Dockerfile
├── PROJECT_PLAN.md
├── README.md
├── pyproject.toml
├── src/
│   └── litellm_docs_mcp/
│       ├── config.py
│       ├── ingestion/
│       │   ├── cli.py
│       │   ├── source.py
│       │   └── pipeline.py
│       ├── markdown/
│       │   ├── parser.py
│       │   └── chunker.py
│       ├── embeddings/
│       │   └── openai_client.py
│       ├── persistence/
│       │   ├── schema.sql
│       │   └── sqlite_store.py
│       ├── retrieval/
│       │   └── semantic.py
│       ├── reranking/
│       │   └── scorer.py
│       └── mcp_server/
│           ├── app.py
│           └── tools.py
└── tests/
    ├── fixtures/
    ├── unit/
    └── integration/
```

The package structure separates ingestion, processing, persistence, retrieval, reranking, and MCP tool definitions.

## Testing Approach

Normal automated tests will use small Markdown fixtures and fake deterministic embeddings. This keeps the suite fast, repeatable, and independent of API availability. Real OpenAI calls will be limited to the full ingestion and an optional small smoke test.

The main tests will verify:

- recursive discovery includes nested Markdown files and excludes unrelated files;
- the complete original Markdown text is preserved and returned by `get_doc`;
- document IDs remain deterministic for the same repository-relative paths;
- every stored section belongs to a valid document;
- semantic retrieval returns the expected sections for a small controlled corpus;
- the custom reranker changes ordering for relevant term, title, heading, or phrase matches;
- search results contain usable source references;
- an unknown document ID returns a clear error;
- MCP tool discovery returns exactly `search_docs` and `get_doc`;
- the container starts locally and both tools can be called; and
- secrets and generated database files are excluded from Git and Docker build output.

After the implementation is stable on fixtures, the full corpus will be ingested. The resulting document and section counts, database size, ingestion duration, and a small set of representative search results will be recorded in the README.

## Main Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Markdown and MDX structures produce poor section boundaries | Use Markdown block parsing, retain heading context, test observed edge cases, and preserve the untouched original document separately. |
| OpenAI budget, rate limits, or temporary API failures interrupt ingestion | Use the smaller embedding model, batch requests, apply bounded retries, and keep failed runs from replacing a valid database where practical. |
| Semantic results or reranking are not useful enough | Test representative LiteLLM questions, inspect candidate ordering, and adjust chunk settings and reranker weights based on observed results. |
| MCP or container integration behaves differently from local Python execution | Add an integration test with the official SDK client and run a container smoke test from a clean build. |
| The API key is exposed accidentally | Read it only from the runtime environment, exclude secret files from Git and Docker contexts, avoid logging configuration values, and inspect tracked files, Docker configuration, and Git history before submission. |

## Definition of Done

The project is complete when:

- the implementation uses Python and a pinned, tested version of the official MCP Python SDK;
- the ingestion command independently processes every Markdown file under `docs/`, including subdirectories;
- SQLite contains the complete original Markdown documents, searchable sections, metadata, and valid embeddings;
- document IDs are stable and `get_doc(doc_id)` returns the complete original Markdown text;
- `search_docs(query, top_k)` performs semantic retrieval, applies the custom reranker, respects `top_k`, and returns source references;
- MCP tool discovery exposes exactly `search_docs` and `get_doc`, with no ingestion, health, statistics, or debugging tools;
- the server runs locally in a container through Streamable HTTP;
- automated tests cover the main ingestion, storage, retrieval, reranking, MCP, and error-handling behavior;
- the README contains the project description, architecture overview, installation, ingestion, startup, and roadmap instructions;
- the OpenAI key is absent from source code, Git history, logs, generated files, and Docker layers; and
- the documented workflow succeeds from a clean clone before the October 24, 2026 submission.
