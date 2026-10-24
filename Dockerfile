FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCS_MCP_DATABASE_PATH=/app/data/docs.sqlite3 \
    DOCS_MCP_HOST=0.0.0.0 \
    DOCS_MCP_PORT=8000

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /app/data \
    && chown app:app /app/data

USER app

EXPOSE 8000

CMD ["python", "-m", "litellm_docs_mcp.mcp_server.app"]
