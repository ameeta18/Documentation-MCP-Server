"""Create searchable, size-bounded sections from parsed Markdown."""

from __future__ import annotations

import re
from dataclasses import replace

import tiktoken
from markdown_it import MarkdownIt

from litellm_docs_mcp.ingestion.models import Document, DocumentSection
from litellm_docs_mcp.markdown.parser import parse_markdown

_IMPORT_LINE = re.compile(r"(?m)^\s*import\s+.*?;?\s*$")
_BASE64_IMAGE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_MARKDOWN_IMAGE = re.compile(r"!\[([^]]*)]\([^)]+\)")


def section_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[Document, list[DocumentSection]]:
    """Return the titled document and its searchable sections."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    parsed = parse_markdown(document.markdown)
    title = parsed.title or document.title
    titled_document = replace(document, title=title)
    encoding = tiktoken.get_encoding("cl100k_base")

    sections: list[DocumentSection] = []
    for parsed_section in parsed.sections:
        chunks = _chunk_markdown(
            parsed_section.markdown,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding,
        )
        for markdown_chunk in chunks:
            ordinal = len(sections)
            sections.append(
                DocumentSection(
                    section_id=f"{document.doc_id}:{ordinal:04d}",
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    path=document.path,
                    heading_path=parsed_section.heading_path,
                    markdown=markdown_chunk,
                    embedding_text=_embedding_text(
                        title,
                        parsed_section.heading_path,
                        markdown_chunk,
                    ),
                )
            )
    return titled_document, sections


def _chunk_markdown(
    markdown: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    blocks = _markdown_blocks(markdown)
    bounded_blocks = [
        piece for block in blocks for piece in _split_oversized_block(block, chunk_size, encoding)
    ]

    chunks: list[str] = []
    current: list[str] = []
    for block in bounded_blocks:
        candidate = "".join([*current, block])
        if current and _token_count(candidate, encoding) > chunk_size:
            chunks.append("".join(current))
            current = _overlap_suffix(current, chunk_overlap, encoding)
            while current and _token_count("".join([*current, block]), encoding) > chunk_size:
                current.pop(0)
        current.append(block)

    if current:
        chunks.append("".join(current))
    return chunks or [""]


def _markdown_blocks(markdown: str) -> list[str]:
    if not markdown:
        return [""]
    lines = markdown.splitlines(keepends=True)
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    starts = {0}
    starts.update(token.map[0] for token in tokens if token.level == 0 and token.map)
    ordered_starts = sorted(starts)
    return [
        "".join(lines[start:end])
        for start, end in zip(ordered_starts, [*ordered_starts[1:], len(lines)], strict=True)
        if start < end
    ]


def _split_oversized_block(
    block: str,
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    if _token_count(block, encoding) <= chunk_size:
        return [block]

    pieces: list[str] = []
    current = ""
    for line in block.splitlines(keepends=True):
        candidate = current + line
        if current and _token_count(candidate, encoding) > chunk_size:
            pieces.append(current)
            current = ""
        if _token_count(line, encoding) > chunk_size:
            pieces.extend(_split_text_by_tokens(line, chunk_size, encoding))
        else:
            current += line
    if current:
        pieces.append(current)
    return pieces


def _split_text_by_tokens(
    text: str,
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while remaining:
        low = 1
        high = len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if _token_count(remaining[:middle], encoding) <= chunk_size:
                low = middle
            else:
                high = middle - 1
        pieces.append(remaining[:low])
        remaining = remaining[low:]
    return pieces


def _overlap_suffix(
    blocks: list[str],
    overlap: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    selected: list[str] = []
    for block in reversed(blocks):
        candidate = [block, *selected]
        if _token_count("".join(candidate), encoding) > overlap:
            break
        selected = candidate
    return selected


def _embedding_text(title: str, heading_path: tuple[str, ...], markdown: str) -> str:
    context_parts = [title]
    context_parts.extend(
        heading for heading in heading_path if heading.casefold() != title.casefold()
    )
    normalized = _normalize_markdown(markdown)
    return "\n\n".join(part for part in [" > ".join(context_parts), normalized] if part)


def _normalize_markdown(markdown: str) -> str:
    text = _IMPORT_LINE.sub("", markdown)
    text = _BASE64_IMAGE.sub("[embedded image omitted]", text)
    text = _MARKDOWN_IMAGE.sub(lambda match: match.group(1), text)
    return " ".join(text.split())


def _token_count(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))
