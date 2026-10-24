from __future__ import annotations

import tiktoken

from litellm_docs_mcp.ingestion.models import Document
from litellm_docs_mcp.markdown import section_document


def _document(markdown: str) -> Document:
    return Document(
        doc_id="doc_test",
        path="docs/guide.md",
        title="guide",
        markdown=markdown,
    )


def test_front_matter_sets_title_without_changing_original_markdown() -> None:
    markdown = "---\ntitle: API Guide\nsidebar_label: Guide\n---\n# Introduction\nHello.\n"

    document, sections = section_document(_document(markdown), 100, 10)

    assert document.title == "API Guide"
    assert document.markdown == markdown
    assert sections[0].heading_path == ("Introduction",)
    assert "title: API Guide" not in sections[0].embedding_text


def test_heading_hierarchy_is_kept_for_each_section() -> None:
    markdown = (
        "# API\nOverview.\n"
        "## Authentication\nUse a key.\n"
        "### Headers\nSet a header.\n"
        "## Errors\nHandle errors.\n"
    )

    _, sections = section_document(_document(markdown), 100, 10)

    assert [section.heading_path for section in sections] == [
        ("API",),
        ("API", "Authentication"),
        ("API", "Authentication", "Headers"),
        ("API", "Errors"),
    ]


def test_heading_like_text_inside_fence_does_not_create_section() -> None:
    markdown = "# Example\n```markdown\n# Not a real heading\n```\nMore text.\n"

    _, sections = section_document(_document(markdown), 100, 10)

    assert len(sections) == 1
    assert sections[0].heading_path == ("Example",)
    assert "# Not a real heading" in sections[0].markdown


def test_embedding_cleanup_does_not_change_stored_markdown() -> None:
    markdown = (
        "import Tabs from '@theme/Tabs';\n\n"
        "# Images\n"
        "![Architecture](data:image/png;base64,AAAA)\n"
        "See [the guide](https://example.com).\n"
    )

    document, sections = section_document(_document(markdown), 100, 10)

    assert document.markdown == markdown
    assert "import Tabs" not in sections[0].embedding_text
    assert "AAAA" not in sections[1].embedding_text
    assert "Architecture" in sections[1].embedding_text
    assert "the guide" in sections[1].embedding_text


def test_large_section_is_split_and_keeps_heading_context() -> None:
    markdown = "# Limits\n" + "\n\n".join(f"Paragraph number {number}." for number in range(40))

    _, sections = section_document(_document(markdown), 35, 5)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(sections) > 1
    assert all(section.heading_path == ("Limits",) for section in sections)
    assert all(len(encoding.encode(section.markdown)) <= 35 for section in sections)
    assert [section.ordinal for section in sections] == list(range(len(sections)))
    assert sections[0].section_id == "doc_test:0000"


def test_table_content_remains_searchable() -> None:
    markdown = "# Models\n| Name | Type |\n|---|---|\n| gpt-test | chat |\n"

    _, sections = section_document(_document(markdown), 100, 10)

    assert "gpt-test" in sections[0].embedding_text
