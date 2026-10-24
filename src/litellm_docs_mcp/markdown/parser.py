"""Parse Markdown into sections while preserving the original source text."""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Markdown grouped under one heading path."""

    heading_path: tuple[str, ...]
    markdown: str


@dataclass(frozen=True, slots=True)
class ParsedMarkdown:
    """The useful structure extracted from a Markdown document."""

    title: str | None
    sections: tuple[ParsedSection, ...]


def parse_markdown(markdown: str) -> ParsedMarkdown:
    """Extract front matter, headings, and raw heading-aware sections."""

    front_matter, body = _split_front_matter(markdown)
    body_lines = body.splitlines(keepends=True)
    tokens = _markdown_parser().parse(body)
    headings = _extract_headings(tokens)
    title = _front_matter_title(front_matter) or _first_level_one_heading(headings)

    sections: list[ParsedSection] = []
    if headings and headings[0][0] > 0:
        preamble = "".join(body_lines[: headings[0][0]])
        if preamble.strip():
            sections.append(ParsedSection(heading_path=(), markdown=preamble))

    heading_stack: list[tuple[int, str]] = []
    for index, (start_line, level, heading) in enumerate(headings):
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))
        end_line = headings[index + 1][0] if index + 1 < len(headings) else len(body_lines)
        sections.append(
            ParsedSection(
                heading_path=tuple(text for _, text in heading_stack),
                markdown="".join(body_lines[start_line:end_line]),
            )
        )

    if not sections:
        sections.append(ParsedSection(heading_path=(), markdown=body))

    return ParsedMarkdown(title=title, sections=tuple(sections))


def _markdown_parser() -> MarkdownIt:
    return MarkdownIt("commonmark").enable("table")


def _split_front_matter(markdown: str) -> tuple[object, str]:
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return None, markdown

    for end_index in range(1, len(lines)):
        if lines[end_index].strip() not in {"---", "..."}:
            continue
        yaml_text = "".join(lines[1:end_index])
        try:
            metadata = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            metadata = None
        return metadata, "".join(lines[end_index + 1 :])
    return None, markdown


def _extract_headings(tokens: list[Token]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        inline = tokens[index + 1]
        headings.append((token.map[0], int(token.tag[1:]), _inline_text(inline)))
    return headings


def _inline_text(token: Token) -> str:
    if not token.children:
        return token.content.strip()
    parts: list[str] = []
    for child in token.children:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts).strip()


def _front_matter_title(front_matter: object) -> str | None:
    if not isinstance(front_matter, dict):
        return None
    title = front_matter.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return title.strip()


def _first_level_one_heading(headings: list[tuple[int, int, str]]) -> str | None:
    return next((text for _, level, text in headings if level == 1 and text), None)
