"""Markdown document parser preserving headings and structural sections."""

from __future__ import annotations

import re
from typing import Any

from src.ingestion.base import BaseParser
from src.ingestion.models import Document, DocumentElement, DocumentType


class MarkdownParser(BaseParser):
    """Parses Markdown (.md) documents with hierarchical section extraction."""

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.MD]

    def can_parse(self, extension: str, mime_type: str | None = None) -> bool:
        ext = extension.lower().lstrip(".")
        if ext in ("md", "markdown", "mdown", "mkd"):
            return True
        if mime_type and "markdown" in mime_type.lower():
            return True
        return False

    def parse_bytes(
        self,
        content: bytes,
        filename: str = "document.md",
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Parse Markdown bytes into structured elements and common Document."""
        if not content:
            raise ValueError("Cannot parse empty markdown byte stream")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        extra_metadata: dict[str, Any] = {}

        # Check for YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", normalized, re.DOTALL)
        body_text = normalized
        if frontmatter_match:
            raw_fm = frontmatter_match.group(1)
            body_text = normalized[frontmatter_match.end() :]
            # Simple YAML key-value extraction without requiring pyyaml
            for line in raw_fm.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    extra_metadata[key.strip()] = val.strip().strip("'\"")

        elements: list[DocumentElement] = []
        lines = body_text.split("\n")

        current_heading: str | None = None
        current_block: list[str] = []
        in_code_block = False
        code_lang = ""
        el_idx = 0

        def flush_current_block(element_type: str = "paragraph", extra_meta: dict | None = None):
            nonlocal el_idx, current_block
            block_content = "\n".join(current_block).strip()
            if block_content:
                meta = extra_meta.copy() if extra_meta else {}
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type=element_type,
                        content=block_content,
                        section_title=current_heading,
                        metadata=meta,
                    )
                )
                el_idx += 1
            current_block = []

        for line in lines:
            # Check for code fence
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    current_block.append(line)
                    flush_current_block(element_type="code", extra_meta={"language": code_lang})
                    in_code_block = False
                    code_lang = ""
                else:
                    # Flush previous paragraph if any
                    flush_current_block()
                    in_code_block = True
                    code_lang = line.lstrip("`").strip()
                    current_block.append(line)
                continue

            if in_code_block:
                current_block.append(line)
                continue

            # Heading detection (# H1, ## H2, etc.)
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                flush_current_block()
                level = len(heading_match.group(1))
                heading_title = heading_match.group(2).strip()
                current_heading = heading_title
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type="heading",
                        content=line.strip(),
                        section_title=current_heading,
                        metadata={"heading_level": level, "title": heading_title},
                    )
                )
                el_idx += 1
                continue

            # Blank lines delineate paragraphs
            if not line.strip():
                flush_current_block()
                continue

            current_block.append(line)

        # Flush any trailing block
        flush_current_block(element_type="code" if in_code_block else "paragraph")

        if kwargs.get("document_type"):
            extra_metadata["document_type"] = kwargs["document_type"]
        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            extra_metadata.update(kwargs["extra_metadata"])

        return Document.create(
            source_name=filename,
            file_type=DocumentType.MD,
            elements=elements,
            file_size_bytes=file_size_bytes or len(content),
            extra_metadata=extra_metadata,
            raw_content=body_text.strip(),
        )
