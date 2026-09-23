"""HTML document parser using BeautifulSoup."""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from src.ingestion.base import BaseParser
from src.ingestion.models import Document, DocumentElement, DocumentType


class HtmlParser(BaseParser):
    """Parses HTML (.html, .htm) files, stripping boilerplate and extracting clean content."""

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.HTML]

    def can_parse(self, extension: str, mime_type: str | None = None) -> bool:
        ext = extension.lower().lstrip(".")
        if ext in ("html", "htm", "xhtml"):
            return True
        if mime_type and "text/html" in mime_type.lower():
            return True
        return False

    def parse_bytes(
        self,
        content: bytes,
        filename: str = "document.html",
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Parse HTML bytes, strip boilerplate, and produce structured Document."""
        if not content:
            raise ValueError("Cannot parse empty HTML byte stream")

        # Decode content
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                html_str = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            html_str = content.decode("utf-8", errors="replace")

        soup = BeautifulSoup(html_str, "html.parser")

        # Extract title and meta
        extra_metadata: dict[str, Any] = {}
        if soup.title and soup.title.string:
            extra_metadata["title"] = soup.title.string.strip()

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            extra_metadata["description"] = meta_desc["content"].strip()

        # Remove clutter, scripts, styles, navigation, footer
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "svg"]):
            tag.decompose()

        elements: list[DocumentElement] = []
        full_text_blocks: list[str] = []
        current_heading: str | None = None
        el_idx = 0

        # Find body or fallback to entire soup
        root = soup.body if soup.body else soup

        # Extract structured content tags in document order
        content_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre", "blockquote"]
        for element in root.find_all(content_tags):
            tag_name = element.name.lower()
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            if tag_name.startswith("h"):
                current_heading = text
                el_type = "heading"
                level = int(tag_name[1])
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type=el_type,
                        content=text,
                        section_title=current_heading,
                        metadata={"tag": tag_name, "level": level},
                    )
                )
            elif tag_name == "table":
                el_type = "table"
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type=el_type,
                        content=text,
                        section_title=current_heading,
                        metadata={"tag": tag_name},
                    )
                )
            elif tag_name == "pre":
                el_type = "code"
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type=el_type,
                        content=text,
                        section_title=current_heading,
                        metadata={"tag": tag_name},
                    )
                )
            else:
                el_type = "paragraph"
                elements.append(
                    DocumentElement(
                        id=f"temp_el_{el_idx}",
                        element_type=el_type,
                        content=text,
                        section_title=current_heading,
                        metadata={"tag": tag_name},
                    )
                )

            full_text_blocks.append(text)
            el_idx += 1

        # Fallback if no tags were matched
        if not elements:
            fallback_text = root.get_text(separator="\n\n", strip=True)
            if fallback_text:
                full_text_blocks.append(fallback_text)
                elements.append(
                    DocumentElement(
                        id="temp_el_0",
                        element_type="paragraph",
                        content=fallback_text,
                        metadata={"fallback": True},
                    )
                )

        full_content = "\n\n".join(full_text_blocks)

        if kwargs.get("document_type"):
            extra_metadata["document_type"] = kwargs["document_type"]
        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            extra_metadata.update(kwargs["extra_metadata"])

        return Document.create(
            source_name=filename,
            file_type=DocumentType.HTML,
            elements=elements,
            file_size_bytes=file_size_bytes or len(content),
            extra_metadata=extra_metadata,
            raw_content=full_content,
        )
