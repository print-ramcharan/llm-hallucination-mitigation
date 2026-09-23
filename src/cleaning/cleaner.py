"""Document cleaning engine for mitigating context degradation.

Cleans extra whitespace, duplicate headers, page numbers, broken line breaks,
irrelevant symbols, and repeated footers while strictly preserving tables,
headings, code blocks, and essential semantic formatting.
"""

from __future__ import annotations

import re
from collections import Counter

from src.ingestion.models import DocumentElement


class DocumentCleaner:
    """Targeted, non-destructive document cleaner.

    Sanitizes noise that degrades LLM attention (headers, page numbers, broken wraps)
    while safeguarding tables, headings, code, and lists.
    """

    # Matches split hyphenated words across lines (e.g. 'degrad-\nation' -> 'degradation')
    HYPHENATED_LINEBREAK_REGEX = re.compile(r"(\b[A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,}\b)")

    # Standalone page number patterns: "Page 12", "12 of 45", "12/45", "— 12 —", "- 12 -", "[12]", solitary digits
    PAGE_NUMBER_REGEX = re.compile(
        r"^(?:"
        r"(?:page\s+)?\d+(?:\s*(?:of|\/)\s*\d+)?"
        r"|[-—–]\s*\d+\s*[-—–]"
        r"|\[\s*\d+\s*\]"
        r"|\d+"
        r")$",
        re.IGNORECASE,
    )

    # Irrelevant / corrupted symbols and control characters (excluding standard \t, \n, \r)
    IRRELEVANT_SYMBOLS_REGEX = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\u200b\u200c\u200d\ufeff]"
    )

    # Common running footer/header noise: copyright notices, confidentiality disclaimers
    REPEATED_DISCLAIMER_REGEX = re.compile(
        r"^(?:"
        r"all rights reserved\.?"
        r"|copyright\s+(?:©|\(c\))?\s*\d{4}.*"
        r"|confidential(?:\s+and\s+proprietary)?"
        r"|internal\s+use\s+only"
        r"|draft\s*[-—–]?\s*do\s+not\s+distribute"
        r")$",
        re.IGNORECASE,
    )

    # Table / structure detection
    MARKDOWN_TABLE_ROW_REGEX = re.compile(r"^\s*\|.*\|\s*$")
    MARKDOWN_HEADING_REGEX = re.compile(r"^\s*#{1,6}\s+\S+")
    LIST_ITEM_REGEX = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S+")

    def clean_elements(self, elements: list[DocumentElement]) -> list[DocumentElement]:
        """Clean a sequence of DocumentElements produced by format parsers.

        1. Identifies running headers and footers across multiple pages.
        2. Cleans each element's content individually.
        3. Removes empty or noise-only elements (e.g. isolated page numbers).
        4. Deduplicates consecutive identical headings.
        """
        if not elements:
            return []

        # Step 1: Detect recurring page headers/footers across pages
        running_headers, running_footers = self._detect_running_headers_footers(elements)

        cleaned_elements: list[DocumentElement] = []
        last_heading_content: str | None = None

        for el in elements:
            # Code and table elements are preserved with minimal symbol cleanup
            if el.element_type in ("code", "table"):
                cleaned_text = self.clean_text(
                    el.content,
                    preserve_tables=True,
                    preserve_code=True,
                    strip_page_numbers=False,
                )
                if cleaned_text.strip():
                    el_copy = el.model_copy(
                        update={
                            "content": cleaned_text,
                            "char_count": len(cleaned_text),
                            "word_count": len(cleaned_text.split()),
                        }
                    )
                    cleaned_elements.append(el_copy)
                continue

            raw_text = el.content
            if not raw_text or not raw_text.strip():
                continue

            # Check if this entire element is a standalone page number or running header/footer
            stripped_single = raw_text.strip()
            if self.PAGE_NUMBER_REGEX.match(stripped_single):
                continue
            if stripped_single.lower() in running_headers or stripped_single.lower() in running_footers:
                continue

            # Clean element text
            cleaned_text = self.clean_text(
                raw_text,
                preserve_tables=True,
                preserve_code=True,
                strip_page_numbers=True,
                running_headers=running_headers,
                running_footers=running_footers,
            )

            if not cleaned_text.strip():
                continue

            # Deduplicate consecutive identical headings
            if el.element_type == "heading":
                norm_heading = re.sub(r"\s+", " ", cleaned_text).strip().lower()
                if norm_heading == last_heading_content:
                    continue
                last_heading_content = norm_heading
            else:
                last_heading_content = None

            el_copy = el.model_copy(
                update={
                    "content": cleaned_text,
                    "char_count": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                }
            )
            cleaned_elements.append(el_copy)

        return cleaned_elements

    def clean_text(
        self,
        text: str,
        preserve_tables: bool = True,
        preserve_code: bool = True,
        strip_page_numbers: bool = True,
        running_headers: set[str] | None = None,
        running_footers: set[str] | None = None,
    ) -> str:
        """Sanitize raw document text with boundary protection.

        - Removes irrelevant symbols (control chars, zero-width spaces, BOM)
        - Fixes broken hyphenated words ('degra-\\ndation' -> 'degradation')
        - Merges soft-wrapped broken line breaks (preserving table rows & lists)
        - Removes page numbers and running headers/footers
        - Normalizes whitespace (collapsing 3+ newlines to 2)
        """
        if not text or not text.strip():
            return ""

        # 1. Remove irrelevant symbols and control characters
        text = self.IRRELEVANT_SYMBOLS_REGEX.sub("", text)

        # 2. Fix split hyphenated words across line breaks (e.g. 'atten-\ntion' -> 'attention')
        text = self.HYPHENATED_LINEBREAK_REGEX.sub(r"\1\2", text)

        # 3. Line-by-line processing respecting code blocks and tables
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Check for fenced code block toggle (```)
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
                continue

            # Inside code blocks, preserve line exactly
            if in_code_block:
                cleaned_lines.append(line)
                continue

            # Table rows: preserve columns and formatting exactly
            if preserve_tables and self.is_table_row(line):
                cleaned_lines.append(line)
                continue

            # Skip standalone page numbers if requested
            if strip_page_numbers and self.PAGE_NUMBER_REGEX.match(stripped):
                continue

            # Skip running headers/footers if identified
            if running_headers and stripped.lower() in running_headers:
                continue
            if running_footers and stripped.lower() in running_footers:
                continue

            # Skip generic repeated disclaimers if solitary
            if self.REPEATED_DISCLAIMER_REGEX.match(stripped):
                continue

            # Normal lines: clean trailing whitespace and collapse consecutive internal spaces
            norm_line = re.sub(r"[ \t]{2,}", " ", line.rstrip())
            cleaned_lines.append(norm_line)

        # 4. Repair soft-wrapped line breaks (prose unwrapping) without corrupting tables/headings/lists
        repaired_text = self._repair_soft_linebreaks(cleaned_lines)

        # 5. Normalize extra whitespace (collapse 3+ newlines to 2)
        final_text = re.sub(r"\n{3,}", "\n\n", repaired_text).strip()

        return final_text

    def is_table_row(self, line: str) -> bool:
        """Detect if a line is a Markdown table row or separator."""
        stripped = line.strip()
        if not stripped:
            return False
        if self.MARKDOWN_TABLE_ROW_REGEX.match(stripped):
            return True
        # Markdown table separator row like |:---|---:| or |---|---|
        if stripped.startswith("|") and re.match(r"^\|[\s\-:|]+\|$", stripped):
            return True
        return False

    def is_heading(self, line: str) -> bool:
        """Detect if a line is a Markdown heading."""
        return bool(self.MARKDOWN_HEADING_REGEX.match(line.strip()))

    def is_list_item(self, line: str) -> bool:
        """Detect if a line is a bullet or numbered list item."""
        return bool(self.LIST_ITEM_REGEX.match(line.strip()))

    def _repair_soft_linebreaks(self, lines: list[str]) -> str:
        """Merge lines broken mid-sentence by PDF/text extractors while preserving structure.

        Rules for NOT merging line A with line B:
        - Line A is empty, or Line B is empty
        - Line A or Line B is a code fence (```)
        - Line A or Line B is a table row (|)
        - Line A or Line B is a heading (#)
        - Line B is a list item (- / * / 1.)
        - Line A ends with terminal punctuation (. ? ! : ;) or dash
        - Line B starts with a capital letter and line A ends with a period/quote
        """
        if not lines:
            return ""

        merged: list[str] = []
        i = 0
        n = len(lines)

        while i < n:
            curr_line = lines[i]

            # If current line is empty, preserve it
            if not curr_line.strip():
                merged.append("")
                i += 1
                continue

            # Peek next line to see if it should merge
            while i + 1 < n:
                next_line = lines[i + 1]
                curr_strip = curr_line.strip()
                next_strip = next_line.strip()

                if not next_strip:
                    break

                # Boundary guards
                if (
                    curr_strip.startswith("```")
                    or next_strip.startswith("```")
                    or self.is_table_row(curr_line)
                    or self.is_table_row(next_line)
                    or self.is_heading(curr_line)
                    or self.is_heading(next_line)
                    or self.is_list_item(next_line)
                ):
                    break

                # If current line ends with sentence terminal punctuation, do not merge
                if curr_strip[-1] in (".", "!", "?", ":", ";", ">", "—"):
                    break

                # Soft wrap detected! Merge next line onto curr_line with a space
                curr_line = f"{curr_strip} {next_strip}"
                i += 1

            merged.append(curr_line)
            i += 1

        return "\n".join(merged)

    def _detect_running_headers_footers(
        self, elements: list[DocumentElement]
    ) -> tuple[set[str], set[str]]:
        """Identify candidate running headers and footers that repeat across pages."""
        page_headers: Counter[str] = Counter()
        page_footers: Counter[str] = Counter()
        pages_seen: set[int] = set()

        # Group elements by page
        page_elements: dict[int, list[DocumentElement]] = {}
        for el in elements:
            if el.page_number is not None:
                pages_seen.add(el.page_number)
                page_elements.setdefault(el.page_number, []).append(el)

        # Require at least 2 distinct pages to identify running headers/footers
        if len(pages_seen) < 2:
            return set(), set()

        for page_num, el_list in page_elements.items():
            if not el_list:
                continue

            # Top element on page (candidate header)
            first_lines = [line_str.strip().lower() for line_str in el_list[0].content.splitlines() if line_str.strip()]
            if first_lines and len(first_lines[0]) < 120:
                page_headers[first_lines[0]] += 1

            # Bottom element on page (candidate footer)
            last_lines = [line_str.strip().lower() for line_str in el_list[-1].content.splitlines() if line_str.strip()]
            if last_lines and len(last_lines[-1]) < 120:
                page_footers[last_lines[-1]] += 1

        # A phrase is a running header/footer if it appears on >= 50% of pages (or >= 2 pages)
        threshold = max(2, len(pages_seen) // 2)
        running_h = {text for text, count in page_headers.items() if count >= threshold}
        running_f = {text for text, count in page_footers.items() if count >= threshold}

        return running_h, running_f


default_cleaner = DocumentCleaner()
