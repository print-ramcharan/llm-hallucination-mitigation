"""Unit tests for the Document Cleaning Engine."""

from src.cleaning.cleaner import DocumentCleaner
from src.ingestion.models import DocumentElement


class TestDocumentCleaner:
    def setup_method(self):
        self.cleaner = DocumentCleaner()

    def test_strip_extra_whitespace(self):
        dirty = "This   has    excessive   spaces.\n\n\n\nAnd four blank lines."
        cleaned = self.cleaner.clean_text(dirty)
        assert "This has excessive spaces." in cleaned
        assert "\n\n\n" not in cleaned
        assert "spaces.\n\nAnd" in cleaned

    def test_fix_hyphenated_linebreaks(self):
        dirty = "Contextual degra-\ndation leads to hallu-\ncinations in LLMs."
        cleaned = self.cleaner.clean_text(dirty)
        assert "degradation" in cleaned
        assert "hallucinations" in cleaned
        assert "degra-" not in cleaned

    def test_repair_soft_linebreaks(self):
        dirty = "This is a sentence that was\naccidentally wrapped across two\nlines by an extractor."
        cleaned = self.cleaner.clean_text(dirty)
        assert cleaned == "This is a sentence that was accidentally wrapped across two lines by an extractor."

    def test_strip_page_numbers(self):
        samples = [
            "Page 12",
            "12 / 45",
            "12 of 45",
            "— 14 —",
            "- 15 -",
            "[ 16 ]",
            "42",
        ]
        for s in samples:
            assert self.cleaner.clean_text(s) == "", f"Failed to strip page number: {s}"

    def test_remove_irrelevant_symbols_and_control_chars(self):
        # Null bytes, form feed, zero-width space, unicode replacement character
        dirty = "Normal text\x00 with formfeed\x0c and zero-width\u200b space and bad \ufffd char."
        cleaned = self.cleaner.clean_text(dirty)
        assert "\x00" not in cleaned
        assert "\x0c" not in cleaned
        assert "\u200b" not in cleaned
        assert "\ufffd" not in cleaned
        assert "Normal text with formfeed and zero-width space and bad char." in cleaned

    def test_preserve_markdown_tables(self):
        table_text = (
            "| Model | Accuracy | F1 |\n"
            "|:---|:---:|---:|\n"
            "| GPT-4 | 92.4% | 0.91 |\n"
            "| Claude | 91.8% | 0.90 |"
        )
        cleaned = self.cleaner.clean_text(table_text)
        # Table structure, line breaks, and pipes must remain intact
        assert "| Model | Accuracy | F1 |" in cleaned
        assert "|:---|:---:|---:|" in cleaned
        assert "| GPT-4 | 92.4% | 0.91 |" in cleaned
        assert "| Claude | 91.8% | 0.90 |" in cleaned

    def test_preserve_code_blocks(self):
        code_text = (
            "```python\n"
            "def calculate(x, y):\n"
            "    # 4 spaces indentation\n"
            "    return x + y\n"
            "```"
        )
        cleaned = self.cleaner.clean_text(code_text)
        assert "def calculate(x, y):" in cleaned
        assert "    # 4 spaces indentation" in cleaned
        assert "    return x + y" in cleaned

    def test_preserve_headings_and_lists(self):
        text = (
            "# Main Heading\n\n"
            "## Subheading 1\n"
            "- First bullet point\n"
            "- Second bullet point\n\n"
            "1. Numbered item A\n"
            "2. Numbered item B"
        )
        cleaned = self.cleaner.clean_text(text)
        assert "# Main Heading" in cleaned
        assert "## Subheading 1" in cleaned
        assert "- First bullet point" in cleaned
        assert "- Second bullet point" in cleaned
        assert "1. Numbered item A" in cleaned
        assert "2. Numbered item B" in cleaned

    def test_deduplicate_consecutive_headings(self):
        elements = [
            DocumentElement(id="el_0", element_type="heading", content="Leave Policy"),
            DocumentElement(id="el_1", element_type="heading", content="Leave Policy"),
            DocumentElement(id="el_2", element_type="paragraph", content="Employees receive 20 days off."),
        ]
        cleaned_els = self.cleaner.clean_elements(elements)
        assert len(cleaned_els) == 2
        assert cleaned_els[0].content == "Leave Policy"
        assert cleaned_els[1].content == "Employees receive 20 days off."

    def test_remove_repeated_running_footers_and_headers(self):
        # 3-page document where each page has "Confidential - Company Internal" at the bottom
        elements = [
            DocumentElement(id="el_0", element_type="paragraph", content="Page 1 content here.", page_number=1),
            DocumentElement(id="el_1", element_type="paragraph", content="Confidential - Company Internal", page_number=1),
            DocumentElement(id="el_2", element_type="paragraph", content="Page 2 content here.", page_number=2),
            DocumentElement(id="el_3", element_type="paragraph", content="Confidential - Company Internal", page_number=2),
            DocumentElement(id="el_4", element_type="paragraph", content="Page 3 content here.", page_number=3),
            DocumentElement(id="el_5", element_type="paragraph", content="Confidential - Company Internal", page_number=3),
        ]
        cleaned_els = self.cleaner.clean_elements(elements)
        # Running footer should be stripped across all pages
        assert len(cleaned_els) == 3
        for el in cleaned_els:
            assert "Confidential" not in el.content
            assert "content here" in el.content
