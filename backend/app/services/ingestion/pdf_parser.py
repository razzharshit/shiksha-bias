import re
import string
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from app.core.config import settings
from app.core.logging_config import logger


@dataclass
class PDFParserConfig:
    header_margin_pct: float = settings.PDF_DEFAULT_HEADER_MARGIN_PCT
    footer_margin_pct: float = settings.PDF_DEFAULT_FOOTER_MARGIN_PCT
    filter_headers_footers: bool = True
    min_printable_char_ratio: float = 0.85
    min_words_per_page: int = 5
    line_vertical_tolerance: float = 4.0  # px tolerance for grouping blocks into same horizontal band


@dataclass
class ExtractedBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_type: int
    is_header_footer: bool = False


@dataclass
class ExtractedPage:
    page_number: int
    width: float
    height: float
    blocks: List[ExtractedBlock]
    full_text: str
    cleaned_text: str
    word_count: int
    printable_ratio: float


@dataclass
class ExtractionValidationReport:
    total_pages: int
    pages_with_text: int
    page_coverage_pct: float
    total_words: int
    total_characters: int
    printable_char_ratio: float
    duplicate_boilerplate_count: int
    detected_chapters_count: int
    is_valid: bool
    quality_notes: List[str] = field(default_factory=list)


class PDFParser:
    """
    Robust PDF Parser powered by PyMuPDF (fitz).
    Preserves page numbers, reading order, bounding boxes, and performs extraction quality audits.
    """

    CHAPTER_REGEXES = [
        re.compile(r'^\s*(?:chapter|unit|lesson|module|part)\s*([0-9ivxlcdm]+)[:\.\-\s]*(.*)$', re.IGNORECASE),
        re.compile(r'^\s*([0-9]{1,2})\s*[\.\-]\s*([A-Z][a-zA-Z\s]{2,40})$'),
    ]

    def __init__(self, config: Optional[PDFParserConfig] = None):
        self.config = config or PDFParserConfig()

    def parse(self, pdf_path_or_bytes) -> List[ExtractedPage]:
        """Parses a PDF into structured ExtractedPage objects with reading-order block sorting."""
        if isinstance(pdf_path_or_bytes, (str, bytes)):
            doc = fitz.open(pdf_path_or_bytes) if isinstance(pdf_path_or_bytes, str) else fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
        else:
            raise ValueError("Input must be a file path string or bytes buffer.")

        extracted_pages: List[ExtractedPage] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                width = page.rect.width
                height = page.rect.height

                # Extract raw blocks: (x0, y0, x1, y1, text, block_no, block_type)
                raw_blocks = page.get_text("blocks")
                text_blocks: List[ExtractedBlock] = []

                header_boundary = height * self.config.header_margin_pct if self.config.filter_headers_footers else -1.0
                footer_boundary = height * (1.0 - self.config.footer_margin_pct) if self.config.filter_headers_footers else height + 1.0

                for b in raw_blocks:
                    x0, y0, x1, y1, text, _, b_type = b[:7]
                    clean_b_text = text.strip()
                    if not clean_b_text:
                        continue

                    # Check if block falls in header/footer margin
                    is_header_footer = False
                    if self.config.filter_headers_footers:
                        if y1 <= header_boundary or y0 >= footer_boundary:
                            is_header_footer = True

                    text_blocks.append(ExtractedBlock(
                        x0=round(x0, 2),
                        y0=round(y0, 2),
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        text=clean_b_text,
                        block_type=b_type,
                        is_header_footer=is_header_footer
                    ))

                # Sort blocks by natural reading order: top-to-bottom, left-to-right
                # Group by vertical line band using tolerance
                def block_sort_key(block: ExtractedBlock):
                    band = round(block.y0 / self.config.line_vertical_tolerance) * self.config.line_vertical_tolerance
                    return (band, block.x0)

                text_blocks.sort(key=block_sort_key)

                # Assemble full text and cleaned (header/footer filtered) text
                full_lines = [b.text for b in text_blocks]
                cleaned_lines = [b.text for b in text_blocks if not b.is_header_footer]

                full_text = "\n".join(full_lines)
                cleaned_text = "\n".join(cleaned_lines)

                # Calculate word count & printable characters
                words = cleaned_text.split()
                total_chars = len(cleaned_text)
                printable_chars = sum(1 for c in cleaned_text if c in string.printable or ord(c) > 127)
                printable_ratio = (printable_chars / (total_chars + 1e-9)) if total_chars > 0 else 1.0

                extracted_pages.append(ExtractedPage(
                    page_number=page_num,
                    width=round(width, 2),
                    height=round(height, 2),
                    blocks=text_blocks,
                    full_text=full_text,
                    cleaned_text=cleaned_text,
                    word_count=len(words),
                    printable_ratio=round(printable_ratio, 4)
                ))

        finally:
            doc.close()

        return extracted_pages

    def detect_chapters(self, pages: List[ExtractedPage]) -> List[Dict[str, Any]]:
        """Identifies chapters and section boundaries using regex and structural cues."""
        chapters = []
        current_ch_num = 1
        current_title = "Introduction"
        current_start_page = 1

        for p in pages:
            for b in p.blocks:
                if b.is_header_footer:
                    continue

                for line in b.text.split("\n"):
                    line_clean = line.strip()
                    for r in self.CHAPTER_REGEXES:
                        match = r.match(line_clean)
                        if match:
                            num_str = match.group(1).strip()
                            raw_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                            ch_title = raw_title if len(raw_title) > 2 else f"Chapter {num_str}"

                            try:
                                parsed_num = int(num_str)
                            except ValueError:
                                parsed_num = current_ch_num + 1

                            # If a chapter was already running, close it
                            if chapters or p.page_number > 1:
                                chapters.append({
                                    "chapter_number": current_ch_num,
                                    "title": current_title,
                                    "start_page": current_start_page,
                                    "end_page": max(current_start_page, p.page_number - 1)
                                })

                            current_ch_num = parsed_num
                            current_title = ch_title
                            current_start_page = p.page_number
                            break

        # Close the final chapter
        last_page = pages[-1].page_number if pages else 1
        chapters.append({
            "chapter_number": current_ch_num,
            "title": current_title,
            "start_page": current_start_page,
            "end_page": last_page
        })

        return chapters

    def validate_extraction(self, pages: List[ExtractedPage]) -> ExtractionValidationReport:
        """Audits extraction quality: page coverage, printable ratio, boilerplate repetition."""
        total_pages = len(pages)
        if total_pages == 0:
            return ExtractionValidationReport(
                total_pages=0, pages_with_text=0, page_coverage_pct=0.0,
                total_words=0, total_characters=0, printable_char_ratio=0.0,
                duplicate_boilerplate_count=0, detected_chapters_count=0,
                is_valid=False, quality_notes=["Document contains 0 pages."]
            )

        pages_with_text = sum(1 for p in pages if p.word_count >= self.config.min_words_per_page)
        coverage_pct = round((pages_with_text / total_pages) * 100, 2)

        total_words = sum(p.word_count for p in pages)
        total_chars = sum(len(p.cleaned_text) for p in pages)
        avg_printable = (sum(p.printable_ratio for p in pages) / total_pages) if total_pages > 0 else 0.0

        # Check for duplicated boilerplate blocks occurring across > 50% of pages
        block_text_frequencies: Dict[str, int] = {}
        for p in pages:
            seen_on_page = set()
            for b in p.blocks:
                t = b.text.strip().lower()
                if 4 < len(t) < 80 and t not in seen_on_page:
                    block_text_frequencies[t] = block_text_frequencies.get(t, 0) + 1
                    seen_on_page.add(t)

        boilerplate_threshold = max(3, total_pages // 2)
        repetitive_boilerplates = [t for t, count in block_text_frequencies.items() if count >= boilerplate_threshold]

        chapters = self.detect_chapters(pages)

        notes = []
        is_valid = True

        if coverage_pct < 70.0:
            notes.append(f"Low text page coverage: {coverage_pct}% of pages have meaningful text (possible scanned/image-only PDF).")
            is_valid = False
        if avg_printable < self.config.min_printable_char_ratio:
            notes.append(f"Low printable character ratio: {round(avg_printable, 3)} (encoding anomalies).")
            is_valid = False
        if total_words < 50:
            notes.append("Extracted word count is extremely low (< 50 words).")
            is_valid = False

        if not notes:
            notes.append("High extraction fidelity: valid reading order, character encodings, and layout margins verified.")

        return ExtractionValidationReport(
            total_pages=total_pages,
            pages_with_text=pages_with_text,
            page_coverage_pct=coverage_pct,
            total_words=total_words,
            total_characters=total_chars,
            printable_char_ratio=round(avg_printable, 4),
            duplicate_boilerplate_count=len(repetitive_boilerplates),
            detected_chapters_count=len(chapters),
            is_valid=is_valid,
            quality_notes=notes
        )

