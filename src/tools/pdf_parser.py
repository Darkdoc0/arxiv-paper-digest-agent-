"""Section-aware PDF parser using PyMuPDF (fitz) with robust error and scanned PDF fallbacks."""

import io
import re
import urllib.request
import logging
from typing import Dict, List, Optional
import pymupdf as fitz  # PyMuPDF
from src.state import ParsedDocument, PaperMetadata

logger = logging.getLogger(__name__)

# Standard academic section heading patterns
SECTION_PATTERNS = [
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?abstract\b", re.IGNORECASE), "Abstract"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?introduction\b", re.IGNORECASE), "Introduction"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?(?:related\s+work|background|prior\s+work)\b", re.IGNORECASE), "Background & Related Work"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?(?:methodology|method|model|approach|architecture|system\s+design)\b", re.IGNORECASE), "Methodology & Architecture"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?(?:experiments?|evaluation|results?|empirical\s+analysis)\b", re.IGNORECASE), "Experiments & Results"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?(?:discussion|limitations?|broader\s+impacts?)\b", re.IGNORECASE), "Limitations & Discussion"),
    (re.compile(r"^(?:(?:[1-9]\d*|[I|V|X]+)\.?\s+)?(?:conclusions?|future\s+work|summary)\b", re.IGNORECASE), "Conclusion"),
]


def download_pdf_bytes(pdf_url: str, timeout: int = 25) -> Optional[bytes]:
    """Download PDF bytes from URL with proper User-Agent header and timeout."""
    headers = {
        "User-Agent": "Autonomous-arXiv-Agent/1.0 (academic research tool; mailto:research@example.com)"
    }
    # Ensure URL uses https
    if pdf_url.startswith("http://"):
        pdf_url = "https://" + pdf_url[7:]
    # Ensure URL ends with .pdf if missing in some formats
    if not pdf_url.endswith(".pdf") and "arxiv.org/pdf/" in pdf_url:
        pdf_url = pdf_url + ".pdf"

    req = urllib.request.Request(pdf_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        logger.error(f"Failed to download PDF from {pdf_url}: {e}")
        return None


def parse_pdf_from_bytes(pdf_bytes: bytes, fallback_abstract: str = "") -> ParsedDocument:
    """Parse PDF bytes into structured sections, detecting headings and page content."""
    warnings: List[str] = []
    sections: Dict[str, str] = {}
    current_section = "Introduction"
    section_texts: Dict[str, List[str]] = {
        "Abstract": [],
        "Introduction": [],
        "Background & Related Work": [],
        "Methodology & Architecture": [],
        "Experiments & Results": [],
        "Limitations & Discussion": [],
        "Conclusion": [],
        "Other": []
    }
    fulltext_parts: List[str] = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        total_extracted_chars = 0

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            total_extracted_chars += len(text.strip())
            fulltext_parts.append(text)

            # Analyze lines for section headers
            lines = text.splitlines()
            for line in lines:
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Check if this line looks like a major section header (short line, matches regex)
                matched_section = None
                if len(clean_line) < 60:
                    for pattern, sec_name in SECTION_PATTERNS:
                        if pattern.match(clean_line):
                            matched_section = sec_name
                            break

                if matched_section:
                    current_section = matched_section
                else:
                    if current_section in section_texts:
                        section_texts[current_section].append(clean_line)
                    else:
                        section_texts["Other"].append(clean_line)

        # Scanned PDF check: if page count > 0 but total characters < 200, it's likely scanned images
        if page_count > 0 and total_extracted_chars < 200:
            warnings.append(
                "PDF appears to be a scanned image or contains non-extractable text. "
                "Engaging abstract-only fallback."
            )
            return ParsedDocument(
                abstract=fallback_abstract,
                sections={"Abstract": fallback_abstract},
                fulltext=fallback_abstract,
                page_count=page_count,
                is_abstract_fallback=True,
                warnings=warnings
            )

        # Consolidate sections
        for sec_name, lines in section_texts.items():
            content = " ".join(lines).strip()
            if content:
                sections[sec_name] = content

        # Ensure Abstract section exists
        if not sections.get("Abstract") and fallback_abstract:
            sections["Abstract"] = fallback_abstract

        fulltext = "\n\n".join(fulltext_parts).strip()

        return ParsedDocument(
            abstract=sections.get("Abstract", fallback_abstract),
            sections=sections,
            fulltext=fulltext,
            page_count=page_count,
            is_abstract_fallback=False,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"PyMuPDF error parsing PDF: {e}")
        warnings.append(f"PDF parsing error encountered ({str(e)}). Abstract-only fallback used.")
        return ParsedDocument(
            abstract=fallback_abstract,
            sections={"Abstract": fallback_abstract},
            fulltext=fallback_abstract,
            page_count=0,
            is_abstract_fallback=True,
            warnings=warnings
        )


def extract_paper_content(paper: PaperMetadata) -> ParsedDocument:
    """Download and extract content for an arXiv paper, returning ParsedDocument."""
    pdf_bytes = download_pdf_bytes(paper.pdf_url)
    if not pdf_bytes:
        logger.warning(f"Could not download PDF for {paper.arxiv_id}. Using abstract fallback.")
        return ParsedDocument(
            abstract=paper.summary,
            sections={"Abstract": paper.summary},
            fulltext=paper.summary,
            page_count=0,
            is_abstract_fallback=True,
            warnings=[f"Failed to download PDF from {paper.pdf_url}. Abstract used as sole context."]
        )

    return parse_pdf_from_bytes(pdf_bytes, fallback_abstract=paper.summary)
