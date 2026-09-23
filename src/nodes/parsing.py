"""Node: Fetch and parse paper PDF with robust fallback handling."""

import logging
from typing import Dict, Any
from src.state import AgentState, ParsedDocument
from src.tools.pdf_parser import extract_paper_content

logger = logging.getLogger(__name__)


def fetch_and_parse_pdf_node(state: AgentState) -> Dict[str, Any]:
    """Download and extract structured section content from the selected paper's PDF.

    Gracefully falls back to abstract-only extraction if the PDF is corrupt,
    inaccessible, encrypted, or a non-OCR scanned image.
    """
    selected_paper = state.get("selected_paper")
    if not selected_paper:
        return {
            "parsed_paper": None,
            "error": "No paper selected for PDF parsing."
        }

    logger.info(f"Downloading and parsing PDF for arXiv paper: {selected_paper.arxiv_id} ({selected_paper.pdf_url})")

    try:
        parsed_doc: ParsedDocument = extract_paper_content(selected_paper)

        if parsed_doc.is_abstract_fallback:
            logger.warning(
                f"PDF parsing for {selected_paper.arxiv_id} triggered fallback mode: "
                f"{'; '.join(parsed_doc.warnings)}"
            )
            return {
                "parsed_paper": parsed_doc,
                "is_fallback_abstract": True,
                "error": None
            }

        logger.info(
            f"Successfully parsed PDF for {selected_paper.arxiv_id}: "
            f"{parsed_doc.page_count} pages, sections: {list(parsed_doc.sections.keys())}"
        )
        return {
            "parsed_paper": parsed_doc,
            "is_fallback_abstract": False,
            "error": None
        }

    except Exception as e:
        logger.error(f"Unexpected exception during PDF extraction for {selected_paper.arxiv_id}: {e}")
        # Build safe fallback document
        fallback_doc = ParsedDocument(
            abstract=selected_paper.summary,
            sections={"Abstract": selected_paper.summary},
            fulltext=selected_paper.summary,
            page_count=0,
            is_abstract_fallback=True,
            warnings=[f"Extraction exception ({str(e)}). Fallen back to abstract analysis."]
        )
        return {
            "parsed_paper": fallback_doc,
            "is_fallback_abstract": True,
            "error": None
        }
