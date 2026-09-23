"""ArXiv API client wrapper with robust ID normalization and error handling."""

import re
import logging
from typing import List, Optional
import arxiv
from src.state import PaperMetadata

logger = logging.getLogger(__name__)

ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(?P<id>(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z\-]+(?:\.[a-zA-Z]+)?/\d{7}(?:v\d+)?))",
    re.IGNORECASE
)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract and normalize an arXiv ID from a URL or raw string."""
    text = text.strip()
    match = ARXIV_ID_PATTERN.search(text)
    if match:
        raw_id = match.group("id")
        # Strip version suffixes like 'v1', 'v2' for canonical querying if desired,
        # but keep it if the user specified a precise version
        return raw_id
    return None


def fetch_paper_by_id(arxiv_id: str, client: Optional[arxiv.Client] = None) -> Optional[PaperMetadata]:
    """Retrieve a single paper by its arXiv ID."""
    clean_id = extract_arxiv_id(arxiv_id) or arxiv_id.strip()
    # Strip any trailing '.pdf' if present
    clean_id = re.sub(r"\.pdf$", "", clean_id, flags=re.IGNORECASE)

    if client is None:
        client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)

    search = arxiv.Search(id_list=[clean_id])
    try:
        results = list(client.results(search))
        if not results:
            logger.warning(f"No paper found for arXiv ID: {clean_id}")
            return None
        res = results[0]
        return _convert_arxiv_result(res)
    except Exception as e:
        logger.error(f"Error fetching paper with ID {clean_id}: {e}")
        return None


def search_papers(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    client: Optional[arxiv.Client] = None
) -> List[PaperMetadata]:
    """Search arXiv by topic or free-form query and return PaperMetadata list."""
    if client is None:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)

    # Sanitize query: arXiv query syntax can fail if double quotes are unbalanced or special chars exist
    sanitized_query = re.sub(r'[\r\n\t]+', ' ', query).strip()
    
    search = arxiv.Search(
        query=sanitized_query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=arxiv.SortOrder.Descending
    )

    try:
        results = list(client.results(search))
        papers = [_convert_arxiv_result(r) for r in results]
        return papers
    except Exception as e:
        logger.error(f"Error executing arXiv search for query '{query}': {e}")
        return []


def _convert_arxiv_result(res: arxiv.Result) -> PaperMetadata:
    """Helper to convert an arxiv.Result object into our Pydantic PaperMetadata."""
    clean_id = extract_arxiv_id(res.entry_id) or res.get_short_id()
    authors = [a.name for a in res.authors]
    pub_date = res.published.strftime("%Y-%m-%d") if res.published else "Unknown"

    return PaperMetadata(
        title=res.title.replace("\n", " ").strip(),
        authors=authors,
        arxiv_id=clean_id,
        published=pub_date,
        summary=res.summary.replace("\n", " ").strip(),
        pdf_url=res.pdf_url,
        primary_category=res.primary_category,
        comment=res.comment,
        doi=res.doi
    )
