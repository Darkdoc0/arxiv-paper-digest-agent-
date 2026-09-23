"""Node: Query parsing and intent classification."""

import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.arxiv_client import extract_arxiv_id
from src.tools.llm import get_llm

logger = logging.getLogger(__name__)


def parse_query_node(state: AgentState) -> Dict[str, Any]:
    """Classify user query intent into direct_id vs. topic_search and extract search parameters."""
    raw_query = state.get("query", "").strip()
    logger.info(f"parse_query_node evaluating query: '{raw_query}'")

    # Step 1: Check for explicit arXiv ID or URL
    extracted_id = extract_arxiv_id(raw_query)
    if extracted_id:
        logger.info(f"Detected direct arXiv ID: {extracted_id}")
        return {
            "intent": "direct_id",
            "arxiv_id": extracted_id,
            "search_query": None,
            "error": None,
            "retry_count": 0
        }

    # Step 2: Fall back to topic search. Extract clean keywords using LLM or heuristic
    logger.info("No direct arXiv ID detected. Treating as topic search.")
    search_query = raw_query

    # If the user query is conversational (e.g. "Can you find me papers on speculative decoding for transformers?"),
    # use the LLM to extract concise keywords suitable for arXiv's search API.
    if len(raw_query.split()) > 4:
        try:
            llm = get_llm(temperature=0.0)
            prompt = (
                "You are an academic search query optimizer for arXiv. "
                "Given a conversational user request, extract 2 to 5 essential academic keywords "
                "or search phrases for arXiv's title/abstract search. Do not include boolean operators, "
                "quotes, or punctuation. Output ONLY the keywords separated by spaces.\n\n"
                f"User request: {raw_query}\n"
                "Keywords:"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            extracted_keywords = response.content.strip().replace('"', '').replace("'", "")
            if extracted_keywords:
                search_query = extracted_keywords
                logger.info(f"Refined search query via LLM: '{search_query}'")
        except Exception as e:
            logger.warning(f"LLM query refinement failed ({e}), using raw query.")

    return {
        "intent": "topic_search",
        "arxiv_id": None,
        "search_query": search_query,
        "error": None,
        "retry_count": 0
    }
