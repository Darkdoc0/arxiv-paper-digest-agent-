"""Nodes: ArXiv retrieval and candidate paper ranking/selection."""

import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from src.state import AgentState, PaperMetadata
from src.tools.arxiv_client import fetch_paper_by_id, search_papers
from src.tools.llm import get_llm, extract_text_content

logger = logging.getLogger(__name__)


class SelectionDecision(BaseModel):
    """Structured output for paper ranking decision."""
    selected_arxiv_id: str = Field(description="The exact arXiv ID of the most relevant candidate paper")
    reasoning: str = Field(description="Crisp rationale explaining why this paper was selected over other candidates")


def arxiv_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve papers from arXiv API based on direct ID or topic search."""
    intent = state.get("intent")
    arxiv_id = state.get("arxiv_id")
    search_query = state.get("search_query") or state.get("query", "")
    retry_count = state.get("retry_count", 0)

    logger.info(f"arxiv_retrieval_node started (intent={intent}, retry={retry_count})")

    if intent == "direct_id" and arxiv_id:
        paper = fetch_paper_by_id(arxiv_id)
        if paper:
            return {
                "candidate_papers": [paper],
                "selected_paper": paper,
                "error": None
            }
        else:
            return {
                "candidate_papers": [],
                "selected_paper": None,
                "error": f"Could not find arXiv paper matching ID '{arxiv_id}'. Please check the ID or URL."
            }

    # Otherwise topic search
    candidates = search_papers(search_query, max_results=5)

    # Edge Case: Zero papers returned
    if not candidates:
        if retry_count < 2:
            logger.warning(f"0 papers found for '{search_query}'. Attempting LLM query reformulation (retry {retry_count + 1})...")
            try:
                llm = get_llm(temperature=0.3)
                reformulate_prompt = (
                    "The following arXiv search query returned zero results:\n"
                    f"'{search_query}'\n\n"
                    "Please reformulate this into a broader, more standard academic search term "
                    "(e.g., replace niche acronyms with standard terminology, remove restrictive words). "
                    "Output ONLY the revised search query (1-4 words), nothing else."
                )
                res = llm.invoke([HumanMessage(content=reformulate_prompt)])
                raw_res = extract_text_content(res.content)
                new_query = raw_res.strip().replace('"', '').replace("'", "")
                logger.info(f"Reformulated query: '{new_query}'")
                
                # Immediately retry with the new query
                retried_candidates = search_papers(new_query, max_results=5)
                if retried_candidates:
                    return {
                        "search_query": new_query,
                        "candidate_papers": retried_candidates,
                        "retry_count": retry_count + 1,
                        "error": None
                    }
                else:
                    return {
                        "search_query": new_query,
                        "candidate_papers": [],
                        "retry_count": retry_count + 1,
                        "error": f"Zero papers found after query reformulation ('{new_query}')."
                    }
            except Exception as e:
                logger.error(f"Error during query reformulation: {e}")

        return {
            "candidate_papers": [],
            "selected_paper": None,
            "error": f"Zero papers found for query '{search_query}'. Please try broader search keywords."
        }

    return {
        "candidate_papers": candidates,
        "error": None
    }


def rank_and_select_node(state: AgentState) -> Dict[str, Any]:
    """Rank candidate papers using LLM and select the most relevant one."""
    # If already selected (e.g. direct_id), return
    if state.get("selected_paper"):
        return {"selected_paper": state["selected_paper"], "selection_reasoning": "Direct ID match."}

    candidates = state.get("candidate_papers", [])
    if not candidates:
        return {
            "selected_paper": None,
            "error": "No candidate papers available to rank."
        }

    if len(candidates) == 1:
        return {
            "selected_paper": candidates[0],
            "selection_reasoning": "Single relevant candidate returned by search."
        }

    user_query = state.get("query", "")
    logger.info(f"Ranking {len(candidates)} candidate papers for query: '{user_query}'")

    # Format candidates for ranking
    candidates_text = []
    for i, p in enumerate(candidates, 1):
        candidates_text.append(
            f"Candidate [{i}] arXiv ID: {p.arxiv_id}\n"
            f"Title: {p.title}\n"
            f"Authors: {', '.join(p.authors[:4])}\n"
            f"Published: {p.published}\n"
            f"Abstract: {p.summary[:400]}...\n"
        )
    candidate_block = "\n".join(candidates_text)

    prompt = (
        f"User Research Query: \"{user_query}\"\n\n"
        f"Available Candidate Papers from arXiv:\n{candidate_block}\n\n"
        "Evaluate the candidate papers and select the single best paper based on:\n"
        "1. Direct semantic alignment with the user's specific request.\n"
        "2. Foundational methodological significance vs. tangential application.\n"
        "3. Recency and technical relevance.\n\n"
        "Return the chosen paper's arXiv ID and your concise selection reasoning."
    )

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(SelectionDecision)
        decision = structured_llm.invoke([
            SystemMessage(content="You are an expert AI researcher evaluating scientific papers."),
            HumanMessage(content=prompt)
        ])

        # Find matching candidate
        selected_id = decision.selected_arxiv_id.strip()
        matched_paper = next((p for p in candidates if selected_id in p.arxiv_id or p.arxiv_id in selected_id), None)

        if not matched_paper:
            # Fallback to the first candidate if ID parsing mismatch
            matched_paper = candidates[0]
            logger.warning(f"Could not cleanly match returned ID '{selected_id}'. Falling back to first candidate: {matched_paper.arxiv_id}")

        return {
            "selected_paper": matched_paper,
            "selection_reasoning": decision.reasoning
        }

    except Exception as e:
        logger.error(f"Error during candidate ranking: {e}. Defaulting to top candidate.")
        return {
            "selected_paper": candidates[0],
            "selection_reasoning": f"Defaulted to highest-ranking search match due to ranking model fallback ({e})."
        }
