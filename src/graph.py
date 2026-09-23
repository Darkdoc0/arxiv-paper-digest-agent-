"""LangGraph StateGraph compilation for Autonomous arXiv Agent."""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END, START
from src.state import AgentState
from src.nodes.query import parse_query_node
from src.nodes.retrieval import arxiv_retrieval_node, rank_and_select_node
from src.nodes.parsing import fetch_and_parse_pdf_node
from src.nodes.briefing import chunk_and_embed_node, summarize_briefing_node
from src.nodes.rag import qa_node

logger = logging.getLogger(__name__)


def route_after_retrieval(state: AgentState) -> Literal["rank_and_select", "__end__"]:
    """Conditional edge routing after arXiv retrieval."""
    if state.get("error") or not state.get("candidate_papers"):
        logger.warning(f"Routing to END due to retrieval failure: {state.get('error')}")
        return "__end__"
    return "rank_and_select"


def route_after_pdf_parse(state: AgentState) -> Literal["chunk_and_embed", "__end__"]:
    """Conditional edge routing after PDF parsing."""
    if state.get("error") or not state.get("parsed_paper"):
        logger.warning("Routing to END due to PDF parsing failure.")
        return "__end__"
    return "chunk_and_embed"


def build_ingestion_graph() -> StateGraph:
    """Build the primary paper ingestion, parsing, embedding, and executive briefing graph."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("arxiv_retrieval", arxiv_retrieval_node)
    workflow.add_node("rank_and_select", rank_and_select_node)
    workflow.add_node("fetch_and_parse_pdf", fetch_and_parse_pdf_node)
    workflow.add_node("chunk_and_embed", chunk_and_embed_node)
    workflow.add_node("summarize_briefing", summarize_briefing_node)

    # Edges
    workflow.add_edge(START, "parse_query")
    workflow.add_edge("parse_query", "arxiv_retrieval")
    
    workflow.add_conditional_edges(
        "arxiv_retrieval",
        route_after_retrieval,
        {
            "rank_and_select": "rank_and_select",
            "__end__": END
        }
    )

    workflow.add_edge("rank_and_select", "fetch_and_parse_pdf")

    workflow.add_conditional_edges(
        "fetch_and_parse_pdf",
        route_after_pdf_parse,
        {
            "chunk_and_embed": "chunk_and_embed",
            "__end__": END
        }
    )

    workflow.add_edge("chunk_and_embed", "summarize_briefing")
    workflow.add_edge("summarize_briefing", END)

    return workflow.compile()


def build_qa_graph() -> StateGraph:
    """Build the interactive QA graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("qa_node", qa_node)
    workflow.add_edge(START, "qa_node")
    workflow.add_edge("qa_node", END)
    return workflow.compile()


# Cached compiled graph instances
ingestion_app = build_ingestion_graph()
qa_app = build_qa_graph()
