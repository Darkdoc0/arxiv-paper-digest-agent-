"""Nodes: Chunking & local embedding, and structured Executive Briefing generation."""

import logging
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState, ExecutiveBriefing
from src.tools.vector_store import create_vector_store_from_parsed
from src.tools.llm import get_llm

logger = logging.getLogger(__name__)

# In-memory registry to hold active Chroma vector stores for sessions
ACTIVE_VECTOR_STORES = {}


def chunk_and_embed_node(state: AgentState) -> Dict[str, Any]:
    """Chunk paper sections and build local Chroma vector store with sentence-transformers embeddings."""
    parsed = state.get("parsed_paper")
    selected_paper = state.get("selected_paper")

    if not parsed or not selected_paper:
        return {
            "error": "Cannot chunk and embed: Missing parsed paper or paper metadata."
        }

    logger.info(f"chunk_and_embed_node: Indexing paper '{selected_paper.arxiv_id}'...")
    try:
        vector_store, collection_name = create_vector_store_from_parsed(
            parsed=parsed,
            paper_id=selected_paper.arxiv_id
        )
        ACTIVE_VECTOR_STORES[collection_name] = vector_store
        logger.info(f"Successfully indexed into vector store collection: {collection_name}")
        return {
            "vector_store_collection": collection_name,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return {
            "error": f"Failed to build vector store: {str(e)}"
        }


def summarize_briefing_node(state: AgentState) -> Dict[str, Any]:
    """Generate a strict, structured Executive Briefing using LLM structured output."""
    paper = state.get("selected_paper")
    parsed = state.get("parsed_paper")

    if not paper or not parsed:
        return {
            "briefing": None,
            "error": "Cannot generate executive briefing without paper metadata and content."
        }

    logger.info(f"summarize_briefing_node: Generating structured briefing for '{paper.title}'...")

    # Build context window from parsed sections
    context_blocks = [
        f"Title: {paper.title}",
        f"Authors: {', '.join(paper.authors)}",
        f"arXiv ID: {paper.arxiv_id}",
        f"Published Date: {paper.published}",
        f"Abstract:\n{parsed.abstract}\n"
    ]

    if not parsed.is_abstract_fallback:
        for section_name in ["Introduction", "Methodology & Architecture", "Experiments & Results", "Limitations & Discussion", "Conclusion"]:
            if section_name in parsed.sections:
                content = parsed.sections[section_name]
                # Truncate section if very long to comfortably fit prompt
                snippet = content[:2500] if len(content) > 2500 else content
                context_blocks.append(f"--- SECTION: {section_name} ---\n{snippet}\n")
    else:
        context_blocks.append("NOTE: Full text unavailable. Summary generated solely from the official arXiv abstract.")

    paper_context = "\n".join(context_blocks)

    system_prompt = (
        "You are an elite AI research scientist providing an executive briefing on an academic paper.\n"
        "You must analyze the provided paper content and produce a strict, highly detailed Executive Briefing.\n\n"
        "Guidelines:\n"
        "1. Title, authors, arxiv_id, and published_date must match the paper exactly.\n"
        "2. 'why_this_matters': Exactly one punchy, high-impact paragraph explaining why this paper is significant for the field and enterprise/engineering practitioners.\n"
        "3. 'problem_statement': Clarify the foundational problem, current limitations of prior methods, and core research question.\n"
        "4. 'method': Bullet points detailing key algorithms, architectures, training objectives, and mechanisms.\n"
        "5. 'key_results_claims': Bullet points detailing empirical results with concrete metrics (accuracy, throughput, perplexity, speedup, FLOP reduction) and key findings.\n"
        "6. 'explicit_limitations': MANDATORY NON-EMPTY LIST. If the authors explicitly state limitations, enumerate them. If the paper omits them, YOU must critically identify unaddressed assumptions, evaluation gaps, scalability bottlenecks, or vulnerability to failure modes.\n"
        "7. 'follow_up_questions': Exactly 3 rigorous, actionable research or engineering questions for future investigation.\n"
    )

    try:
        llm = get_llm(temperature=0.1)
        structured_llm = llm.with_structured_output(ExecutiveBriefing)
        briefing: ExecutiveBriefing = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Paper Content:\n\n{paper_context}")
        ])

        # Enforce non-empty explicit limitations (safety guardrail)
        if not briefing.explicit_limitations:
            briefing.explicit_limitations = [
                "Evaluation scope limited to reported benchmark datasets; generalizability to open-domain distribution shifts unverified.",
                "Computational overhead and inference latency under constrained hardware environments not thoroughly characterized."
            ]

        # Enforce exact length of follow_up_questions
        if len(briefing.follow_up_questions) < 3:
            briefing.follow_up_questions.extend([
                "How does this approach scale when context lengths or parameter counts increase by an order of magnitude?",
                "What are the deployment latency tradeoffs when applied in real-time edge serving pipelines?"
            ][:3 - len(briefing.follow_up_questions)])

        logger.info(f"Executive briefing generated successfully for {paper.arxiv_id}")
        return {
            "briefing": briefing,
            "error": None
        }

    except Exception as e:
        logger.error(f"Failed to generate structured briefing: {e}")
        # Fallback briefing
        fallback_briefing = ExecutiveBriefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            published_date=paper.published,
            why_this_matters="This paper addresses key research challenges outlined in its abstract.",
            problem_statement=paper.summary[:300],
            method=["Refer to paper abstract for methodological approach."],
            key_results_claims=["Empirical benchmarks detailed in the paper."],
            explicit_limitations=["Detailed analysis restricted due to structured parsing error."],
            follow_up_questions=[
                "What are the primary computational requirements for reproducing this work?",
                "How does this compare against the latest open-source baselines?",
                "What failure modes emerge under adversarial distribution shift?"
            ]
        )
        return {
            "briefing": fallback_briefing,
            "error": f"Briefing generated with fallback schema due to: {str(e)}"
        }
