"""Node: Grounded RAG Question Answering with strict citation and hallucination refusal."""

import logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.nodes.briefing import ACTIVE_VECTOR_STORES
from src.tools.vector_store import retrieve_relevant_chunks
from src.tools.llm import get_llm, extract_text_content

logger = logging.getLogger(__name__)

STRICT_REFUSAL_PHRASE = "Based on the paper's retrieved sections, this information is not provided."


def qa_node(state: AgentState) -> Dict[str, Any]:
    """Execute grounded question answering against the indexed paper vector store."""
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [AIMessage(content="Please provide a question about the paper.")]
        }

    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_user_message:
        return {
            "messages": [AIMessage(content="No user question found in message history.")]
        }

    question = last_user_message.content.strip()
    selected_paper = state.get("selected_paper")
    paper_title = selected_paper.title if selected_paper else "Academic Paper"

    collection_name = state.get("vector_store_collection")
    vector_store = ACTIVE_VECTOR_STORES.get(collection_name)

    if not vector_store:
        logger.warning("No active vector store found for QA.")
        parsed = state.get("parsed_paper")
        if parsed and parsed.abstract:
            context = f"--- [Section: Abstract] ---\n{parsed.abstract}"
        else:
            return {
                "messages": [AIMessage(content=STRICT_REFUSAL_PHRASE)]
            }
    else:
        # Perform thresholded similarity search
        matching_docs, context = retrieve_relevant_chunks(
            vector_store=vector_store,
            query=question,
            top_k=4,
            similarity_threshold=0.20
        )

        if not matching_docs or not context.strip():
            logger.info(f"Question '{question}' yielded no chunks above similarity threshold.")
            return {
                "messages": [AIMessage(content=STRICT_REFUSAL_PHRASE)]
            }

    system_prompt = (
        f'You are a factual research assistant answering questions about the academic paper: "{paper_title}".\n\n'
        f"RETRIEVED CONTEXT FROM PAPER:\n{context}\n\n"
        "RULES:\n"
        "1. Answer ONLY using the retrieved context above.\n"
        "2. For each factual claim, reference the relevant section or chunk context (e.g., \"[Section: Methodology]\").\n"
        "3. If the answer cannot be directly determined from the provided context, state:\n"
        f'   "{STRICT_REFUSAL_PHRASE}"\n'
        "   Do NOT attempt to infer, extrapolate, or use outside knowledge.\n"
        "4. Keep answers concise, technical, and precise."
    )

    try:
        llm = get_llm(temperature=0.0)
        ai_response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
        
        raw_text = extract_text_content(ai_response.content)
        answer_text = raw_text.strip()

        # Guardrail: Check if the model answered despite lack of evidence
        if "i do not know" in answer_text.lower() or "not mentioned" in answer_text.lower() or "cannot be answered" in answer_text.lower():
            if STRICT_REFUSAL_PHRASE not in answer_text:
                answer_text = STRICT_REFUSAL_PHRASE

        return {
            "messages": [AIMessage(content=answer_text)]
        }

    except Exception as e:
        logger.error(f"Error during QA generation: {e}")
        return {
            "messages": [AIMessage(content=f"An error occurred while answering your question: {str(e)}")]
        }
