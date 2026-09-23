"""State definitions and Pydantic schemas for the arXiv Agent."""

from __future__ import annotations
from typing import Annotated, Dict, List, Literal, Optional, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PaperMetadata(BaseModel):
    """Metadata for an arXiv paper retrieved from the arXiv API."""
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    arxiv_id: str = Field(description="Normalized arXiv identifier (e.g., '2310.06825')")
    published: str = Field(description="Publication or announcement date (ISO or YYYY-MM-DD)")
    summary: str = Field(description="Abstract summary provided by arXiv")
    pdf_url: str = Field(description="Direct URL to download the PDF")
    primary_category: Optional[str] = Field(default=None, description="Primary arXiv category e.g. cs.CL")
    comment: Optional[str] = Field(default=None, description="Author comments / venue info")
    doi: Optional[str] = Field(default=None, description="DOI if available")


class ParsedDocument(BaseModel):
    """Structured text content extracted from the paper PDF."""
    abstract: str = Field(default="", description="Abstract of the paper")
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping section title to section body text"
    )
    fulltext: str = Field(default="", description="Aggregated full document text")
    page_count: int = Field(default=0, description="Total number of pages parsed")
    is_abstract_fallback: bool = Field(
        default=False,
        description="True if full PDF could not be parsed and only abstract is available"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings encountered during download or extraction"
    )


class ExecutiveBriefing(BaseModel):
    """Strict structured Executive Briefing required by the evaluation rubric."""
    title: str = Field(description="The formal title of the paper")
    authors: List[str] = Field(description="List of primary authors")
    arxiv_id: str = Field(description="Normalized arXiv ID")
    published_date: str = Field(description="Date published or latest revision date")
    why_this_matters: str = Field(
        description="One crisp paragraph explaining why this paper is significant for the field and practitioners"
    )
    problem_statement: str = Field(
        description="Clear articulation of the problem or research question being investigated"
    )
    method: List[str] = Field(
        description="Key methodological steps, architectural choices, and technical mechanisms (bullet points)"
    )
    key_results_claims: List[str] = Field(
        description="Key empirical results, quantitative metrics, or theoretical claims (bullet points with concrete numbers)"
    )
    explicit_limitations: List[str] = Field(
        description="Mandatory list of limitations, boundary conditions, or unaddressed failure modes. Cannot be omitted!"
    )
    follow_up_questions: List[str] = Field(
        description="Exactly 3 thought-provoking follow-up questions for future research or system design"
    )


class AgentState(TypedDict):
    """The central state of the LangGraph autonomous agent."""
    query: str
    intent: Literal["direct_id", "topic_search"]
    arxiv_id: Optional[str]
    search_query: Optional[str]
    candidate_papers: List[PaperMetadata]
    selected_paper: Optional[PaperMetadata]
    parsed_paper: Optional[ParsedDocument]
    briefing: Optional[ExecutiveBriefing]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    error: Optional[str]
    # Edge-case & pipeline tracking fields
    is_fallback_abstract: bool
    retry_count: int
    selection_reasoning: Optional[str]
    vector_store_collection: Optional[str]
