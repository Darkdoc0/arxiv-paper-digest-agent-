"""CLI Entrypoint for the Autonomous arXiv Paper Digest & QA Agent."""

import sys
import argparse
import logging
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.state import AgentState, ExecutiveBriefing
from src.graph import ingestion_app, qa_app

console = Console()


def display_banner():
    """Display attractive Rich startup banner."""
    console.print(
        Panel.fit(
            "[bold cyan]Autonomous arXiv Paper Digest & QA Agent[/bold cyan]\n"
            "[dim]Production-grade, modular LangGraph workflow with local embeddings & grounded RAG[/dim]",
            border_style="cyan"
        )
    )


def display_candidate_papers(candidates, selected_paper=None):
    """Render table of candidate papers returned by search."""
    table = Table(title="arXiv Retrieved Candidate Papers", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("Title", style="bold white", width=42)
    table.add_column("Published", style="green", width=12)
    table.add_column("Primary Author", style="yellow", width=18)

    for idx, paper in enumerate(candidates, 1):
        first_author = paper.authors[0] if paper.authors else "N/A"
        is_sel = selected_paper and selected_paper.arxiv_id == paper.arxiv_id
        prefix = "-> " if is_sel else "   "
        table.add_row(
            f"{prefix}{idx}",
            paper.arxiv_id,
            paper.title[:40] + ("..." if len(paper.title) > 40 else ""),
            paper.published,
            first_author
        )
    console.print(table)


def display_briefing(briefing: ExecutiveBriefing, is_fallback: bool = False):
    """Render the structured Executive Briefing using Rich panels and tables."""
    if is_fallback:
        console.print(
            Panel(
                "[bold yellow]WARNING: PDF download/parsing encountered an issue or is a scanned image.\n"
                "The agent gracefully fell back to extracting context from the official arXiv abstract.[/bold yellow]",
                title="Fallback Mode Notice",
                border_style="yellow"
            )
        )

    # Header Card
    meta_md = (
        f"**Title:** {briefing.title}\n\n"
        f"**Authors:** {', '.join(briefing.authors)}\n\n"
        f"**arXiv ID:** `{briefing.arxiv_id}` | **Published:** `{briefing.published_date}`"
    )
    console.print(Panel(Markdown(meta_md), title="Paper Overview", border_style="blue"))

    # Why this matters
    console.print(
        Panel(
            f"[bold italic]{briefing.why_this_matters}[/bold italic]",
            title="[bold green]Why This Matters[/bold green]",
            border_style="green"
        )
    )

    # Problem Statement
    console.print(
        Panel(
            briefing.problem_statement,
            title="Problem Statement & Research Question",
            border_style="cyan"
        )
    )

    # Method & Technical Mechanisms
    methods_text = "\n".join([f"- {m}" for m in briefing.method])
    console.print(Panel(Markdown(methods_text), title="Method & Core Mechanisms", border_style="magenta"))

    # Key Results & Quantitative Claims
    results_text = "\n".join([f"- {r}" for r in briefing.key_results_claims])
    console.print(Panel(Markdown(results_text), title="Key Results & Empirical Claims", border_style="bright_blue"))

    # Explicit Limitations (Mandatory)
    limitations_text = "\n".join([f"- {lim}" for lim in briefing.explicit_limitations])
    console.print(
        Panel(
            Markdown(limitations_text),
            title="[bold red]Explicit Limitations & Assumptions (Mandatory)[/bold red]",
            border_style="red"
        )
    )

    # Follow-Up Questions
    questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(briefing.follow_up_questions, 1)])
    console.print(
        Panel(
            Markdown(questions_text),
            title="Suggested Follow-Up Research Questions",
            border_style="gold1"
        )
    )


def interactive_qa_loop(state: AgentState):
    """Interactive loop allowing the user to ask grounded questions about the indexed paper."""
    console.print("\n[bold green]=== Interactive QA Session Active ===[/bold green]")
    console.print("[dim]Type your question and press Enter. (Type 'exit', 'quit', or 'q' to end)[/dim]\n")

    while True:
        try:
            user_question = Prompt.ask("[bold cyan]Question[/bold cyan]")
            if not user_question or user_question.strip().lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Exiting QA session. Thank you![/yellow]")
                break

            with console.status("[bold green]Searching paper and generating grounded response...[/bold green]", spinner="dots"):
                # Append user question to state messages
                state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=user_question)]
                qa_result = qa_app.invoke(state)
                # Update state messages
                state["messages"] = qa_result["messages"]
                last_reply = state["messages"][-1].content

            # Render response
            is_refusal = (
                "information is not provided" in last_reply.lower() or
                "does not contain sufficient information" in last_reply.lower()
            )
            border_color = "red" if is_refusal else "green"
            title = "Grounded Response" if not is_refusal else "Insufficient Context Refusal"

            console.print(Panel(Markdown(last_reply), title=title, border_style=border_color))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting QA session.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error during QA interaction: {e}[/red]")


def run_pipeline(user_query: str):
    """Execute the full agent workflow for a given query."""
    initial_state: AgentState = {
        "query": user_query,
        "intent": "topic_search",
        "arxiv_id": None,
        "search_query": None,
        "candidate_papers": [],
        "selected_paper": None,
        "parsed_paper": None,
        "briefing": None,
        "messages": [],
        "error": None,
        "is_fallback_abstract": False,
        "retry_count": 0,
        "selection_reasoning": None,
        "vector_store_collection": None,
    }

    with console.status("[bold cyan]Running Autonomous arXiv Agent workflow...[/bold cyan]", spinner="bouncingBar") as status:
        status.update("[bold cyan]Parsing query intent...[/bold cyan]")
        result = ingestion_app.invoke(initial_state)

    if result.get("error"):
        console.print(Panel(f"[bold red]Pipeline Error:[/bold red] {result['error']}", border_style="red"))
        return

    # Display candidate papers if topic search was run
    if result.get("candidate_papers") and len(result["candidate_papers"]) > 1:
        display_candidate_papers(result["candidate_papers"], result.get("selected_paper"))
        if result.get("selection_reasoning"):
            console.print(
                Panel(
                    f"[bold]Selected:[/bold] {result['selected_paper'].title}\n"
                    f"[dim]Rationale: {result['selection_reasoning']}[/dim]",
                    title="LLM Ranking Decision",
                    border_style="purple"
                )
            )

    # Display Executive Briefing
    if result.get("briefing"):
        display_briefing(result["briefing"], result.get("is_fallback_abstract", False))
        # Start interactive QA loop
        interactive_qa_loop(result)
    else:
        console.print("[red]No executive briefing was generated.[/red]")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Autonomous arXiv Paper Digest & QA Agent")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="arXiv ID (e.g. '2310.06825'), URL, or topic search query"
    )
    args = parser.parse_args()

    display_banner()

    query = args.query
    if not query:
        query = Prompt.ask("[bold green]Enter arXiv ID, paper URL, or research topic[/bold green]")

    if not query or not query.strip():
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    run_pipeline(query.strip())


if __name__ == "__main__":
    main()
