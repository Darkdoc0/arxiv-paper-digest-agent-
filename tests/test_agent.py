"""Comprehensive unit and integration test suite for the Autonomous arXiv Agent."""

import pytest
from langchain_core.messages import HumanMessage
from src.tools.arxiv_client import extract_arxiv_id
from src.tools.pdf_parser import parse_pdf_from_bytes
from src.state import ExecutiveBriefing, PaperMetadata, ParsedDocument
from src.nodes.query import parse_query_node
from src.nodes.rag import STRICT_REFUSAL_PHRASE, qa_node
from src.tools.vector_store import create_vector_store_from_parsed, retrieve_relevant_chunks


class TestArXivIDExtraction:
    """Test suite for arXiv ID and URL normalization."""

    @pytest.mark.parametrize("input_str,expected_id", [
        ("2310.06825", "2310.06825"),
        ("2310.06825v1", "2310.06825v1"),
        ("https://arxiv.org/abs/2310.06825", "2310.06825"),
        ("http://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("arxiv.org/abs/quant-ph/0101001v2", "quant-ph/0101001v2"),
        ("Please analyze paper 2401.12345 for me", "2401.12345"),
    ])
    def test_valid_arxiv_id_extraction(self, input_str, expected_id):
        result = extract_arxiv_id(input_str)
        assert result is not None
        assert expected_id in result

    def test_non_arxiv_string(self):
        assert extract_arxiv_id("deep learning architectures for robotics") is None
        assert extract_arxiv_id("https://google.com/search?q=ai") is None


class TestQueryParsingNode:
    """Test query intent routing."""

    def test_direct_id_intent(self):
        state = {"query": "2310.06825"}
        res = parse_query_node(state)
        assert res["intent"] == "direct_id"
        assert res["arxiv_id"] == "2310.06825"
        assert res["search_query"] is None

    def test_topic_search_intent(self):
        state = {"query": "speculative decoding"}
        res = parse_query_node(state)
        assert res["intent"] == "topic_search"
        assert res["arxiv_id"] is None
        assert res["search_query"] is not None


class TestPDFParserFallback:
    """Test section extraction and fallback handling for corrupted/empty PDF streams."""

    def test_corrupt_pdf_fallback(self):
        corrupt_bytes = b"NOT_A_VALID_PDF_HEADER_12345"
        fallback_summary = "This is a fallback summary of the research paper."
        parsed = parse_pdf_from_bytes(corrupt_bytes, fallback_abstract=fallback_summary)

        assert parsed.is_abstract_fallback is True
        assert parsed.abstract == fallback_summary
        assert len(parsed.warnings) > 0

    def test_empty_scanned_pdf_fallback(self):
        # Even if bytes are valid PDF with blank pages or image scans
        parsed = parse_pdf_from_bytes(b"", fallback_abstract="Abstract fallback")
        assert parsed.is_abstract_fallback is True


class TestExecutiveBriefingSchema:
    """Validate strict Pydantic requirements for executive briefing."""

    def test_briefing_schema_success(self):
        data = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "arxiv_id": "1706.03762",
            "published_date": "2017-06-12",
            "why_this_matters": "Introduced the Transformer architecture, replacing recurrent layers entirely.",
            "problem_statement": "Sequential computation in RNNs prevents parallelization during training.",
            "method": ["Multi-Head Self-Attention", "Sinusoidal Positional Encodings"],
            "key_results_claims": ["28.4 BLEU on WMT 2014 English-to-German", "Trained in 3.5 days on 8 GPUs"],
            "explicit_limitations": ["Quadratic time and memory complexity with respect to sequence length."],
            "follow_up_questions": [
                "How can quadratic attention complexity be linearized?",
                "Can positional encodings generalize to arbitrary lengths?",
                "What pretraining objectives maximize downstream transfer?"
            ]
        }
        briefing = ExecutiveBriefing(**data)
        assert len(briefing.follow_up_questions) == 3
        assert len(briefing.explicit_limitations) >= 1
        assert briefing.arxiv_id == "1706.03762"


class TestVectorStoreAndGroundedQA:
    """Test Chroma vector store indexing and grounded QA thresholding."""

    def test_indexing_and_similarity_retrieval(self):
        parsed = ParsedDocument(
            abstract="We present Mistral 7B, a 7-billion-parameter language model engineered for superior efficiency.",
            sections={
                "Methodology & Architecture": "Mistral 7B utilizes grouped-query attention (GQA) and sliding window attention (SWA).",
                "Experiments & Results": "Mistral 7B outperforms Llama 2 13B across all benchmarks tested."
            },
            fulltext="Full paper text...",
            page_count=9,
            is_abstract_fallback=False
        )

        vector_store, collection_name = create_vector_store_from_parsed(parsed, paper_id="2310.06825")
        assert vector_store is not None
        assert collection_name.startswith("paper_2310_06825")

        # In-scope query
        docs, context = retrieve_relevant_chunks(vector_store, "What attention mechanisms does Mistral use?", top_k=2)
        assert len(docs) > 0
        assert "sliding window attention" in context.lower() or "grouped-query" in context.lower()

        # Out-of-scope query with strict thresholding
        unrelated_docs, unrelated_context = retrieve_relevant_chunks(
            vector_store,
            "What is the recipe for chocolate chip cookies?",
            top_k=2,
            similarity_threshold=0.85
        )
        assert len(unrelated_docs) == 0
        assert unrelated_context == ""
