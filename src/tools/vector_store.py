"""ChromaDB vector store builder with local sentence-transformers embeddings and grounded retrieval."""

import os
import re
import uuid
import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.state import ParsedDocument

logger = logging.getLogger(__name__)

# Cached local embedding instance
_EMBEDDINGS = None


def get_embeddings():
    """Retrieve or initialize local sentence-transformers embedding model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Initializing HuggingFaceEmbeddings with all-MiniLM-L6-v2...")
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.warning(f"Failed to load langchain_huggingface, trying community fallback: {e}")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
    return _EMBEDDINGS


def create_vector_store_from_parsed(
    parsed: ParsedDocument,
    paper_id: str,
    persist_dir: Optional[str] = None
):
    """Chunk sections and index into an in-memory or persisted Chroma vector store."""
    from langchain_chroma import Chroma

    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents: List[Document] = []
    chunk_index = 0

    # If abstract fallback, index abstract
    if parsed.is_abstract_fallback or not parsed.sections:
        abstract_chunks = text_splitter.split_text(parsed.abstract)
        for c in abstract_chunks:
            documents.append(
                Document(
                    page_content=c,
                    metadata={
                        "paper_id": paper_id,
                        "section": "Abstract",
                        "chunk_id": f"chunk_{chunk_index}",
                        "source": f"{paper_id}#Abstract"
                    }
                )
            )
            chunk_index += 1
    else:
        # Index each section separately preserving section metadata
        for section_title, content in parsed.sections.items():
            if not content.strip():
                continue
            chunks = text_splitter.split_text(content)
            for c in chunks:
                documents.append(
                    Document(
                        page_content=c,
                        metadata={
                            "paper_id": paper_id,
                            "section": section_title,
                            "chunk_id": f"chunk_{chunk_index}",
                            "source": f"{paper_id}#{section_title}"
                        }
                    )
                )
                chunk_index += 1

    if not documents:
        logger.warning(f"No documents generated to index for paper {paper_id}")
        # Add fallback empty document
        documents.append(
            Document(
                page_content="No content available for this paper.",
                metadata={"paper_id": paper_id, "section": "Empty", "chunk_id": "chunk_0", "source": paper_id}
            )
        )

    # Unique collection name sanitized for Chroma (letters, numbers, underscores, dashes, 3-63 chars)
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_id)
    collection_name = f"paper_{clean_id}_{uuid.uuid4().hex[:6]}"

    logger.info(f"Indexing {len(documents)} chunks into Chroma collection '{collection_name}'...")

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )

    return vector_store, collection_name


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    top_k: int = 4,
    similarity_threshold: float = 0.25
) -> Tuple[List[Document], str]:
    """Retrieve top-k relevant chunks with similarity score thresholding.

    Returns:
        (documents, formatted_context_string)
    """
    try:
        # Chroma similarity_search_with_score returns (Document, cosine_distance)
        results_with_distance = vector_store.similarity_search_with_score(query, k=top_k)
        results_with_scores = [(doc, 1.0 - dist) for doc, dist in results_with_distance]
    except Exception as e:
        logger.warning(f"Distance scoring failed: {e}. Falling back to standard similarity search.")
        docs = vector_store.similarity_search(query, k=top_k)
        results_with_scores = [(doc, 1.0) for doc in docs]

    filtered_docs: List[Document] = []
    context_blocks: List[str] = []

    for doc, score in results_with_scores:
        if score >= similarity_threshold:
            filtered_docs.append(doc)
            sec = doc.metadata.get("section", "Unknown Section")
            cid = doc.metadata.get("chunk_id", "chunk")
            context_blocks.append(
                f"--- [Source: {doc.metadata.get('paper_id', 'Paper')} | Section: {sec} | Chunk: {cid} | Score: {score:.2f}] ---\n"
                f"{doc.page_content.strip()}"
            )

    formatted_context = "\n\n".join(context_blocks).strip()
    return filtered_docs, formatted_context
