"""
Retriever Module for RAG Pipeline

Provides high-level retrieval interface for the RAG system.
Handles query embedding, vector search, and result formatting.
"""

from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K_CHUNKS, logger
from src.embeddings import get_embedding
from src.vector_store import VectorStore, SearchResult, load_index


@dataclass
class RetrievalResult:
    """Container for retrieval results with formatted context."""
    query: str
    chunks: List[SearchResult]
    context_text: str


class Retriever:
    """
    High-level retriever for the RAG pipeline.

    Wraps vector store operations and provides convenient methods
    for querying and formatting results.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        """
        Initialize retriever.

        Args:
            store: Optional VectorStore instance. If None, loads from disk.
        """
        self.store = store

    def _ensure_store(self):
        """Ensure vector store is loaded."""
        if self.store is None:
            logger.info("Loading vector store from disk...")
            self.store = load_index()

    def retrieve(self, query: str, k: int = TOP_K_CHUNKS) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's question
            k: Number of chunks to retrieve

        Returns:
            RetrievalResult with chunks and formatted context
        """
        self._ensure_store()

        # Embed query
        query_embedding = get_embedding(query)

        # Search
        chunks = self.store.search(query_embedding, k=k)

        # Format context for LLM
        context_parts = []
        for i, chunk in enumerate(chunks):
            company = chunk.metadata.get('company', 'Unknown')
            filing_date = chunk.metadata.get('filing_date', 'Unknown')
            section = chunk.metadata.get('section', 'Unknown')

            context_parts.append(
                f"[Source {i+1}: {company} S-1 ({filing_date}), {section}]\n"
                f"{chunk.text}\n"
            )

        context_text = "\n---\n".join(context_parts)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            context_text=context_text
        )

    def retrieve_for_company(self, query: str, ticker: str, k: int = TOP_K_CHUNKS) -> RetrievalResult:
        """
        Retrieve chunks only from a specific company.

        Args:
            query: User's question
            ticker: Company ticker to filter by
            k: Number of chunks to retrieve

        Returns:
            RetrievalResult with filtered chunks
        """
        self._ensure_store()

        # Get more chunks and filter
        query_embedding = get_embedding(query)
        all_chunks = self.store.search(query_embedding, k=k * 3)

        # Filter by ticker
        filtered = [c for c in all_chunks if c.metadata.get('ticker') == ticker][:k]

        # Format context
        context_parts = []
        for i, chunk in enumerate(filtered):
            company = chunk.metadata.get('company', 'Unknown')
            filing_date = chunk.metadata.get('filing_date', 'Unknown')
            section = chunk.metadata.get('section', 'Unknown')

            context_parts.append(
                f"[Source {i+1}: {company} S-1 ({filing_date}), {section}]\n"
                f"{chunk.text}\n"
            )

        context_text = "\n---\n".join(context_parts)

        return RetrievalResult(
            query=query,
            chunks=filtered,
            context_text=context_text
        )


# Singleton retriever for convenience
_retriever: Optional[Retriever] = None


def retrieve(query: str, k: int = TOP_K_CHUNKS) -> RetrievalResult:
    """
    Retrieve relevant chunks for a query.
    Uses singleton retriever instance.
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever.retrieve(query, k)


def main():
    """Test retrieval."""
    print("=" * 60)
    print("Retriever Test")
    print("=" * 60)

    test_queries = [
        "What was Airbnb's revenue in 2019?",
        "How many Dashers does DoorDash have?",
        "What are Snowflake's main products?",
        "When did Rivian start producing vehicles?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        result = retrieve(query, k=3)

        for i, chunk in enumerate(result.chunks):
            print(f"\n{i+1}. Score: {chunk.score:.3f}")
            print(f"   Company: {chunk.metadata['company']}")
            print(f"   Section: {chunk.metadata['section']}")
            # Safe print that handles encoding
            text_preview = chunk.text[:150].encode('ascii', 'replace').decode()
            print(f"   Text: {text_preview}...")


if __name__ == "__main__":
    main()
