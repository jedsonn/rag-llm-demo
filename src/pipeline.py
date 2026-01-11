"""
Query Pipeline for RAG Demo

Orchestrates the three comparison modes:
1. Raw LLM: Direct query with no context (may hallucinate)
2. RAG Only: Retrieved chunks (user reads raw excerpts)
3. RAG + LLM: Retrieved chunks fed to LLM for synthesis
"""

import time
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K_CHUNKS, logger
from src.retriever import Retriever, RetrievalResult, retrieve
from src.llm_client import LLMClient, LLMResponse, raw_llm_query, rag_llm_query
from src.vector_store import SearchResult


@dataclass
class RawLLMResult:
    """Result from Raw LLM query."""
    answer: str
    elapsed_time: float
    model: str
    warning: str = "This response is from the LLM's memory and may contain inaccuracies."


@dataclass
class RAGOnlyResult:
    """Result from RAG-only retrieval."""
    chunks: List[SearchResult]
    elapsed_time: float
    formatted_chunks: str


@dataclass
class RAGPlusLLMResult:
    """Result from RAG + LLM query."""
    answer: str
    chunks: List[SearchResult]
    elapsed_time: float
    retrieval_time: float
    llm_time: float
    model: str
    note: str = "This response is grounded in SEC filings."


@dataclass
class ComparisonResult:
    """Complete comparison of all three approaches."""
    query: str
    raw_llm: Optional[RawLLMResult] = None
    rag_only: Optional[RAGOnlyResult] = None
    rag_plus_llm: Optional[RAGPlusLLMResult] = None
    errors: List[str] = field(default_factory=list)


class QueryPipeline:
    """
    Main pipeline for comparing Raw LLM vs RAG approaches.
    """

    def __init__(self):
        """Initialize the pipeline components."""
        self.retriever = Retriever()
        self.llm_client = LLMClient()

    def query_raw_llm(self, question: str) -> RawLLMResult:
        """
        Query the LLM directly without any context.
        """
        start_time = time.time()
        response = self.llm_client.raw_query(question)
        elapsed = time.time() - start_time

        return RawLLMResult(
            answer=response.text,
            elapsed_time=elapsed,
            model=response.model
        )

    def query_rag_only(self, question: str, k: int = TOP_K_CHUNKS) -> RAGOnlyResult:
        """
        Retrieve relevant chunks without LLM synthesis.
        User reads the raw excerpts.
        """
        start_time = time.time()
        result = self.retriever.retrieve(question, k=k)
        elapsed = time.time() - start_time

        # Format chunks for display
        formatted_parts = []
        for i, chunk in enumerate(result.chunks):
            meta = chunk.metadata
            formatted_parts.append(
                f"**Chunk {i+1}** (Score: {chunk.score:.3f})\n"
                f"*{meta['company']} | {meta['filing_date']} | {meta['section']}*\n\n"
                f"{chunk.text}\n"
            )

        formatted = "\n---\n\n".join(formatted_parts)

        return RAGOnlyResult(
            chunks=result.chunks,
            elapsed_time=elapsed,
            formatted_chunks=formatted
        )

    def query_rag_plus_llm(self, question: str, k: int = TOP_K_CHUNKS) -> RAGPlusLLMResult:
        """
        Retrieve relevant chunks and synthesize with LLM.
        """
        # Retrieval
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(question, k=k)
        retrieval_time = time.time() - retrieval_start

        # LLM synthesis
        llm_start = time.time()
        llm_response = self.llm_client.rag_query(question, retrieval_result.context_text)
        llm_time = time.time() - llm_start

        total_time = retrieval_time + llm_time

        return RAGPlusLLMResult(
            answer=llm_response.text,
            chunks=retrieval_result.chunks,
            elapsed_time=total_time,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            model=llm_response.model
        )

    def compare_all(self, question: str, k: int = TOP_K_CHUNKS) -> ComparisonResult:
        """
        Run all three approaches and compare results.
        """
        result = ComparisonResult(query=question)

        # RAG Only (always works, no API needed)
        try:
            result.rag_only = self.query_rag_only(question, k=k)
            logger.info(f"RAG Only: {result.rag_only.elapsed_time:.2f}s")
        except Exception as e:
            error_msg = f"RAG Only failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        # Raw LLM (needs API key)
        try:
            result.raw_llm = self.query_raw_llm(question)
            logger.info(f"Raw LLM: {result.raw_llm.elapsed_time:.2f}s")
        except Exception as e:
            error_msg = f"Raw LLM failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        # RAG + LLM (needs API key)
        try:
            result.rag_plus_llm = self.query_rag_plus_llm(question, k=k)
            logger.info(f"RAG + LLM: {result.rag_plus_llm.elapsed_time:.2f}s")
        except Exception as e:
            error_msg = f"RAG + LLM failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        return result


def main():
    """Test the pipeline."""
    print("=" * 70)
    print("RAG Demo - Query Pipeline Test")
    print("=" * 70)

    pipeline = QueryPipeline()

    test_questions = [
        "What was Airbnb's total revenue in fiscal year 2019?",
        # "How many employees did DoorDash have at IPO?",
    ]

    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = pipeline.compare_all(question)

        # RAG Only
        print("\n--- RAG ONLY (Retrieved Chunks) ---")
        if result.rag_only:
            print(f"Time: {result.rag_only.elapsed_time:.2f}s")
            print(f"Retrieved {len(result.rag_only.chunks)} chunks:")
            for i, chunk in enumerate(result.rag_only.chunks[:2]):  # Show first 2
                print(f"\n  Chunk {i+1} (Score: {chunk.score:.3f}):")
                print(f"  Company: {chunk.metadata['company']}")
                # Safe text preview
                text = chunk.text[:200].encode('ascii', 'replace').decode()
                print(f"  Text: {text}...")
        else:
            print("  FAILED")

        # Raw LLM
        print("\n--- RAW LLM (No Context) ---")
        if result.raw_llm:
            print(f"Time: {result.raw_llm.elapsed_time:.2f}s")
            print(f"Model: {result.raw_llm.model}")
            print(f"Answer: {result.raw_llm.answer}")
            print(f"[Warning: {result.raw_llm.warning}]")
        else:
            print("  FAILED (check API key)")

        # RAG + LLM
        print("\n--- RAG + LLM (Grounded Answer) ---")
        if result.rag_plus_llm:
            print(f"Total Time: {result.rag_plus_llm.elapsed_time:.2f}s")
            print(f"  - Retrieval: {result.rag_plus_llm.retrieval_time:.2f}s")
            print(f"  - LLM: {result.rag_plus_llm.llm_time:.2f}s")
            print(f"Model: {result.rag_plus_llm.model}")
            print(f"Answer: {result.rag_plus_llm.answer}")
            print(f"[{result.rag_plus_llm.note}]")
        else:
            print("  FAILED (check API key)")

        if result.errors:
            print(f"\nErrors: {result.errors}")


if __name__ == "__main__":
    main()
