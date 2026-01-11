"""
Script 04: Test the complete query pipeline

Tests all three modes (Raw LLM, RAG Only, RAG + LLM) with sample queries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import QueryPipeline


def main():
    print("=" * 70)
    print("RAG Demo - Complete Pipeline Test")
    print("=" * 70)

    pipeline = QueryPipeline()

    test_questions = [
        "What was Airbnb's total revenue in fiscal year 2019?",
        "How many Dashers were on DoorDash's platform?",
        "What is Snowflake's primary business?",
    ]

    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = pipeline.compare_all(question)

        # RAG Only (should always work)
        print("\n[1] RAG ONLY (Retrieved Chunks)")
        print("-" * 40)
        if result.rag_only:
            print(f"Time: {result.rag_only.elapsed_time:.2f}s")
            print(f"Chunks: {len(result.rag_only.chunks)}")
            for i, chunk in enumerate(result.rag_only.chunks[:2]):
                print(f"\n  Chunk {i+1}:")
                print(f"    Score: {chunk.score:.3f}")
                print(f"    Company: {chunk.metadata['company']}")
                print(f"    Section: {chunk.metadata['section']}")
                text = chunk.text[:150].encode('ascii', 'replace').decode()
                print(f"    Preview: {text}...")
        else:
            print("  FAILED")

        # Raw LLM
        print("\n[2] RAW LLM (No Context)")
        print("-" * 40)
        if result.raw_llm:
            print(f"Time: {result.raw_llm.elapsed_time:.2f}s")
            print(f"Model: {result.raw_llm.model}")
            answer = result.raw_llm.answer.encode('ascii', 'replace').decode()
            print(f"Answer: {answer[:500]}...")
        else:
            print("  FAILED (API key may not be set)")

        # RAG + LLM
        print("\n[3] RAG + LLM (Grounded)")
        print("-" * 40)
        if result.rag_plus_llm:
            print(f"Time: {result.rag_plus_llm.elapsed_time:.2f}s")
            print(f"  Retrieval: {result.rag_plus_llm.retrieval_time:.2f}s")
            print(f"  LLM: {result.rag_plus_llm.llm_time:.2f}s")
            print(f"Model: {result.rag_plus_llm.model}")
            answer = result.rag_plus_llm.answer.encode('ascii', 'replace').decode()
            print(f"Answer: {answer[:500]}...")
        else:
            print("  FAILED (API key may not be set)")

        if result.errors:
            print(f"\nErrors: {result.errors}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
