"""
LLM Client Module for RAG Pipeline

Provides interfaces for:
1. Raw LLM queries (no context - may hallucinate)
2. RAG + LLM queries (grounded in retrieved documents)

Uses OpenAI GPT-4.1-mini for cost-effective responses.
"""

import time
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, logger


@dataclass
class LLMResponse:
    """Container for LLM response."""
    text: str
    model: str
    elapsed_time: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMClient:
    """
    Client for OpenAI API calls.

    Provides two modes:
    1. raw_query: Direct question answering (no context)
    2. rag_query: Question answering with retrieved context
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenAI API key. If None, uses environment variable.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        self.model = LLM_MODEL

    def _ensure_client(self):
        """Initialize OpenAI client if needed."""
        if self.client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please set it in .env file or environment."
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

    def raw_query(self, question: str) -> LLMResponse:
        """
        Query the LLM directly without any context.
        This relies purely on the model's parametric knowledge.
        May produce hallucinations for specific factual questions.

        Args:
            question: User's question

        Returns:
            LLMResponse with the answer
        """
        self._ensure_client()

        prompt = f"""You are a helpful financial analyst assistant. Answer the following
question about a company's financial information. Be specific with numbers if you know them.
If you're not certain, indicate your uncertainty.

Question: {question}

Answer:"""

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        elapsed = time.time() - start_time

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            elapsed_time=elapsed,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

    def rag_query(self, question: str, context: str) -> LLMResponse:
        """
        Query the LLM with retrieved context (RAG).
        Grounds the response in actual SEC filing excerpts.

        Args:
            question: User's question
            context: Retrieved document excerpts

        Returns:
            LLMResponse with the grounded answer
        """
        self._ensure_client()

        prompt = f"""You are a helpful financial analyst assistant. Answer the following
question based ONLY on the provided SEC filing excerpts. Be precise with numbers and
cite which company/filing the information comes from.

If the information to answer the question is not in the provided excerpts, say
"The provided excerpts do not contain information to answer this question."

SEC FILING EXCERPTS:
{context}

QUESTION: {question}

ANSWER (based only on the excerpts above):"""

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        elapsed = time.time() - start_time

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            elapsed_time=elapsed,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )


# Singleton client
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def raw_llm_query(question: str) -> LLMResponse:
    """Query LLM directly without context."""
    return get_llm_client().raw_query(question)


def rag_llm_query(question: str, context: str) -> LLMResponse:
    """Query LLM with retrieved context."""
    return get_llm_client().rag_query(question, context)


def main():
    """Test the LLM client."""
    print("=" * 60)
    print("LLM Client Test (OpenAI GPT-4.1-mini)")
    print("=" * 60)

    client = LLMClient()

    # Test raw query
    question = "What was Airbnb's total revenue in fiscal year 2019?"
    print(f"\nQuestion: {question}")

    print("\n--- Raw LLM (no context) ---")
    try:
        response = client.raw_query(question)
        print(f"Time: {response.elapsed_time:.2f}s")
        print(f"Answer: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test RAG query with sample context
    sample_context = """
[Source 1: Airbnb S-1 (2020-12-07), FINANCIAL DATA]
For the year ended December 31, 2019, we generated revenue of $4.81 billion,
representing a 32% increase from revenue of $3.65 billion for the year ended
December 31, 2018.

[Source 2: Airbnb S-1 (2020-12-07), MD&A]
Revenue increased 32% from 2018 to 2019, driven primarily by growth in
Nights and Experiences Booked and an increase in average daily rates.
"""

    print("\n--- RAG + LLM (with context) ---")
    try:
        response = client.rag_query(question, sample_context)
        print(f"Time: {response.elapsed_time:.2f}s")
        print(f"Answer: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
