"""
Embeddings Module for RAG Pipeline

Generates embeddings for text using sentence-transformers.
Uses all-MiniLM-L6-v2 by default (free, local, no API needed).
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, logger


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.

    Uses all-MiniLM-L6-v2 by default:
    - 384 dimensions
    - Fast and efficient
    - Good quality for semantic search
    - Runs locally (no API costs)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.dimension = EMBEDDING_DIMENSION

    def _load_model(self):
        """Load the model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def batch_embed(self, texts: List[str], batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        self._load_model()

        logger.info(f"Embedding {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)


# Singleton instance for convenience
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding for a single text.
    Uses a singleton model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model.get_embedding(text)


def batch_embed(texts: List[str], **kwargs) -> np.ndarray:
    """
    Get embeddings for multiple texts.
    Uses a singleton model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model.batch_embed(texts, **kwargs)


def main():
    """Test the embedding model."""
    print("=" * 60)
    print("Embedding Model Test")
    print("=" * 60)

    # Test single embedding
    test_text = "What was Airbnb's revenue in 2019?"
    print(f"\nTest text: {test_text}")

    embedding = get_embedding(test_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    test_texts = [
        "Airbnb reported revenue of $4.81 billion in fiscal year 2019.",
        "DoorDash had 1 million Dashers on its platform.",
        "Snowflake is a cloud-based data warehouse company.",
        "Rivian manufactures electric vehicles.",
        "Palantir provides data analytics software."
    ]

    print(f"\nBatch embedding {len(test_texts)} texts...")
    embeddings = batch_embed(test_texts)
    print(f"Batch embeddings shape: {embeddings.shape}")

    # Test similarity
    from numpy.linalg import norm

    query = get_embedding("What was Airbnb's revenue?")
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = np.dot(query, emb) / (norm(query) * norm(emb))
        similarities.append((sim, test_texts[i][:50]))

    print("\nSimilarity to 'What was Airbnb's revenue?':")
    for sim, text in sorted(similarities, reverse=True):
        print(f"  {sim:.3f}: {text}...")


if __name__ == "__main__":
    main()
