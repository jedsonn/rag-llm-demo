"""
Vector Store Module using FAISS

Stores embeddings and enables fast similarity search.
FAISS (Facebook AI Similarity Search) is fast and runs locally.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INDEX_DIR, EMBEDDING_DIMENSION, TOP_K_CHUNKS, PROCESSED_DATA_DIR, logger
from src.embeddings import batch_embed, get_embedding


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """
    FAISS-based vector store for similarity search.

    Uses IndexFlatIP (Inner Product) for cosine similarity search
    with normalized vectors.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension (384 for MiniLM)
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def create_index(self):
        """Create a new empty FAISS index."""
        # Using Inner Product (IP) for cosine similarity
        # (embeddings will be normalized before adding)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info(f"Created new FAISS index with dimension {self.dimension}")

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add vectors to the index.

        Args:
            embeddings: Array of shape (n, dimension)
            metadata_list: List of metadata dicts for each vector
        """
        if self.index is None:
            self.create_index()

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(metadata_list)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = TOP_K_CHUNKS) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(SearchResult(
                    id=meta.get('id', str(idx)),
                    text=meta.get('text', ''),
                    score=float(score),
                    metadata=meta
                ))

        return results

    def save(self, path: Path):
        """
        Save index and metadata to disk.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to: {index_path}")

        # Save metadata
        metadata_path = path / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        logger.info(f"Saved metadata to: {metadata_path}")

    def load(self, path: Path) -> bool:
        """
        Load index and metadata from disk.

        Args:
            path: Directory to load from

        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        index_path = path / "index.faiss"
        metadata_path = path / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Index files not found in: {path}")
            return False

        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        return True


def build_index_from_chunks() -> VectorStore:
    """
    Build a FAISS index from all chunked documents.

    Returns:
        VectorStore with all chunks indexed
    """
    # Load all chunks
    chunks_path = PROCESSED_DATA_DIR / "all_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    logger.info(f"Loaded {len(chunks)} chunks")

    # Extract texts for embedding
    texts = [c['text'] for c in chunks]

    # Generate embeddings
    logger.info("Generating embeddings (this may take a few minutes)...")
    embeddings = batch_embed(texts, batch_size=32, show_progress=True)

    # Create vector store
    store = VectorStore()
    store.create_index()

    # Prepare metadata (include full chunk info)
    metadata_list = []
    for chunk in chunks:
        metadata_list.append({
            'id': chunk['id'],
            'text': chunk['text'],
            'company': chunk['company'],
            'ticker': chunk['ticker'],
            'filing_type': chunk['filing_type'],
            'filing_date': chunk['filing_date'],
            'section': chunk['section'],
            'chunk_index': chunk['chunk_index'],
            'total_chunks': chunk['total_chunks']
        })

    # Add to index
    store.add_vectors(embeddings, metadata_list)

    # Save to disk
    store.save(INDEX_DIR)

    return store


def load_index() -> VectorStore:
    """
    Load the saved FAISS index.

    Returns:
        VectorStore loaded from disk
    """
    store = VectorStore()
    if store.load(INDEX_DIR):
        return store
    else:
        raise FileNotFoundError(f"No index found at {INDEX_DIR}")


def main():
    """Build and test the vector index."""
    print("=" * 60)
    print("Vector Store Builder")
    print("=" * 60)

    # Build index
    print("\nBuilding FAISS index from chunks...")
    store = build_index_from_chunks()

    print(f"\nIndex contains {store.index.ntotal} vectors")

    # Test search
    print("\n" + "=" * 60)
    print("Testing Search")
    print("=" * 60)

    test_queries = [
        "What was Airbnb's revenue in 2019?",
        "How many employees does DoorDash have?",
        "What are Snowflake's risk factors?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        query_embedding = get_embedding(query)
        results = store.search(query_embedding, k=3)

        for i, r in enumerate(results):
            print(f"\n{i+1}. Score: {r.score:.3f} | {r.metadata['company']} | {r.metadata['section']}")
            print(f"   {r.text[:150]}...")


if __name__ == "__main__":
    main()
