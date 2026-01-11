"""
Script 03: Build the FAISS vector index

Generates embeddings for all chunks and creates a searchable index.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import build_index_from_chunks
from config import INDEX_DIR


def main():
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)
    print(f"\nThis will generate embeddings for all chunks and build the index.")
    print(f"Output directory: {INDEX_DIR}")
    print("-" * 60)

    store = build_index_from_chunks()

    print("\n" + "=" * 60)
    print("Index Build Complete")
    print("=" * 60)
    print(f"Total vectors: {store.index.ntotal}")
    print(f"Dimension: {store.dimension}")
    print(f"Index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
