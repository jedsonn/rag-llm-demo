"""
Text Chunker for RAG Pipeline

Splits documents into overlapping chunks suitable for embedding and retrieval.
Preserves section context and adds metadata to each chunk.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_DATA_DIR,
    COMPANIES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAW_DATA_DIR,
    logger
)


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    text: str
    company: str
    ticker: str
    filing_type: str
    filing_date: str
    section: str
    chunk_index: int
    total_chunks: int


class TextChunker:
    """
    Splits documents into chunks for RAG.

    Uses character-based chunking with approximate token estimation.
    ~4 characters = 1 token (rough approximation for English)
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens (~500)
            overlap: Overlap between chunks in tokens (~50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Approximate: 4 characters = 1 token for English text
        self.chars_per_token = 4
        self.target_chars = chunk_size * self.chars_per_token
        self.overlap_chars = overlap * self.chars_per_token

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Tries to break at sentence boundaries when possible.
        """
        if len(text) <= self.target_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.target_chars

            if end >= len(text):
                # Last chunk - take everything remaining
                chunks.append(text[start:])
                break

            # Try to find a good break point (sentence boundary)
            # Look for period, question mark, or newline near the target
            search_start = max(start + self.target_chars - 200, start)
            search_end = min(start + self.target_chars + 100, len(text))
            search_text = text[search_start:search_end]

            # Find best break point
            best_break = None

            # Priority 1: Double newline (paragraph break)
            para_break = search_text.rfind('\n\n')
            if para_break != -1:
                best_break = search_start + para_break + 2

            # Priority 2: Single newline
            if best_break is None:
                line_break = search_text.rfind('\n')
                if line_break != -1:
                    best_break = search_start + line_break + 1

            # Priority 3: Period followed by space
            if best_break is None:
                period_break = search_text.rfind('. ')
                if period_break != -1:
                    best_break = search_start + period_break + 2

            # Fallback: just use target position
            if best_break is None or best_break <= start:
                best_break = end

            chunks.append(text[start:best_break])

            # Move start position, accounting for overlap
            start = best_break - self.overlap_chars
            if start < 0:
                start = best_break

        return chunks

    def detect_section(self, text: str) -> str:
        """
        Detect which section a chunk belongs to based on its content.
        """
        text_upper = text[:500].upper()  # Check first 500 chars

        if 'RISK FACTOR' in text_upper:
            return 'RISK FACTORS'
        elif 'REVENUE' in text_upper or 'NET INCOME' in text_upper or 'FISCAL' in text_upper:
            return 'FINANCIAL DATA'
        elif 'BUSINESS' in text_upper and 'RISK' not in text_upper:
            return 'BUSINESS'
        elif 'PROCEED' in text_upper:
            return 'USE OF PROCEEDS'
        elif 'MANAGEMENT' in text_upper and 'DISCUSSION' in text_upper:
            return 'MD&A'
        elif 'EMPLOYEE' in text_upper or 'PERSONNEL' in text_upper:
            return 'BUSINESS'
        else:
            return 'GENERAL'

    def chunk_document(self, ticker: str) -> Optional[List[Chunk]]:
        """
        Chunk a processed document.

        Args:
            ticker: Company ticker (e.g., "ABNB")

        Returns:
            List of Chunk objects, or None if failed
        """
        # Load processed text
        text_path = PROCESSED_DATA_DIR / f"{ticker}_S-1.txt"
        if not text_path.exists():
            logger.error(f"Processed file not found: {text_path}")
            return None

        text = text_path.read_text(encoding='utf-8')

        # Load metadata
        metadata_path = RAW_DATA_DIR / f"{ticker}_S-1_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            filing_date = metadata.get('filingDate', 'Unknown')
        else:
            filing_date = 'Unknown'

        company_name = COMPANIES.get(ticker, {}).get('name', ticker)

        # Split into chunks
        text_chunks = self.chunk_text(text)
        total = len(text_chunks)

        logger.info(f"{ticker}: Split into {total} chunks")

        # Create Chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            section = self.detect_section(chunk_text)

            chunk = Chunk(
                id=f"{ticker}_S-1_chunk_{i:04d}",
                text=chunk_text.strip(),
                company=company_name,
                ticker=ticker,
                filing_type="S-1",
                filing_date=filing_date,
                section=section,
                chunk_index=i,
                total_chunks=total
            )
            chunks.append(chunk)

        return chunks

    def save_chunks(self, chunks: List[Chunk], output_path: Path):
        """Save chunks to JSON file."""
        data = [asdict(c) for c in chunks]
        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"Saved {len(chunks)} chunks to: {output_path}")

    def chunk_all_documents(self) -> Dict[str, List[Chunk]]:
        """
        Chunk all processed documents.

        Returns:
            Dict mapping ticker to list of chunks
        """
        all_chunks = {}

        for ticker in COMPANIES:
            chunks = self.chunk_document(ticker)
            if chunks:
                all_chunks[ticker] = chunks

                # Save individual company chunks
                output_path = PROCESSED_DATA_DIR / f"{ticker}_chunks.json"
                self.save_chunks(chunks, output_path)

        # Save combined chunks file
        combined = []
        for ticker, chunks in all_chunks.items():
            combined.extend(chunks)

        combined_path = PROCESSED_DATA_DIR / "all_chunks.json"
        combined_data = [asdict(c) for c in combined]
        combined_path.write_text(json.dumps(combined_data, indent=2), encoding='utf-8')
        logger.info(f"Saved {len(combined)} total chunks to: {combined_path}")

        return all_chunks


def main():
    """Test chunking with all documents."""
    print("=" * 60)
    print("Document Chunker")
    print("=" * 60)
    print(f"\nChunk size: ~{CHUNK_SIZE} tokens")
    print(f"Overlap: ~{CHUNK_OVERLAP} tokens")
    print("-" * 60)

    chunker = TextChunker()
    all_chunks = chunker.chunk_all_documents()

    print("\n" + "=" * 60)
    print("Chunking Summary")
    print("=" * 60)

    total = 0
    for ticker, chunks in all_chunks.items():
        print(f"{ticker}: {len(chunks):>4} chunks")
        total += len(chunks)

    print("-" * 60)
    print(f"Total: {total} chunks")

    # Show a sample chunk
    if all_chunks:
        first_ticker = list(all_chunks.keys())[0]
        sample = all_chunks[first_ticker][5]  # Get 6th chunk (more interesting)
        print(f"\nSample chunk ({sample.id}):")
        print(f"  Company: {sample.company}")
        print(f"  Section: {sample.section}")
        print(f"  Filing date: {sample.filing_date}")
        print(f"  Text preview: {sample.text[:200]}...")


if __name__ == "__main__":
    main()
