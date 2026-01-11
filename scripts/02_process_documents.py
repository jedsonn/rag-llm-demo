"""
Script 02: Process downloaded S-1 filings

Parses HTML files and extracts clean text for each company.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_parser import DocumentParser
from config import COMPANIES, PROCESSED_DATA_DIR


def main():
    print("=" * 60)
    print("S-1 Document Processor")
    print("=" * 60)
    print(f"\nProcessing {len(COMPANIES)} company filings...")
    print(f"Output directory: {PROCESSED_DATA_DIR}")
    print("-" * 60)

    parser = DocumentParser()
    results = {}

    for ticker in COMPANIES:
        print(f"\nProcessing {ticker}...", end=" ", flush=True)
        try:
            result = parser.process_filing(ticker)
            if result:
                results[ticker] = result
                print(f"OK ({result['text_length']:,} chars)")
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)

    total_chars = 0
    for ticker, result in results.items():
        chars = result['text_length']
        total_chars += chars
        sections = list(result['sections'].keys())
        print(f"{ticker}: {chars:>10,} chars | Sections: {', '.join(sections[:3])}...")

    print("-" * 60)
    print(f"Total: {total_chars:,} characters across {len(results)} files")

    # List processed files
    print(f"\nFiles in {PROCESSED_DATA_DIR}:")
    for f in sorted(PROCESSED_DATA_DIR.glob("*.txt")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
