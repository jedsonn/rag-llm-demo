"""
Script 01: Download S-1 filings from SEC EDGAR

Downloads S-1 (pre-IPO registration) filings for all configured companies.
These filings contain detailed financial and business information before companies went public.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.edgar_downloader import EDGARDownloader
from config import COMPANIES, RAW_DATA_DIR


def main():
    print("=" * 60)
    print("SEC EDGAR S-1 Filing Downloader")
    print("=" * 60)
    print(f"\nDownloading S-1 filings for {len(COMPANIES)} companies:")
    for ticker, info in COMPANIES.items():
        print(f"  - {info['name']} ({ticker})")
    print(f"\nOutput directory: {RAW_DATA_DIR}")
    print("-" * 60)

    downloader = EDGARDownloader()
    results = downloader.download_all_companies()

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    success_count = 0
    for ticker, path in results.items():
        status = "[OK]" if path else "[FAILED]"
        if path:
            success_count += 1
            file_size = Path(path).stat().st_size / 1024 / 1024  # MB
            print(f"{status} {ticker}: {file_size:.1f} MB")
        else:
            print(f"{status} {ticker}")

    print("-" * 60)
    print(f"Downloaded: {success_count}/{len(COMPANIES)} filings")

    # List files in raw directory
    print(f"\nFiles in {RAW_DATA_DIR}:")
    for f in sorted(RAW_DATA_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
