"""
SEC EDGAR Downloader
Downloads S-1 filings (pre-IPO registration statements) from SEC EDGAR.

SEC EDGAR API documentation: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEC_BASE_URL,
    SEC_EDGAR_USER_AGENT,
    SEC_REQUEST_DELAY,
    COMPANIES,
    RAW_DATA_DIR,
    logger
)


class EDGARDownloader:
    """
    Downloads SEC filings from EDGAR.

    SEC requires:
    - User-Agent header with contact information
    - Rate limiting (10 requests/second max)
    """

    def __init__(self, user_agent: Optional[str] = None):
        """Initialize the downloader with required headers."""
        self.user_agent = user_agent or SEC_EDGAR_USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure we don't exceed SEC's rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < SEC_REQUEST_DELAY:
            time.sleep(SEC_REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a rate-limited request to SEC EDGAR."""
        self._rate_limit()

        try:
            # Update host header based on URL
            if 'data.sec.gov' in url:
                self.session.headers['Host'] = 'data.sec.gov'
            else:
                self.session.headers['Host'] = 'www.sec.gov'

            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response

        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_company_filings(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Get all filings for a company from SEC EDGAR.

        Args:
            cik: Central Index Key (SEC company identifier)

        Returns:
            JSON data with company info and filings, or None if failed
        """
        # CIK must be 10 digits with leading zeros
        cik_padded = cik.lstrip('0').zfill(10)
        url = f"{SEC_BASE_URL}/submissions/CIK{cik_padded}.json"

        logger.info(f"Fetching company filings from: {url}")

        response = self._make_request(url)
        if response is None:
            return None

        return response.json()

    def find_s1_filing(self, filings_data: Dict[str, Any], cik: str) -> Optional[Dict[str, str]]:
        """
        Find the S-1 filing (or S-1/A amendment) from company filings.
        Searches both recent filings and older archived filings.

        Args:
            filings_data: JSON data from get_company_filings()
            cik: Company CIK (needed to fetch older filing files)

        Returns:
            Dict with filing info (accessionNumber, primaryDocument, filingDate)
            or None if not found
        """
        if 'filings' not in filings_data:
            logger.error("Unexpected filings data structure")
            return None

        # First check recent filings
        recent = filings_data['filings'].get('recent', {})
        result = self._search_filings_for_s1(recent)
        if result:
            return result

        # If not in recent, check older filing files
        # SEC stores older filings in separate JSON files
        files = filings_data['filings'].get('files', [])
        for file_info in files:
            file_name = file_info.get('name', '')
            if file_name:
                older_filings = self._fetch_older_filings(cik, file_name)
                if older_filings:
                    result = self._search_filings_for_s1(older_filings)
                    if result:
                        return result

        logger.warning("No S-1 filing found for this company")
        return None

    def _search_filings_for_s1(self, filings: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Search a filings dict for S-1 or S-1/A forms."""
        forms = filings.get('form', [])
        for i, form_type in enumerate(forms):
            if form_type in ['S-1', 'S-1/A']:
                return {
                    'accessionNumber': filings['accessionNumber'][i],
                    'primaryDocument': filings['primaryDocument'][i],
                    'filingDate': filings['filingDate'][i],
                    'form': form_type
                }
        return None

    def _fetch_older_filings(self, cik: str, filename: str) -> Optional[Dict[str, Any]]:
        """Fetch older filings from a separate JSON file."""
        cik_padded = cik.lstrip('0').zfill(10)
        url = f"{SEC_BASE_URL}/submissions/{filename}"

        logger.info(f"Fetching older filings from: {url}")

        response = self._make_request(url)
        if response is None:
            return None

        return response.json()

    def download_filing(self, cik: str, accession_number: str, document_name: str) -> Optional[str]:
        """
        Download a specific filing document.

        Args:
            cik: Central Index Key
            accession_number: Accession number (e.g., "0001193125-20-304852")
            document_name: Document filename (e.g., "d804422ds1.htm")

        Returns:
            Filing HTML content or None if failed
        """
        # Format accession number (remove dashes for URL)
        acc_no_formatted = accession_number.replace('-', '')
        cik_padded = cik.lstrip('0').zfill(10)

        url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_formatted}/{document_name}"

        logger.info(f"Downloading filing from: {url}")

        response = self._make_request(url)
        if response is None:
            return None

        return response.text

    def download_s1_for_company(self, ticker: str) -> Optional[str]:
        """
        Download the S-1 filing for a company.

        Args:
            ticker: Company ticker symbol (e.g., "ABNB")

        Returns:
            Path to saved file or None if failed
        """
        if ticker not in COMPANIES:
            logger.error(f"Unknown ticker: {ticker}")
            return None

        company = COMPANIES[ticker]
        cik = company['cik']
        name = company['name']

        logger.info(f"Downloading S-1 for {name} ({ticker}), CIK: {cik}")

        # Step 1: Get company filings
        filings_data = self.get_company_filings(cik)
        if filings_data is None:
            return None

        # Step 2: Find S-1 filing
        s1_info = self.find_s1_filing(filings_data, cik)
        if s1_info is None:
            return None

        logger.info(f"Found {s1_info['form']} filed on {s1_info['filingDate']}")

        # Step 3: Download the filing
        content = self.download_filing(
            cik,
            s1_info['accessionNumber'],
            s1_info['primaryDocument']
        )
        if content is None:
            return None

        # Step 4: Save to file
        output_path = RAW_DATA_DIR / f"{ticker}_S-1.html"
        output_path.write_text(content, encoding='utf-8')

        logger.info(f"Saved to: {output_path}")
        logger.info(f"File size: {len(content):,} bytes")

        # Also save metadata
        metadata_path = RAW_DATA_DIR / f"{ticker}_S-1_metadata.json"
        metadata = {
            'ticker': ticker,
            'company_name': name,
            'cik': cik,
            'form': s1_info['form'],
            'filingDate': s1_info['filingDate'],
            'accessionNumber': s1_info['accessionNumber'],
            'primaryDocument': s1_info['primaryDocument'],
            'file_path': str(output_path)
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        return str(output_path)

    def download_all_companies(self) -> Dict[str, Optional[str]]:
        """
        Download S-1 filings for all configured companies.

        Returns:
            Dict mapping ticker to file path (or None if failed)
        """
        results = {}

        for ticker in COMPANIES:
            try:
                path = self.download_s1_for_company(ticker)
                results[ticker] = path

                if path:
                    logger.info(f"[OK] {ticker}")
                else:
                    logger.warning(f"[FAIL] {ticker}")

            except Exception as e:
                logger.error(f"[ERROR] {ticker}: {e}")
                results[ticker] = None

        return results


def main():
    """Test the downloader with one company."""
    print("=" * 50)
    print("SEC EDGAR S-1 Downloader Test")
    print("=" * 50)

    downloader = EDGARDownloader()

    # Test with Airbnb first
    test_ticker = "ABNB"
    print(f"\nTesting download for {test_ticker}...")

    result = downloader.download_s1_for_company(test_ticker)

    if result:
        print(f"\n[SUCCESS] Downloaded to: {result}")

        # Show first 500 chars as sanity check
        content = Path(result).read_text(encoding='utf-8')
        print(f"\nFirst 500 characters:")
        print("-" * 50)
        print(content[:500])
        print("-" * 50)
    else:
        print("\n[FAILED] Could not download filing")


if __name__ == "__main__":
    main()
