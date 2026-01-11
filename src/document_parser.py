"""
Document Parser for SEC S-1 Filings

Parses HTML S-1 filings from SEC EDGAR and extracts clean text.
Handles tables, removes boilerplate, and identifies key sections.
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup, NavigableString

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, S1_SECTIONS, logger


class DocumentParser:
    """
    Parser for SEC S-1 filings (HTML format).

    S-1 filings have specific structure:
    - Table of Contents at the beginning
    - Key sections: Business, Risk Factors, Financial Data, MD&A, etc.
    - Financial tables embedded in HTML
    """

    def __init__(self):
        # Patterns for section headers in S-1 filings
        self.section_patterns = [
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?BUSINESS",
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?RISK\s*FACTORS",
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?SELECTED\s*(?:CONSOLIDATED\s*)?FINANCIAL\s*DATA",
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?MANAGEMENT.{0,5}S?\s*DISCUSSION\s*AND\s*ANALYSIS",
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?FINANCIAL\s*STATEMENTS",
            r"(?:ITEM\s*\d+[A-Z]?\.?\s*)?USE\s*OF\s*PROCEEDS",
            r"PROSPECTUS\s*SUMMARY",
            r"THE\s*OFFERING",
        ]

    def parse_html(self, filepath: Path) -> str:
        """
        Parse HTML file and extract clean text.

        Args:
            filepath: Path to HTML file

        Returns:
            Clean text content
        """
        logger.info(f"Parsing: {filepath}")

        html_content = filepath.read_text(encoding='utf-8', errors='replace')

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()

        # Process tables specially to preserve structure
        self._process_tables(soup)

        # Get text
        text = soup.get_text(separator='\n')

        # Clean the text
        text = self._clean_text(text)

        logger.info(f"Extracted {len(text):,} characters of text")

        return text

    def _process_tables(self, soup: BeautifulSoup):
        """
        Process tables to preserve their structure in text form.
        Converts tables to readable text format.
        """
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text(strip=True)
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    rows.append(' | '.join(cells))

            if rows:
                table_text = '\n'.join(rows)
                table.replace_with(NavigableString(f"\n[TABLE]\n{table_text}\n[/TABLE]\n"))
            else:
                table.decompose()

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text:
        - Normalize whitespace
        - Remove excessive blank lines
        - Fix common encoding issues
        """
        # Fix common encoding issues
        text = text.replace('\xa0', ' ')  # Non-breaking spaces
        text = text.replace('\u2019', "'")  # Smart quotes
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2014', '-')  # Em dash

        # Normalize whitespace within lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces to single space
            line = re.sub(r' +', ' ', line)
            line = line.strip()
            cleaned_lines.append(line)

        # Join and collapse multiple blank lines
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove lines that are just page numbers or navigation
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that are just numbers (page numbers)
            if re.match(r'^\d+$', line.strip()):
                continue
            # Skip "Table of Contents" navigation links
            if line.strip().lower() == 'table of contents':
                continue
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def extract_sections(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Identify key sections in the document.

        Args:
            text: Clean document text

        Returns:
            Dict mapping section name to {start, end, text}
        """
        sections = {}
        text_upper = text.upper()

        for pattern in self.section_patterns:
            matches = list(re.finditer(pattern, text_upper))
            if matches:
                # Take the first substantive match (skip TOC entries)
                for match in matches:
                    start = match.start()

                    # Skip if this is likely in the table of contents
                    # (TOC entries are usually short lines)
                    line_start = text.rfind('\n', 0, start) + 1
                    line_end = text.find('\n', start)
                    if line_end == -1:
                        line_end = len(text)
                    line = text[line_start:line_end]

                    # If the line is long enough, it's probably a real section header
                    if len(line) < 100:  # TOC entries are usually short
                        # Check if there's substantial content after
                        next_500 = text[start:start+500]
                        if len(next_500) > 200:  # Has content after
                            section_name = self._normalize_section_name(match.group())
                            sections[section_name] = {
                                'start': start,
                                'header': match.group()
                            }
                            break

        # Calculate end positions (next section or end of document)
        section_list = sorted(sections.items(), key=lambda x: x[1]['start'])
        for i, (name, info) in enumerate(section_list):
            if i + 1 < len(section_list):
                info['end'] = section_list[i + 1][1]['start']
            else:
                info['end'] = len(text)
            info['text'] = text[info['start']:info['end']]
            info['length'] = len(info['text'])

        return sections

    def _normalize_section_name(self, header: str) -> str:
        """Normalize section header to standard name."""
        header = header.upper().strip()

        if 'BUSINESS' in header and 'RISK' not in header:
            return 'BUSINESS'
        elif 'RISK' in header:
            return 'RISK FACTORS'
        elif 'FINANCIAL DATA' in header or 'SELECTED' in header:
            return 'SELECTED FINANCIAL DATA'
        elif 'DISCUSSION' in header or 'MD&A' in header:
            return 'MD&A'
        elif 'FINANCIAL STATEMENTS' in header:
            return 'FINANCIAL STATEMENTS'
        elif 'PROCEEDS' in header:
            return 'USE OF PROCEEDS'
        elif 'SUMMARY' in header:
            return 'PROSPECTUS SUMMARY'
        elif 'OFFERING' in header:
            return 'THE OFFERING'
        else:
            return header

    def process_filing(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Process a single S-1 filing end-to-end.

        Args:
            ticker: Company ticker (e.g., "ABNB")

        Returns:
            Dict with processed data, or None if failed
        """
        input_path = RAW_DATA_DIR / f"{ticker}_S-1.html"

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return None

        # Parse HTML
        text = self.parse_html(input_path)

        # Extract sections
        sections = self.extract_sections(text)

        # Save full text
        text_output = PROCESSED_DATA_DIR / f"{ticker}_S-1.txt"
        text_output.write_text(text, encoding='utf-8')
        logger.info(f"Saved text to: {text_output}")

        # Save sections metadata (without full text to keep it small)
        sections_meta = {
            name: {
                'start': info['start'],
                'end': info['end'],
                'length': info['length'],
                'header': info['header']
            }
            for name, info in sections.items()
        }
        sections_output = PROCESSED_DATA_DIR / f"{ticker}_S-1_sections.json"
        sections_output.write_text(json.dumps(sections_meta, indent=2), encoding='utf-8')
        logger.info(f"Saved sections to: {sections_output}")

        return {
            'ticker': ticker,
            'text_length': len(text),
            'text_path': str(text_output),
            'sections': sections_meta,
            'sections_path': str(sections_output)
        }


def main():
    """Test the parser with Airbnb filing."""
    print("=" * 60)
    print("Document Parser Test")
    print("=" * 60)

    parser = DocumentParser()

    # Test with Airbnb
    test_ticker = "ABNB"
    print(f"\nProcessing {test_ticker} S-1 filing...")

    result = parser.process_filing(test_ticker)

    if result:
        print(f"\n[SUCCESS] Processed {test_ticker}")
        print(f"  Text length: {result['text_length']:,} characters")
        print(f"  Text file: {result['text_path']}")
        print(f"\nSections found:")
        for name, info in result['sections'].items():
            print(f"  - {name}: {info['length']:,} chars")

        # Show a sample of the text
        print(f"\nFirst 1000 characters of extracted text:")
        print("-" * 60)
        text = Path(result['text_path']).read_text(encoding='utf-8')
        print(text[:1000])
        print("-" * 60)
    else:
        print(f"\n[FAILED] Could not process {test_ticker}")


if __name__ == "__main__":
    main()
