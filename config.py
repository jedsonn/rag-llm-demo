"""
Configuration settings for RAG vs Raw LLM Demo
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Directory Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = PROJECT_ROOT / "index" / "faiss_index"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API Keys (loaded from environment or Streamlit secrets)
# =============================================================================
def get_secret(key, default=None):
    """Get secret from environment or Streamlit secrets."""
    # First try environment variable
    value = os.getenv(key)
    if value:
        return value
    # Then try Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return default

ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SEC_EDGAR_USER_AGENT = get_secret("SEC_EDGAR_USER_AGENT", "RAGDemo/1.0 (demo@university.edu)")

# =============================================================================
# SEC EDGAR Settings
# =============================================================================
SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_RATE_LIMIT = 10  # requests per second (SEC limit)
SEC_REQUEST_DELAY = 0.15  # seconds between requests (slightly under limit for safety)

# Target companies for the demo
COMPANIES = {
    "ABNB": {"name": "Airbnb", "cik": "0001559720"},
    "DASH": {"name": "DoorDash", "cik": "0001792789"},
    "SNOW": {"name": "Snowflake", "cik": "0001640147"},
    "RIVN": {"name": "Rivian", "cik": "0001874178"},
    "PLTR": {"name": "Palantir", "cik": "0001321655"},
}

# =============================================================================
# Document Processing Settings
# =============================================================================
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Key sections to extract from S-1 filings
S1_SECTIONS = [
    "BUSINESS",
    "RISK FACTORS",
    "SELECTED FINANCIAL DATA",
    "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "FINANCIAL STATEMENTS",
    "USE OF PROCEEDS",
]

# =============================================================================
# Embedding Settings
# =============================================================================
# Using sentence-transformers (free, local) by default
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Alternative: OpenAI embeddings (better quality, costs money)
# EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_DIMENSION = 1536

# =============================================================================
# Retrieval Settings
# =============================================================================
TOP_K_CHUNKS = 5  # Number of chunks to retrieve per query

# =============================================================================
# LLM Settings
# =============================================================================
LLM_MODEL = "gpt-4.1-mini"  # OpenAI model for responses (cost-effective)
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0  # Zero temperature for deterministic, factual responses

# =============================================================================
# UI Settings
# =============================================================================
STREAMLIT_PAGE_TITLE = "RAG vs Raw LLM: Accounting Information Retrieval Demo"
STREAMLIT_PAGE_ICON = "📊"

# =============================================================================
# Logging
# =============================================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
