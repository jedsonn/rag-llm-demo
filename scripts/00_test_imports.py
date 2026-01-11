"""
Test script to verify all required dependencies are installed correctly.
Run this after installing requirements.txt to make sure everything works.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    errors = []

    print("Testing imports...")
    print("-" * 50)

    # Web requests and HTML parsing
    try:
        import requests
        print(f"[OK] requests {requests.__version__}")
    except ImportError as e:
        errors.append(f"requests: {e}")
        print(f"[FAIL] requests")

    try:
        import bs4
        print(f"[OK] beautifulsoup4 {bs4.__version__}")
    except ImportError as e:
        errors.append(f"beautifulsoup4: {e}")
        print(f"[FAIL] beautifulsoup4")

    try:
        import lxml
        print(f"[OK] lxml {lxml.__version__}")
    except ImportError as e:
        errors.append(f"lxml: {e}")
        print(f"[FAIL] lxml")

    # Vector store and numerical
    try:
        import faiss
        print(f"[OK] faiss-cpu")
    except ImportError as e:
        errors.append(f"faiss-cpu: {e}")
        print(f"[FAIL] faiss-cpu")

    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"[FAIL] numpy")

    # Embeddings
    try:
        import sentence_transformers
        print(f"[OK] sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        errors.append(f"sentence-transformers: {e}")
        print(f"[FAIL] sentence-transformers")

    # LLM APIs
    try:
        import anthropic
        print(f"[OK] anthropic {anthropic.__version__}")
    except ImportError as e:
        errors.append(f"anthropic: {e}")
        print(f"[FAIL] anthropic")

    try:
        import openai
        print(f"[OK] openai {openai.__version__}")
    except ImportError as e:
        # OpenAI is optional
        print(f"[SKIP] openai (optional)")

    # Web UI
    try:
        import streamlit
        print(f"[OK] streamlit {streamlit.__version__}")
    except ImportError as e:
        errors.append(f"streamlit: {e}")
        print(f"[FAIL] streamlit")

    # Utilities
    try:
        import dotenv
        print(f"[OK] python-dotenv")
    except ImportError as e:
        errors.append(f"python-dotenv: {e}")
        print(f"[FAIL] python-dotenv")

    try:
        import tqdm
        print(f"[OK] tqdm {tqdm.__version__}")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
        print(f"[FAIL] tqdm")

    try:
        import pandas as pd
        print(f"[OK] pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"pandas: {e}")
        print(f"[FAIL] pandas")

    print("-" * 50)

    if errors:
        print(f"\n[ERROR] {len(errors)} import(s) failed:")
        for err in errors:
            print(f"  - {err}")
        print("\nRun: pip install -r requirements.txt")
        return False
    else:
        print("\n[SUCCESS] All imports OK!")
        return True


def test_config():
    """Test that config.py loads correctly."""
    print("\nTesting config.py...")
    print("-" * 50)

    try:
        # Add parent directory to path so we can import config
        sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0])
        import config

        print(f"[OK] Project root: {config.PROJECT_ROOT}")
        print(f"[OK] Data directory: {config.DATA_DIR}")
        print(f"[OK] Embedding model: {config.EMBEDDING_MODEL}")
        print(f"[OK] LLM model: {config.LLM_MODEL}")
        print(f"[OK] Companies configured: {list(config.COMPANIES.keys())}")

        # Check if API keys are set
        if config.ANTHROPIC_API_KEY:
            print(f"[OK] ANTHROPIC_API_KEY is set")
        else:
            print(f"[WARN] ANTHROPIC_API_KEY not set (needed for LLM calls)")

        print("-" * 50)
        print("\n[SUCCESS] Config loaded OK!")
        return True

    except Exception as e:
        print(f"[FAIL] Could not load config: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Demo - Dependency Check")
    print("=" * 50)

    imports_ok = test_imports()
    config_ok = test_config()

    print("\n" + "=" * 50)
    if imports_ok and config_ok:
        print("All checks passed! Ready to proceed to Phase 2.")
    else:
        print("Some checks failed. Please fix issues before proceeding.")
    print("=" * 50)
