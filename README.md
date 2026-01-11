# RAG vs Raw LLM Demo

A demonstration comparing Raw LLM queries vs Retrieval-Augmented Generation (RAG) for accounting/financial information retrieval.

**Purpose**: Live demo for AAA 2024 conference discussion of Wang (UT Austin) paper on LLM hallucinations in accounting.

## Quick Start

### 1. Set up environment variables

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Run the demo (if index already built)

```bash
streamlit run app/streamlit_app.py
```

Or:
```bash
python run_app.py
```

## Full Setup (First Time)

If starting fresh, run these scripts in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download S-1 filings from SEC EDGAR
python scripts/01_download_filings.py

# 3. Process documents (HTML -> clean text)
python scripts/02_process_documents.py

# 4. Build FAISS vector index (takes ~2 minutes)
python scripts/03_build_index.py

# 5. Test the pipeline
python scripts/04_test_queries.py

# 6. Run the web UI
streamlit run app/streamlit_app.py
```

## Project Structure

```
rag_demo/
├── app/
│   └── streamlit_app.py     # Web UI
├── data/
│   ├── raw/                 # Downloaded S-1 HTML files
│   └── processed/           # Cleaned text and chunks
├── index/
│   └── faiss_index/         # Vector search index
├── scripts/
│   ├── 00_test_imports.py   # Verify dependencies
│   ├── 01_download_filings.py
│   ├── 02_process_documents.py
│   ├── 03_build_index.py
│   └── 04_test_queries.py
├── src/
│   ├── edgar_downloader.py  # SEC EDGAR API
│   ├── document_parser.py   # HTML parsing
│   ├── chunker.py           # Text chunking
│   ├── embeddings.py        # Sentence transformers
│   ├── vector_store.py      # FAISS operations
│   ├── retriever.py         # Query interface
│   ├── llm_client.py        # Claude API
│   └── pipeline.py          # Orchestration
├── config.py                # Configuration
├── requirements.txt
├── run_app.py
└── README.md
```

## How It Works

```
User Query: "What was Airbnb's revenue in 2019?"
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────────┐
│ Raw LLM │   │RAG Only │   │ RAG + LLM   │
│         │   │         │   │             │
│ ~$4.5B  │   │ Chunks  │   │ $4.81B per  │
│ (wrong?)│   │ from    │   │ S-1 filing  │
│         │   │ S-1     │   │ (correct)   │
└─────────┘   └─────────┘   └─────────────┘
```

### Three Approaches Compared

1. **Raw LLM**: Asks Claude directly from its training data. May hallucinate.
2. **RAG Only**: Retrieves relevant chunks from S-1 filings. User reads raw text.
3. **RAG + LLM**: Retrieves chunks and has Claude synthesize an answer. Grounded.

## Companies Included

| Ticker | Company | IPO Year |
|--------|---------|----------|
| ABNB | Airbnb | 2020 |
| DASH | DoorDash | 2020 |
| SNOW | Snowflake | 2020 |
| RIVN | Rivian | 2021 |
| PLTR | Palantir | 2020 |

## Sample Questions

- "What was Airbnb's total revenue in fiscal year 2019?"
- "How many Dashers were on DoorDash's platform at IPO?"
- "What are Snowflake's main risk factors?"
- "What was Rivian's net loss in 2020?"
- "Who were Palantir's largest customers?"

## Requirements

- Python 3.10+
- ~4GB disk space (for models and data)
- Anthropic API key (for LLM calls)

## Technical Details

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS (IndexFlatIP with L2 normalization for cosine similarity)
- **Chunks**: ~500 tokens with 50 token overlap
- **Total Chunks**: ~3,076 across 5 companies
- **LLM**: Claude claude-sonnet-4-20250514 (via Anthropic API)

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
- Make sure `.env` file exists with your API key

**"Index files not found"**
- Run `python scripts/03_build_index.py` first

**Slow first query**
- Model loading takes ~15s on first query, then it's cached

---

*Demo for AAA 2024 Conference*
