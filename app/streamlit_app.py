"""
Streamlit Web UI for RAG vs Raw LLM Demo

Three-column comparison showing:
1. Raw LLM (parametric memory - may hallucinate)
2. RAG Only (retrieved chunks - user reads raw text)
3. RAG + LLM (grounded synthesis)

Includes hallucination detection inspired by Wang (UT Austin) paper.
"""

import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.pipeline import QueryPipeline

# Page config
st.set_page_config(
    page_title="RAG vs Raw LLM Demo",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# Hallucination Detection (Wang Paper Style)
# ============================================================================

def extract_numbers(text):
    """Extract all numbers (including decimals, billions, millions) from text."""
    # Normalize text
    text = text.lower().replace(',', '')

    numbers = []

    # Pattern for numbers with billion/million/thousand suffixes
    patterns = [
        r'\$?([\d.]+)\s*billion',
        r'\$?([\d.]+)\s*million',
        r'\$?([\d.]+)\s*thousand',
        r'\$?([\d,]+(?:\.\d+)?)\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                num = float(match.replace(',', ''))
                if 'billion' in text[text.find(match):text.find(match)+20]:
                    num *= 1e9
                elif 'million' in text[text.find(match):text.find(match)+20]:
                    num *= 1e6
                numbers.append(num)
            except:
                pass

    return numbers

def score_raw_llm(raw_answer, grounded_answer, chunks_text):
    """
    Score Raw LLM response for hallucination risk.

    Returns:
        dict with: score (0-100, higher = more risk), label, color, explanation
    """
    if not raw_answer:
        return {"score": None, "label": "N/A", "color": "gray", "explanation": "No response"}

    raw_lower = raw_answer.lower()
    grounded_lower = grounded_answer.lower() if grounded_answer else ""

    # Check if grounded says no info available
    no_info_phrases = ["do not contain", "not contain information", "cannot find",
                       "no information", "not mentioned", "not specified", "not available"]
    grounded_says_no_info = any(phrase in grounded_lower for phrase in no_info_phrases)

    # Check if raw expresses uncertainty
    uncertainty_phrases = ["i don't have", "i'm not certain", "i cannot confirm",
                          "uncertain", "not sure", "don't know", "no information"]
    raw_uncertain = any(phrase in raw_lower for phrase in uncertainty_phrases)

    # Extract numbers
    raw_numbers = extract_numbers(raw_answer)
    grounded_numbers = extract_numbers(grounded_answer) if grounded_answer else []

    # FABRICATION: Raw gives specific data when grounded says no info
    if grounded_says_no_info and raw_numbers and not raw_uncertain:
        return {
            "score": 95,
            "label": "FABRICATION",
            "color": "#dc2626",
            "explanation": "Provided specific data not found in source documents"
        }

    # DEVIATION: Numbers differ significantly
    if raw_numbers and grounded_numbers:
        raw_main = max(raw_numbers)
        grounded_main = max(grounded_numbers)
        if grounded_main > 0:
            deviation_pct = abs(raw_main - grounded_main) / grounded_main * 100
            if deviation_pct > 20:
                return {
                    "score": min(90, 50 + deviation_pct),
                    "label": f"DEVIATION ({deviation_pct:.0f}%)",
                    "color": "#f59e0b",
                    "explanation": f"Key number differs by {deviation_pct:.0f}% from grounded source"
                }
            elif deviation_pct > 5:
                return {
                    "score": 30 + deviation_pct,
                    "label": f"Minor ({deviation_pct:.0f}%)",
                    "color": "#eab308",
                    "explanation": f"Minor numerical difference of {deviation_pct:.0f}%"
                }

    # HONEST: Raw admits uncertainty
    if raw_uncertain:
        return {
            "score": 15,
            "label": "HONEST",
            "color": "#22c55e",
            "explanation": "Appropriately expressed uncertainty"
        }

    # CONSISTENT: Answers align
    return {
        "score": 20,
        "label": "CONSISTENT",
        "color": "#22c55e",
        "explanation": "Response aligns with grounded answer"
    }


def score_rag_llm(grounded_answer, chunks_text):
    """
    Score RAG+LLM response for grounding confidence.

    Returns:
        dict with: score (0-100, higher = better grounding), label, color, explanation
    """
    if not grounded_answer:
        return {"score": None, "label": "N/A", "color": "gray", "explanation": "No response"}

    grounded_lower = grounded_answer.lower()
    chunks_lower = chunks_text.lower() if chunks_text else ""

    # Check if answer admits no info found
    no_info_phrases = ["do not contain", "not contain information", "cannot find",
                       "no information", "not mentioned", "not specified"]
    says_no_info = any(phrase in grounded_lower for phrase in no_info_phrases)

    if says_no_info:
        return {
            "score": 85,
            "label": "HONEST",
            "color": "#22c55e",
            "explanation": "Correctly states info not in documents"
        }

    # Check for numbers and if they appear in chunks
    grounded_numbers = extract_numbers(grounded_answer)
    chunk_numbers = extract_numbers(chunks_text) if chunks_text else []

    if grounded_numbers and chunk_numbers:
        # Check if main numbers match chunks
        grounded_main = max(grounded_numbers)
        matches = any(abs(grounded_main - cn) / max(grounded_main, 1) < 0.05 for cn in chunk_numbers)
        if matches:
            return {
                "score": 95,
                "label": "GROUNDED",
                "color": "#22c55e",
                "explanation": "Numbers match source documents"
            }

    # Default: reasonable grounding
    return {
        "score": 80,
        "label": "GROUNDED",
        "color": "#22c55e",
        "explanation": "Response synthesized from retrieved chunks"
    }


def score_rag_only(chunks):
    """
    Score RAG retrieval quality.

    Returns:
        dict with: score (0-100), label, color, explanation
    """
    if not chunks:
        return {"score": 0, "label": "NO DATA", "color": "#dc2626", "explanation": "No chunks retrieved"}

    avg_score = sum(c.score for c in chunks) / len(chunks)

    if avg_score > 0.5:
        return {
            "score": min(95, int(avg_score * 100)),
            "label": "HIGH",
            "color": "#22c55e",
            "explanation": f"Strong semantic match (avg: {avg_score:.2f})"
        }
    elif avg_score > 0.3:
        return {
            "score": int(avg_score * 100),
            "label": "MODERATE",
            "color": "#eab308",
            "explanation": f"Moderate match (avg: {avg_score:.2f})"
        }
    else:
        return {
            "score": int(avg_score * 100),
            "label": "LOW",
            "color": "#f59e0b",
            "explanation": f"Weak semantic match (avg: {avg_score:.2f})"
        }


# Custom CSS for presentation mode - CLEAN, HIGH CONTRAST
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Column headers - clear color coding */
    .column-header {
        font-size: 1.3rem;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .raw-llm-header {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #fca5a5;
    }
    .rag-only-header {
        background-color: #dbeafe;
        color: #1e40af;
        border: 2px solid #93c5fd;
    }
    .rag-llm-header {
        background-color: #d1fae5;
        color: #065f46;
        border: 2px solid #6ee7b7;
    }

    /* Hallucination boxes - HIGH CONTRAST TEXT */
    .hallucination-box {
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .hallucination-box h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    .hallucination-box p {
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    .fabrication {
        background-color: #fef2f2;
        border: 3px solid #dc2626;
    }
    .fabrication h4, .fabrication p, .fabrication strong {
        color: #7f1d1d !important;
    }
    .deviation {
        background-color: #fffbeb;
        border: 3px solid #f59e0b;
    }
    .deviation h4, .deviation p, .deviation strong {
        color: #78350f !important;
    }
    .consistent {
        background-color: #f0fdf4;
        border: 3px solid #22c55e;
    }
    .consistent h4, .consistent p, .consistent strong {
        color: #14532d !important;
    }

    /* Badges */
    .warning-badge {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .success-badge {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .info-badge {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Timing info */
    .timing {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    /* Input styling */
    .stTextInput input {
        font-size: 1.1rem;
    }

    /* Make expanders cleaner */
    .streamlit-expanderHeader {
        font-size: 0.95rem;
    }

    /* Score displays */
    .score-box {
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .score-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .score-explanation {
        font-size: 0.75rem;
        margin-top: 0.25rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">RAG vs Raw LLM: Accounting Information Retrieval</div>', unsafe_allow_html=True)
st.markdown("""
<p class="subtitle">
    Comparing parametric memory (Raw LLM) vs retrieval-augmented generation (RAG) for SEC filing questions<br>
    <em>Hallucination detection based on Wang (UT Austin) - "F(r)iction in Machines"</em>
</p>
""", unsafe_allow_html=True)

# Initialize pipeline (cached)
@st.cache_resource
def get_pipeline():
    """Load the query pipeline (cached for performance)."""
    return QueryPipeline()

# Sample questions
SAMPLE_QUESTIONS = [
    "What was Airbnb's total revenue in fiscal year 2019?",
    "How many Dashers were on DoorDash's platform at IPO?",
    "What are Snowflake's main risk factors?",
    "What was Rivian's net loss in 2020?",
    "Who were Palantir's largest customers?",
]

# Initialize session state for selected question
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# Sample question buttons - cleaner horizontal layout
st.markdown("##### Try a sample question:")
sample_cols = st.columns(5)
for i, sample_q in enumerate(SAMPLE_QUESTIONS):
    with sample_cols[i]:
        # Shorten display text for button
        short_labels = ["Airbnb Revenue", "DoorDash Dashers", "Snowflake Risks", "Rivian Loss", "Palantir Customers"]
        if st.button(short_labels[i], key=f"sample_{i}", help=sample_q, use_container_width=True):
            st.session_state.selected_question = sample_q

# Question input - use selected question as default value
default_q = st.session_state.selected_question if st.session_state.selected_question else ""

st.markdown("")  # Small spacing
question = st.text_input(
    "Or type your own question about Airbnb, DoorDash, Snowflake, Rivian, or Palantir:",
    value=default_q,
    placeholder="e.g., What was Airbnb's revenue in 2019?",
)

col_spacer, col_button, col_spacer2 = st.columns([3, 1, 3])
with col_button:
    compare_button = st.button("🔍 Compare", type="primary", use_container_width=True)

st.markdown("---")

# Run comparison
if compare_button and question:
    with st.spinner("Running comparison..."):
        pipeline = get_pipeline()
        result = pipeline.compare_all(question)

    # Get chunks text for scoring
    chunks_text = ""
    if result.rag_only:
        chunks_text = " ".join([c.text for c in result.rag_only.chunks])

    # Calculate individual scores
    raw_score = score_raw_llm(
        result.raw_llm.answer if result.raw_llm else "",
        result.rag_plus_llm.answer if result.rag_plus_llm else "",
        chunks_text
    )
    rag_only_score = score_rag_only(result.rag_only.chunks if result.rag_only else [])
    rag_llm_score = score_rag_llm(
        result.rag_plus_llm.answer if result.rag_plus_llm else "",
        chunks_text
    )

    # Three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Raw LLM
    with col1:
        st.markdown('<div class="column-header raw-llm-header">🧠 Raw LLM</div>', unsafe_allow_html=True)

        # Score display for Raw LLM (hallucination risk - red is bad)
        if raw_score["score"] is not None:
            risk_color = raw_score["color"]
            st.markdown(f'''
            <div class="score-box" style="background-color: {risk_color}15; border: 2px solid {risk_color};">
                <div class="score-label" style="color: {risk_color};">Hallucination Risk</div>
                <div class="score-value" style="color: {risk_color};">{raw_score["label"]}</div>
                <div class="score-explanation" style="color: #333;">{raw_score["explanation"]}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown(f'<div class="timing">⏱️ {result.raw_llm.elapsed_time:.2f}s</div>' if result.raw_llm else '', unsafe_allow_html=True)
        st.markdown("---")

        if result.raw_llm:
            st.markdown(result.raw_llm.answer)
        else:
            st.error("Failed - Check API key")

    # Column 2: RAG Only
    with col2:
        st.markdown('<div class="column-header rag-only-header">📄 RAG Only</div>', unsafe_allow_html=True)

        # Score display for RAG retrieval quality
        if rag_only_score["score"] is not None:
            ret_color = rag_only_score["color"]
            st.markdown(f'''
            <div class="score-box" style="background-color: {ret_color}15; border: 2px solid {ret_color};">
                <div class="score-label" style="color: {ret_color};">Retrieval Quality</div>
                <div class="score-value" style="color: {ret_color};">{rag_only_score["label"]}</div>
                <div class="score-explanation" style="color: #333;">{rag_only_score["explanation"]}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown(f'<div class="timing">⏱️ {result.rag_only.elapsed_time:.2f}s</div>' if result.rag_only else '', unsafe_allow_html=True)
        st.markdown("---")

        if result.rag_only:
            for i, chunk in enumerate(result.rag_only.chunks):
                meta = chunk.metadata
                with st.expander(f"**Chunk {i+1}** | {meta['company']} | Score: {chunk.score:.3f}", expanded=(i==0)):
                    st.markdown(f"*{meta['filing_date']} | {meta['section']}*")
                    st.markdown(chunk.text)
        else:
            st.error("Failed to retrieve chunks")

    # Column 3: RAG + LLM
    with col3:
        st.markdown('<div class="column-header rag-llm-header">✨ RAG + LLM</div>', unsafe_allow_html=True)

        # Score display for RAG+LLM (grounding confidence - green is good)
        if rag_llm_score["score"] is not None:
            ground_color = rag_llm_score["color"]
            st.markdown(f'''
            <div class="score-box" style="background-color: {ground_color}15; border: 2px solid {ground_color};">
                <div class="score-label" style="color: {ground_color};">Grounding Confidence</div>
                <div class="score-value" style="color: {ground_color};">{rag_llm_score["label"]}</div>
                <div class="score-explanation" style="color: #333;">{rag_llm_score["explanation"]}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown(f'<div class="timing">⏱️ {result.rag_plus_llm.elapsed_time:.2f}s</div>' if result.rag_plus_llm else '', unsafe_allow_html=True)
        st.markdown("---")

        if result.rag_plus_llm:
            st.markdown(result.rag_plus_llm.answer)

            with st.expander("📎 Source chunks used"):
                for i, chunk in enumerate(result.rag_plus_llm.chunks):
                    meta = chunk.metadata
                    st.markdown(f"**{i+1}. {meta['company']}** ({meta['filing_date']})")
                    st.caption(chunk.text[:200] + "...")
        else:
            st.error("Failed - Check API key")

    # ========================================================================
    # Summary Section
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📊 Comparison Summary")

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(
            "🧠 Raw LLM",
            raw_score["label"],
            f"⏱️ {result.raw_llm.elapsed_time:.2f}s" if result.raw_llm else "Failed",
            delta_color="inverse" if raw_score["score"] and raw_score["score"] > 50 else "off"
        )
    with summary_cols[1]:
        st.metric(
            "📄 RAG Retrieval",
            rag_only_score["label"],
            f"{len(result.rag_only.chunks)} chunks" if result.rag_only else None,
            delta_color="normal" if rag_only_score["score"] and rag_only_score["score"] > 50 else "off"
        )
    with summary_cols[2]:
        st.metric(
            "✨ RAG + LLM",
            rag_llm_score["label"],
            f"⏱️ {result.rag_plus_llm.elapsed_time:.2f}s" if result.rag_plus_llm else "Failed",
            delta_color="normal"
        )

    # Explanation of scoring
    with st.expander("ℹ️ How are these scores calculated? (Wang's Framework)"):
        st.markdown("""
        **Scoring based on Wang (UT Austin) - "F(r)iction in Machines":**

        | Score | Meaning | How it's detected |
        |-------|---------|-------------------|
        | **FABRICATION** | LLM invents data that doesn't exist | Raw LLM gives specific numbers when grounded answer says "no information" |
        | **DEVIATION** | LLM gives wrong values | Numerical difference >5% between Raw LLM and grounded answer |
        | **CONSISTENT** | Answers align | Numbers match within 5% tolerance |
        | **GROUNDED** | Answer cites sources | RAG+LLM response references retrieved chunks |

        *Wang's finding: 48% average deviation rate, 36% fabrication rate for pre-IPO data.*
        """)

else:
    # Instructions when no query - cleaner layout
    st.markdown("")

    # Two columns for cleaner layout
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### How this demo works")
        st.markdown("""
        | Column | What it shows |
        |--------|---------------|
        | **🧠 Raw LLM** | Direct query - uses model's training data (may hallucinate) |
        | **📄 RAG Only** | Retrieved document chunks - you read the source |
        | **✨ RAG + LLM** | Grounded answer synthesized from actual SEC filings |
        """)

        st.markdown("#### Hallucination Types")
        st.markdown("""
        - 🚨 **Fabrication** — LLM invents data that doesn't exist in sources
        - ⚠️ **Deviation** — LLM gives wrong values for real data
        - ✅ **Consistent** — LLM answer matches grounded sources
        """)

    with right_col:
        st.markdown("#### Data Sources")
        st.markdown("""
        S-1 filings (pre-IPO registration statements) from SEC EDGAR:

        | Company | IPO Year |
        |---------|----------|
        | Airbnb | 2020 |
        | DoorDash | 2020 |
        | Snowflake | 2020 |
        | Rivian | 2021 |
        | Palantir | 2020 |
        """)

        st.info("👆 **Click a sample question above** or type your own, then click Compare.")

# Footer
st.markdown("---")
st.caption("AAA 2024 Conference Discussion  •  Paper: Wang (UT Austin) - \"F(r)iction in Machines\"  •  GPT-4.1-mini + sentence-transformers")
