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

def detect_hallucination(raw_answer, grounded_answer, chunks_text):
    """
    Detect hallucination type based on Wang's paper framework.

    Returns:
        tuple: (hallucination_type, confidence, explanation)
        - hallucination_type: "none", "deviation", "fabrication", "uncertain"
        - confidence: float 0-1
        - explanation: str
    """
    if not raw_answer or not grounded_answer:
        return ("uncertain", 0.5, "Could not compare responses")

    raw_lower = raw_answer.lower()
    grounded_lower = grounded_answer.lower()

    # Check if grounded answer says info not available
    no_info_phrases = [
        "do not contain",
        "not contain information",
        "cannot find",
        "no information",
        "not mentioned",
        "not specified"
    ]

    grounded_says_no_info = any(phrase in grounded_lower for phrase in no_info_phrases)

    # Check if raw LLM expresses uncertainty
    uncertainty_phrases = [
        "i don't have",
        "i'm not certain",
        "i cannot confirm",
        "uncertain",
        "not sure",
        "may vary",
        "approximately",
        "around",
        "roughly"
    ]

    raw_expresses_uncertainty = any(phrase in raw_lower for phrase in uncertainty_phrases)

    # Extract numbers from both
    raw_numbers = extract_numbers(raw_answer)
    grounded_numbers = extract_numbers(grounded_answer)

    # Case 1: Grounded says no info, but Raw gives specific answer = FABRICATION
    if grounded_says_no_info and raw_numbers and not raw_expresses_uncertainty:
        return (
            "fabrication",
            0.9,
            "⚠️ **FABRICATION**: Raw LLM provided specific information that doesn't exist in source documents."
        )

    # Case 2: Both have numbers but they differ significantly = DEVIATION
    if raw_numbers and grounded_numbers:
        # Compare the most prominent numbers
        raw_main = max(raw_numbers) if raw_numbers else 0
        grounded_main = max(grounded_numbers) if grounded_numbers else 0

        if grounded_main > 0:
            deviation = abs(raw_main - grounded_main) / grounded_main

            if deviation > 0.1:  # More than 10% off
                return (
                    "deviation",
                    min(0.9, deviation),
                    f"⚠️ **DEVIATION**: Raw LLM number differs by {deviation*100:.1f}% from grounded source."
                )
            elif deviation > 0.01:  # 1-10% off
                return (
                    "minor_deviation",
                    deviation,
                    f"⚡ **Minor Deviation**: Numbers differ by {deviation*100:.1f}% (rounding difference)."
                )

    # Case 3: Raw expresses uncertainty = HONEST
    if raw_expresses_uncertainty:
        return (
            "honest_uncertainty",
            0.3,
            "✅ **Honest**: Raw LLM appropriately expressed uncertainty."
        )

    # Case 4: Answers seem consistent
    return (
        "consistent",
        0.2,
        "✅ **Consistent**: Responses appear aligned."
    )


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

    # Detect hallucination
    chunks_text = ""
    if result.rag_only:
        chunks_text = " ".join([c.text for c in result.rag_only.chunks])

    hall_type, hall_confidence, hall_explanation = detect_hallucination(
        result.raw_llm.answer if result.raw_llm else "",
        result.rag_plus_llm.answer if result.rag_plus_llm else "",
        chunks_text
    )

    # Three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Raw LLM
    with col1:
        st.markdown('<div class="column-header raw-llm-header">🧠 Raw LLM</div>', unsafe_allow_html=True)
        st.markdown("*Direct query - no documents*")

        if result.raw_llm:
            st.markdown(f'<span class="warning-badge">⚠️ May contain inaccuracies</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="timing">⏱️ {result.raw_llm.elapsed_time:.2f}s</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(result.raw_llm.answer)
        else:
            st.error("Failed - Check API key")
            if result.errors:
                st.caption(str([e for e in result.errors if "Raw LLM" in e]))

    # Column 2: RAG Only
    with col2:
        st.markdown('<div class="column-header rag-only-header">📄 RAG Only</div>', unsafe_allow_html=True)
        st.markdown("*Retrieved chunks - read the source*")

        if result.rag_only:
            st.markdown(f'<span class="info-badge">📚 {len(result.rag_only.chunks)} chunks retrieved</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="timing">⏱️ {result.rag_only.elapsed_time:.2f}s</div>', unsafe_allow_html=True)
            st.markdown("---")

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
        st.markdown("*Grounded in SEC filings*")

        if result.rag_plus_llm:
            st.markdown(f'<span class="success-badge">✅ Grounded in documents</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="timing">⏱️ {result.rag_plus_llm.elapsed_time:.2f}s (retrieval: {result.rag_plus_llm.retrieval_time:.2f}s, LLM: {result.rag_plus_llm.llm_time:.2f}s)</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(result.rag_plus_llm.answer)

            with st.expander("📎 Source chunks used"):
                for i, chunk in enumerate(result.rag_plus_llm.chunks):
                    meta = chunk.metadata
                    st.markdown(f"**{i+1}. {meta['company']}** ({meta['filing_date']})")
                    st.caption(chunk.text[:200] + "...")
        else:
            st.error("Failed - Check API key")
            if result.errors:
                st.caption(str([e for e in result.errors if "RAG + LLM" in e]))

    # ========================================================================
    # Hallucination Analysis Section (Wang Paper Style)
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🔍 Hallucination Analysis (Wang Paper Framework)")

    # Determine color based on type
    if hall_type in ["fabrication"]:
        box_class = "fabrication"
        icon = "🚨"
    elif hall_type in ["deviation", "minor_deviation"]:
        box_class = "deviation"
        icon = "⚠️"
    else:
        box_class = "consistent"
        icon = "✅"

    st.markdown(f"""
    <div class="hallucination-box {box_class}">
        <h4>{icon} {hall_type.upper().replace('_', ' ')}</h4>
        <p>{hall_explanation}</p>
        <p><strong>Confidence:</strong> {hall_confidence*100:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Explanation of types
    with st.expander("ℹ️ What do these terms mean? (from Wang's paper)"):
        st.markdown("""
        **Based on Wang (UT Austin) - "F(r)iction in Machines: Accounting Hallucinations of LLMs":**

        | Type | Definition | Example |
        |------|------------|---------|
        | **Fabrication** | LLM invents information that doesn't exist | Giving revenue for a pre-IPO company when that data wasn't public |
        | **Deviation** | LLM gives wrong values for real data | Saying revenue was $4.5B when it was actually $4.81B |
        | **Honest Uncertainty** | LLM appropriately admits it doesn't know | "I'm not certain about the exact figure..." |
        | **Consistent** | Raw LLM matches grounded answer | Both give the same factual answer |

        *Key finding from Wang's paper: Even the best prompting strategies result in 48% average deviation and 36% fabrication rate for pre-IPO data.*
        """)

    # Summary metrics
    st.markdown("### 📊 Summary")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric(
            "Raw LLM",
            f"{result.raw_llm.elapsed_time:.2f}s" if result.raw_llm else "Failed",
        )
    with summary_cols[1]:
        st.metric(
            "RAG Only",
            f"{result.rag_only.elapsed_time:.2f}s" if result.rag_only else "Failed",
            f"{len(result.rag_only.chunks)} chunks" if result.rag_only else None
        )
    with summary_cols[2]:
        st.metric(
            "RAG + LLM",
            f"{result.rag_plus_llm.elapsed_time:.2f}s" if result.rag_plus_llm else "Failed",
        )
    with summary_cols[3]:
        st.metric(
            "Hallucination Risk",
            hall_type.replace('_', ' ').title(),
            f"{hall_confidence*100:.0f}%" if hall_type in ["fabrication", "deviation"] else "Low",
            delta_color="inverse" if hall_type in ["fabrication", "deviation"] else "off"
        )

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
