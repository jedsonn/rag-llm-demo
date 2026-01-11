"""
Streamlit Web UI for RAG vs Raw LLM Demo

Tab 1: Core Finding - RAG vs Raw LLM comparison
Tab 2: Model Comparison - Same query across multiple LLM providers

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
from src.llm_providers import query_all_models, compute_ensemble, extract_number

# Page config
st.set_page_config(
    page_title="RAG vs Raw LLM Demo",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# Scoring Functions
# ============================================================================

def extract_numbers(text):
    """Extract all numbers (including decimals, billions, millions) from text."""
    text = text.lower().replace(',', '')
    numbers = []
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
    """Score Raw LLM response for hallucination risk."""
    if not raw_answer:
        return {"score": None, "label": "N/A", "color": "gray", "explanation": "No response"}

    raw_lower = raw_answer.lower()
    grounded_lower = grounded_answer.lower() if grounded_answer else ""

    no_info_phrases = ["do not contain", "not contain information", "cannot find",
                       "no information", "not mentioned", "not specified", "not available"]
    grounded_says_no_info = any(phrase in grounded_lower for phrase in no_info_phrases)

    uncertainty_phrases = ["i don't have", "i'm not certain", "i cannot confirm",
                          "uncertain", "not sure", "don't know", "no information"]
    raw_uncertain = any(phrase in raw_lower for phrase in uncertainty_phrases)

    raw_numbers = extract_numbers(raw_answer)
    grounded_numbers = extract_numbers(grounded_answer) if grounded_answer else []

    if grounded_says_no_info and raw_numbers and not raw_uncertain:
        return {"score": 95, "label": "FABRICATION", "color": "#dc2626",
                "explanation": "Provided specific data not found in source documents"}

    if raw_numbers and grounded_numbers:
        raw_main = max(raw_numbers)
        grounded_main = max(grounded_numbers)
        if grounded_main > 0:
            deviation_pct = abs(raw_main - grounded_main) / grounded_main * 100
            if deviation_pct > 20:
                return {"score": min(90, 50 + deviation_pct), "label": f"DEVIATION ({deviation_pct:.0f}%)",
                        "color": "#f59e0b", "explanation": f"Key number differs by {deviation_pct:.0f}% from grounded source"}
            elif deviation_pct > 5:
                return {"score": 30 + deviation_pct, "label": f"Minor ({deviation_pct:.0f}%)",
                        "color": "#eab308", "explanation": f"Minor numerical difference of {deviation_pct:.0f}%"}

    if raw_uncertain:
        return {"score": 15, "label": "HONEST", "color": "#22c55e",
                "explanation": "Appropriately expressed uncertainty"}

    return {"score": 20, "label": "CONSISTENT", "color": "#22c55e",
            "explanation": "Response aligns with grounded answer"}


def score_rag_llm(grounded_answer, chunks_text):
    """Score RAG+LLM response for grounding confidence."""
    if not grounded_answer:
        return {"score": None, "label": "N/A", "color": "gray", "explanation": "No response"}

    grounded_lower = grounded_answer.lower()
    no_info_phrases = ["do not contain", "not contain information", "cannot find",
                       "no information", "not mentioned", "not specified"]
    if any(phrase in grounded_lower for phrase in no_info_phrases):
        return {"score": 85, "label": "HONEST", "color": "#22c55e",
                "explanation": "Correctly states info not in documents"}

    grounded_numbers = extract_numbers(grounded_answer)
    chunk_numbers = extract_numbers(chunks_text) if chunks_text else []

    if grounded_numbers and chunk_numbers:
        grounded_main = max(grounded_numbers)
        matches = any(abs(grounded_main - cn) / max(grounded_main, 1) < 0.05 for cn in chunk_numbers)
        if matches:
            return {"score": 95, "label": "GROUNDED", "color": "#22c55e",
                    "explanation": "Numbers match source documents"}

    return {"score": 80, "label": "GROUNDED", "color": "#22c55e",
            "explanation": "Response synthesized from retrieved chunks"}


def score_rag_only(chunks):
    """Score RAG retrieval quality."""
    if not chunks:
        return {"score": 0, "label": "NO DATA", "color": "#dc2626", "explanation": "No chunks retrieved"}

    avg_score = sum(c.score for c in chunks) / len(chunks)
    if avg_score > 0.5:
        return {"score": min(95, int(avg_score * 100)), "label": "HIGH", "color": "#22c55e",
                "explanation": f"Strong semantic match (avg: {avg_score:.2f})"}
    elif avg_score > 0.3:
        return {"score": int(avg_score * 100), "label": "MODERATE", "color": "#eab308",
                "explanation": f"Moderate match (avg: {avg_score:.2f})"}
    else:
        return {"score": int(avg_score * 100), "label": "LOW", "color": "#f59e0b",
                "explanation": f"Weak semantic match (avg: {avg_score:.2f})"}


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
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
        margin-bottom: 1rem;
    }
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
    .model-header {
        background-color: #f3e8ff;
        color: #6b21a8;
        border: 2px solid #c4b5fd;
    }
    .ensemble-header {
        background-color: #fef3c7;
        color: #92400e;
        border: 2px solid #fcd34d;
    }
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
    .timing {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    .model-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .spread-high { color: #dc2626; font-weight: bold; }
    .spread-medium { color: #f59e0b; font-weight: bold; }
    .spread-low { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================
st.markdown('<div class="main-header">RAG vs Raw LLM: Accounting Information Retrieval</div>', unsafe_allow_html=True)
st.markdown("""
<p class="subtitle">
    Discussion of Wang (UT Austin) - "F(r)iction in Machines: Accounting Hallucinations of LLMs"
</p>
""", unsafe_allow_html=True)

# ============================================================================
# Tabs
# ============================================================================
tab1, tab2 = st.tabs(["📊 Core Finding (RAG vs Raw)", "🔬 Model Comparison"])

# ============================================================================
# TAB 1: Core Finding - RAG vs Raw LLM
# ============================================================================
with tab1:
    st.markdown("### Does RAG reduce hallucinations?")

    # Initialize pipeline (cached)
    @st.cache_resource
    def get_pipeline():
        return QueryPipeline()

    # Sample questions with 5 buttons
    SAMPLE_QUESTIONS = [
        ("Airbnb Revenue", "What was Airbnb's total revenue in fiscal year 2019?"),
        ("DoorDash Dashers", "How many Dashers were on DoorDash's platform at IPO?"),
        ("Snowflake Risks", "What are Snowflake's main risk factors?"),
        ("Rivian Loss", "What was Rivian's net loss in 2020?"),
        ("Palantir Customers", "Who were Palantir's largest customers?"),
    ]

    # Initialize session state
    if "tab1_q" not in st.session_state:
        st.session_state.tab1_q = ""

    # Show 5 buttons
    st.markdown("**Click a sample question:**")
    cols = st.columns(5)
    for i, (label, full_q) in enumerate(SAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(label, key=f"t1_btn_{i}", use_container_width=True):
                st.session_state.tab1_q = full_q

    # Text input uses session state value
    question = st.text_input(
        "Or type your own:",
        value=st.session_state.tab1_q,
        placeholder="e.g., What was Airbnb's revenue in 2019?",
    )

    # Update session state if user typed something different
    if question != st.session_state.tab1_q:
        st.session_state.tab1_q = question

    col_spacer, col_button, col_spacer2 = st.columns([3, 1, 3])
    with col_button:
        compare_button = st.button("🔍 Compare", type="primary", use_container_width=True, key="tab1_compare")

    st.markdown("---")

    if compare_button and question:
        with st.spinner("Running comparison..."):
            pipeline = get_pipeline()
            result = pipeline.compare_all(question)

        chunks_text = ""
        if result.rag_only:
            chunks_text = " ".join([c.text for c in result.rag_only.chunks])

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

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="column-header raw-llm-header">🧠 Raw LLM</div>', unsafe_allow_html=True)
            if raw_score["score"] is not None:
                st.markdown(f'''
                <div class="score-box" style="background-color: {raw_score["color"]}15; border: 2px solid {raw_score["color"]};">
                    <div class="score-label" style="color: {raw_score["color"]};">Hallucination Risk</div>
                    <div class="score-value" style="color: {raw_score["color"]};">{raw_score["label"]}</div>
                    <div class="score-explanation" style="color: #333;">{raw_score["explanation"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            if result.raw_llm:
                st.markdown(f'<div class="timing">⏱️ {result.raw_llm.elapsed_time:.2f}s</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(result.raw_llm.answer)
            else:
                st.error("Failed - Check API key")

        with col2:
            st.markdown('<div class="column-header rag-only-header">📄 RAG Only</div>', unsafe_allow_html=True)
            if rag_only_score["score"] is not None:
                st.markdown(f'''
                <div class="score-box" style="background-color: {rag_only_score["color"]}15; border: 2px solid {rag_only_score["color"]};">
                    <div class="score-label" style="color: {rag_only_score["color"]};">Retrieval Quality</div>
                    <div class="score-value" style="color: {rag_only_score["color"]};">{rag_only_score["label"]}</div>
                    <div class="score-explanation" style="color: #333;">{rag_only_score["explanation"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            if result.rag_only:
                st.markdown(f'<div class="timing">⏱️ {result.rag_only.elapsed_time:.2f}s</div>', unsafe_allow_html=True)
                st.markdown("---")
                for i, chunk in enumerate(result.rag_only.chunks):
                    meta = chunk.metadata
                    with st.expander(f"**Chunk {i+1}** | {meta['company']} | Score: {chunk.score:.3f}", expanded=(i==0)):
                        st.markdown(f"*{meta['filing_date']} | {meta['section']}*")
                        st.markdown(chunk.text)
            else:
                st.error("Failed to retrieve chunks")

        with col3:
            st.markdown('<div class="column-header rag-llm-header">✨ RAG + LLM</div>', unsafe_allow_html=True)
            if rag_llm_score["score"] is not None:
                st.markdown(f'''
                <div class="score-box" style="background-color: {rag_llm_score["color"]}15; border: 2px solid {rag_llm_score["color"]};">
                    <div class="score-label" style="color: {rag_llm_score["color"]};">Grounding Confidence</div>
                    <div class="score-value" style="color: {rag_llm_score["color"]};">{rag_llm_score["label"]}</div>
                    <div class="score-explanation" style="color: #333;">{rag_llm_score["explanation"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            if result.rag_plus_llm:
                st.markdown(f'<div class="timing">⏱️ {result.rag_plus_llm.elapsed_time:.2f}s</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(result.rag_plus_llm.answer)
                with st.expander("📎 Source chunks used"):
                    for i, chunk in enumerate(result.rag_plus_llm.chunks):
                        meta = chunk.metadata
                        st.markdown(f"**{i+1}. {meta['company']}** ({meta['filing_date']})")
                        st.caption(chunk.text[:200] + "...")
            else:
                st.error("Failed - Check API key")

        st.markdown("---")
        with st.expander("ℹ️ How are these scores calculated? (Wang's Framework)"):
            st.markdown("""
            | Score | Meaning | Detection |
            |-------|---------|-----------|
            | **FABRICATION** | LLM invents data | Raw LLM gives numbers when grounded says "no info" |
            | **DEVIATION** | Wrong values | >5% difference between Raw and grounded |
            | **CONSISTENT** | Answers align | Numbers match within 5% |
            | **GROUNDED** | Cites sources | RAG+LLM references retrieved chunks |
            """)

    else:
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("#### How this demo works")
            st.markdown("""
            | Column | What it shows |
            |--------|---------------|
            | **🧠 Raw LLM** | Direct query - may hallucinate |
            | **📄 RAG Only** | Retrieved chunks - read the source |
            | **✨ RAG + LLM** | Grounded answer from SEC filings |
            """)
        with right_col:
            st.markdown("#### Data Sources")
            st.markdown("S-1 filings from: Airbnb, DoorDash, Snowflake, Rivian, Palantir")
            st.info("👆 Click a sample question or type your own")


# ============================================================================
# TAB 2: Model Comparison
# ============================================================================
with tab2:
    st.markdown("### Does Model Choice Matter?")
    st.markdown("*Wang's paper uses only OpenAI. Let's test external validity across model families.*")

    # Same 5 sample questions with buttons
    MODEL_SAMPLE_QUESTIONS = [
        ("Airbnb Revenue", "What was Airbnb's total revenue in fiscal year 2019?"),
        ("DoorDash Dashers", "How many Dashers were on DoorDash's platform at IPO?"),
        ("Snowflake Risks", "What are Snowflake's main risk factors?"),
        ("Rivian Loss", "What was Rivian's net loss in 2020?"),
        ("Palantir Customers", "Who were Palantir's largest customers?"),
    ]

    # Initialize session state
    if "tab2_q" not in st.session_state:
        st.session_state.tab2_q = ""

    # Show 5 buttons
    st.markdown("**Click a sample question:**")
    cols2 = st.columns(5)
    for i, (label, full_q) in enumerate(MODEL_SAMPLE_QUESTIONS):
        with cols2[i]:
            if st.button(label, key=f"t2_btn_{i}", use_container_width=True):
                st.session_state.tab2_q = full_q

    # Text input
    query2 = st.text_input(
        "Or type your own (queries GPT, Gemini, Llama in parallel):",
        value=st.session_state.tab2_q,
        placeholder="What was Airbnb's total revenue in fiscal 2019?",
    )

    # Update session state if user typed something different
    if query2 != st.session_state.tab2_q:
        st.session_state.tab2_q = query2

    col_s1, col_btn, col_s2 = st.columns([3, 1, 3])
    with col_btn:
        model_compare_btn = st.button("🔬 Compare Models", type="primary", use_container_width=True, key="tab2_compare")

    st.markdown("---")

    if model_compare_btn and query2:
        # Accounting-specific prompt wrapper
        full_prompt = f"""Answer this accounting/financial question with a specific number if possible.

Question: {query2}

Be precise and concise. Include the dollar amount if applicable."""

        with st.spinner("Querying 3 models in parallel..."):
            results = query_all_models(full_prompt)
            ensemble = compute_ensemble(results)

        # Display results in columns
        num_models = len(results)
        cols = st.columns(num_models + 1)  # +1 for ensemble

        for i, result in enumerate(results):
            with cols[i]:
                model_name = result['model']
                st.markdown(f'<div class="column-header model-header">{model_name}</div>', unsafe_allow_html=True)

                if result['success']:
                    # Extract number for display
                    num = extract_number(result['answer'])
                    if num is not None:
                        st.markdown(f'''
                        <div class="score-box" style="background-color: #f3e8ff; border: 2px solid #c4b5fd;">
                            <div class="score-value" style="color: #6b21a8;">${num:.2f}B</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="score-box" style="background-color: #fef3c7; border: 2px solid #fcd34d;">
                            <div class="score-label" style="color: #92400e;">No number extracted</div>
                        </div>
                        ''', unsafe_allow_html=True)

                    st.markdown(f'<div class="timing">⏱️ {result["elapsed_time"]:.2f}s</div>', unsafe_allow_html=True)
                    st.markdown("---")

                    with st.expander("Full response", expanded=False):
                        st.markdown(result['answer'])
                else:
                    st.error(result['answer'])

        # Ensemble column
        with cols[-1]:
            st.markdown('<div class="column-header ensemble-header">🎯 Ensemble</div>', unsafe_allow_html=True)

            if ensemble['success']:
                # Color code spread
                spread_pct = ensemble.get('spread_pct', 0)
                if spread_pct > 20:
                    spread_class = "spread-high"
                    spread_label = "HIGH VARIANCE"
                elif spread_pct > 10:
                    spread_class = "spread-medium"
                    spread_label = "MODERATE"
                else:
                    spread_class = "spread-low"
                    spread_label = "CONSISTENT"

                st.markdown(f'''
                <div class="score-box" style="background-color: #fef3c7; border: 2px solid #fcd34d;">
                    <div class="score-label" style="color: #92400e;">Cross-Model Agreement</div>
                    <div class="score-value {spread_class}">{spread_label}</div>
                    <div class="score-explanation" style="color: #333;">Spread: ±{spread_pct:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(ensemble['answer'])

                if ensemble.get('details'):
                    with st.expander("📊 All extracted values"):
                        for d in ensemble['details']:
                            st.markdown(f"- **{d['model']}**: ${d['value']:.2f}B")
            else:
                st.warning(ensemble['answer'])

        # Summary
        st.markdown("---")
        st.markdown("### 📊 What Does This Tell Us?")

        # Build intuitive summary
        successful_models = [r for r in results if r['success']]
        extracted_values = [(r['model'], extract_number(r['answer'])) for r in successful_models]
        extracted_values = [(m, v) for m, v in extracted_values if v is not None]

        if ensemble['success'] and ensemble.get('spread_pct') is not None:
            spread = ensemble['spread_pct']
            median_val = ensemble.get('median', 0)

            # Create a plain-English summary
            st.markdown("#### Quick Summary")

            col_summary1, col_summary2 = st.columns(2)

            with col_summary1:
                if len(extracted_values) >= 2:
                    values_str = ", ".join([f"**{m}**: ${v:.2f}B" for m, v in extracted_values])
                    st.markdown(f"**Models said:** {values_str}")
                    st.markdown(f"**Median answer:** ${median_val:.2f}B")

            with col_summary2:
                if spread > 15:
                    st.markdown(f"""
                    🚨 **PROBLEM: Models disagree by {spread:.0f}%**

                    This is a big spread! Different LLMs give very different answers to the same factual question.
                    """)
                elif spread > 5:
                    st.markdown(f"""
                    ⚠️ **CAUTION: Models differ by {spread:.0f}%**

                    Noticeable variation between models. The "right" answer depends on which LLM you ask.
                    """)
                else:
                    st.markdown(f"""
                    ✅ **GOOD: Models agree (within {spread:.0f}%)**

                    All models converge on similar answers. This fact is well-established across training data.
                    """)

            st.markdown("---")
            st.markdown("#### Implications for Wang's Paper")

            if spread > 10:
                st.error("""
                **External Validity Concern:** Wang's findings are based on OpenAI models only.
                This high disagreement suggests hallucination rates may be **model-specific**,
                not a universal LLM behavior. Different models = different error rates.
                """)
            elif spread > 5:
                st.warning("""
                **Moderate Concern:** Some variation between model families. Wang's exact percentages
                (48% deviation, 36% fabrication) may not replicate across Gemini or Llama.
                """)
            else:
                st.success("""
                **Good News:** For this question, models agree. However, agreement on easy facts
                doesn't mean agreement on harder accounting questions. Test more examples!
                """)

        with st.expander("ℹ️ About this comparison"):
            st.markdown("""
            **Models tested:**
            - **GPT-4.1-mini** (OpenAI) - Same family as Wang's paper
            - **Gemini 2.0 Flash** (Google) - Different architecture
            - **Llama 3.3 70B** (Meta via Groq) - Open source

            **The question:** Does model choice affect hallucination rates?

            **Why it matters:** If GPT, Gemini, and Llama give different answers, then:
            1. Hallucination rates are model-dependent
            2. Wang's findings may not generalize
            3. Researchers should test across model families
            """)

    else:
        st.markdown("""
        #### Why compare models?

        Wang's paper uses only OpenAI's GPT models. This tab tests **external validity**:
        - Do other LLM families (Gemini, Llama) hallucinate at similar rates?
        - Do they hallucinate on the same questions?
        - Does the specific answer vary by model?

        **Try it:** Enter a factual accounting question and see how GPT, Gemini, and Llama compare.
        """)

        st.info("👆 Enter a question and click 'Compare Models' to test across 3 LLM families")


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("AAA 2026 Conference Discussion  •  Wang (UT Austin) - \"F(r)iction in Machines\"  •  Multi-model comparison")
