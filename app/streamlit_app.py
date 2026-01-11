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

    # Sample questions
    SAMPLE_QUESTIONS = [
        "What was Airbnb's total revenue in fiscal year 2019?",
        "How many Dashers were on DoorDash's platform at IPO?",
        "What are Snowflake's main risk factors?",
        "What was Rivian's net loss in 2020?",
        "Who were Palantir's largest customers?",
    ]

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

    st.markdown("##### Try a sample question:")
    sample_cols = st.columns(5)
    short_labels = ["Airbnb Revenue", "DoorDash Dashers", "Snowflake Risks", "Rivian Loss", "Palantir Customers"]
    for i, sample_q in enumerate(SAMPLE_QUESTIONS):
        with sample_cols[i]:
            if st.button(short_labels[i], key=f"sample_{i}", help=sample_q, use_container_width=True):
                st.session_state.selected_question = sample_q

    default_q = st.session_state.selected_question if st.session_state.selected_question else ""
    question = st.text_input(
        "Or type your own question:",
        value=default_q,
        placeholder="e.g., What was Airbnb's revenue in 2019?",
        key="tab1_question"
    )

    col_spacer, col_button, col_spacer2 = st.columns([3, 1, 3])
    with col_button:
        compare_button = st.button("🔍 Compare", type="primary", use_container_width=True, key="tab1_btn")

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

    st.markdown("##### Sample questions:")
    model_sample_cols = st.columns(3)
    model_samples = [
        "What was Airbnb's total revenue in fiscal year 2019?",
        "What was DoorDash's net loss in 2019?",
        "How many vehicles had Rivian produced by IPO?"
    ]
    model_sample_labels = ["Airbnb Revenue", "DoorDash Loss", "Rivian Vehicles"]

    if "model_question" not in st.session_state:
        st.session_state.model_question = ""

    for i, (sample_q, label) in enumerate(zip(model_samples, model_sample_labels)):
        with model_sample_cols[i]:
            if st.button(label, key=f"model_sample_{i}", help=sample_q, use_container_width=True):
                st.session_state.model_question = sample_q

    model_default_q = st.session_state.model_question if st.session_state.model_question else ""
    query2 = st.text_input(
        "Enter question (will query GPT, Gemini, and Llama in parallel):",
        value=model_default_q,
        placeholder="What was Airbnb's total revenue in fiscal 2019?",
        key="tab2_query"
    )

    col_s1, col_btn, col_s2 = st.columns([3, 1, 3])
    with col_btn:
        model_compare_btn = st.button("🔬 Compare Models", type="primary", use_container_width=True, key="tab2_btn")

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
        st.markdown("### 📊 External Validity Analysis")

        if ensemble['success'] and ensemble.get('spread_pct') is not None:
            spread = ensemble['spread_pct']

            if spread > 15:
                st.error(f"""
                **⚠️ High Model Disagreement ({spread:.1f}% spread)**

                Different LLMs give substantially different answers to the same factual question.
                This supports the critique that Wang's findings may not generalize across model families.
                """)
            elif spread > 5:
                st.warning(f"""
                **⚡ Moderate Model Disagreement ({spread:.1f}% spread)**

                Models show some variation in their answers. This suggests model-specific training
                data and architectures affect factual recall.
                """)
            else:
                st.success(f"""
                **✅ Models Agree ({spread:.1f}% spread)**

                All models converge on similar answers, suggesting this is well-established information
                in training data across model families.
                """)

        with st.expander("ℹ️ About this comparison"):
            st.markdown("""
            **Models tested:**
            - **GPT-4.1-mini** (OpenAI) - Same family as Wang's paper
            - **Gemini 2.0 Flash** (Google) - Different architecture
            - **Llama 3.3 70B** (Meta via Groq) - Open source

            **Why this matters:**
            Wang's paper tests only OpenAI models. If different model families give different answers,
            the hallucination rates may be model-specific rather than universal LLM behavior.
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
st.caption("AAA 2024 Conference Discussion  •  Wang (UT Austin) - \"F(r)iction in Machines\"  •  Multi-model comparison")
