"""
Multi-Model LLM Providers for Model Comparison Tab

Supports: GPT (OpenAI), Gemini (Google), Llama (via Groq)
"""

import os
import re
import time
import concurrent.futures
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def get_secret(key: str) -> Optional[str]:
    """Get secret - check Streamlit secrets FIRST (for cloud), then env vars."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Then try environment variable (for local dev)
    value = os.getenv(key)
    if value:
        return value

    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        value = os.getenv(key)
        if value:
            return value
    except Exception:
        pass

    return None


# =============================================================================
# Model Query Functions - secrets loaded at call time, not import time
# =============================================================================

def query_gpt(question: str, timeout: float = 15.0) -> Dict:
    """Query GPT-4o-mini (OpenAI)"""
    start_time = time.time()
    try:
        from openai import OpenAI

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            return {
                "answer": "No OpenAI API key configured",
                "model": "GPT-4.1-mini",
                "success": False,
                "elapsed_time": 0
            }

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": question}],
            max_tokens=1024,
            temperature=0
        )

        elapsed = time.time() - start_time
        return {
            "answer": response.choices[0].message.content,
            "model": "GPT-4.1-mini",
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"GPT error: {e}")
        return {
            "answer": f"Error: {str(e)[:100]}",
            "model": "GPT-4.1-mini",
            "success": False,
            "elapsed_time": elapsed
        }


def query_gemini(question: str, timeout: float = 15.0) -> Dict:
    """Query Gemini 1.5 Flash (Google)"""
    start_time = time.time()
    try:
        import google.generativeai as genai

        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            return {
                "answer": "No Gemini API key configured",
                "model": "Gemini-2.0-Flash",
                "success": False,
                "elapsed_time": 0
            }

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(question)

        elapsed = time.time() - start_time
        return {
            "answer": response.text,
            "model": "Gemini-2.0-Flash",
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Gemini error: {e}")
        return {
            "answer": f"Error: {str(e)[:100]}",
            "model": "Gemini-2.0-Flash",
            "success": False,
            "elapsed_time": elapsed
        }


def query_llama(question: str, timeout: float = 15.0) -> Dict:
    """Query Llama 3.3 70B via Groq"""
    start_time = time.time()
    try:
        from openai import OpenAI

        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            return {
                "answer": "No Groq API key configured",
                "model": "Llama-3.3-70B",
                "success": False,
                "elapsed_time": 0
            }

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            max_tokens=1024,
            temperature=0
        )

        elapsed = time.time() - start_time
        return {
            "answer": response.choices[0].message.content,
            "model": "Llama-3.3-70B",
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Llama/Groq error: {e}")
        return {
            "answer": f"Error: {str(e)[:100]}",
            "model": "Llama-3.3-70B",
            "success": False,
            "elapsed_time": elapsed
        }


# =============================================================================
# Parallel Query All Models
# =============================================================================

def query_all_models(question: str) -> List[Dict]:
    """Query all available models in parallel"""

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(query_gpt, question): "gpt",
            executor.submit(query_gemini, question): "gemini",
            executor.submit(query_llama, question): "llama",
        }

        results = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                results.append(future.result())
            except Exception as e:
                model_name = futures[future]
                results.append({
                    "answer": f"Timeout or error: {e}",
                    "model": model_name,
                    "success": False,
                    "elapsed_time": 0
                })

    # Sort by model name for consistent display
    model_order = ["GPT-4.1-mini", "Gemini-2.0-Flash", "Llama-3.3-70B"]
    results.sort(key=lambda x: model_order.index(x["model"]) if x["model"] in model_order else 99)

    return results


# =============================================================================
# Number Extraction & Ensemble
# =============================================================================

def extract_number(text: str) -> Optional[float]:
    """Extract primary financial number from response (in billions)"""
    if not text:
        return None

    text_lower = text.lower().replace(',', '')

    # Patterns for financial amounts
    patterns = [
        (r'\$\s*([\d.]+)\s*(?:billion|b)\b', 1.0),
        (r'\$\s*([\d.]+)\s*(?:million|m)\b', 0.001),
        (r'([\d.]+)\s*(?:billion)\s*(?:dollars?)?', 1.0),
        (r'([\d.]+)\s*(?:million)\s*(?:dollars?)?', 0.001),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                num = float(match.group(1))
                return num * multiplier
            except:
                pass

    return None


def compute_ensemble(results: List[Dict]) -> Dict:
    """Compute ensemble result from multiple model responses"""

    numbers = []
    for r in results:
        if r.get('success'):
            num = extract_number(r['answer'])
            if num is not None:
                numbers.append({
                    "model": r['model'],
                    "value": num
                })

    if not numbers:
        return {
            "answer": "Could not extract numerical values from model responses",
            "model": "Ensemble",
            "success": False,
            "method": "N/A",
            "details": []
        }

    if len(numbers) == 1:
        return {
            "answer": f"${numbers[0]['value']:.2f}B (from {numbers[0]['model']} only)",
            "model": "Ensemble",
            "success": True,
            "method": "single",
            "details": numbers
        }

    values = [n['value'] for n in numbers]
    avg = sum(values) / len(values)

    variance = sum((v - avg) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5

    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = sorted_values[mid]

    closest = min(numbers, key=lambda x: abs(x['value'] - median))

    spread = max(abs(v - median) for v in values)
    spread_pct = (spread / median * 100) if median > 0 else 0

    return {
        "answer": f"**Median:** ${median:.2f}B\n**Average:** ${avg:.2f}B\n**Spread:** ±{spread_pct:.1f}%",
        "model": "Ensemble",
        "success": True,
        "method": "median",
        "median": median,
        "average": avg,
        "std_dev": std_dev,
        "spread_pct": spread_pct,
        "closest_model": closest['model'],
        "details": numbers
    }
