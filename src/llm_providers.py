"""
Multi-Model LLM Providers for Model Comparison Tab

Supports: Claude (Anthropic), Gemini (Google), Llama (via Groq)
"""

import os
import re
import time
import concurrent.futures
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# API Configuration (loaded from environment or hardcoded for demo)
# =============================================================================

def get_secret(key: str, default: str = None) -> Optional[str]:
    """Get secret from environment or Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return default

# API Keys - loaded from environment or Streamlit secrets
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# =============================================================================
# Model Query Functions
# =============================================================================

def query_claude(question: str, timeout: float = 15.0) -> Dict:
    """Query Claude (Anthropic)"""
    start_time = time.time()
    try:
        from anthropic import Anthropic

        if not ANTHROPIC_API_KEY:
            return {
                "answer": "No Anthropic API key configured",
                "model": "Claude",
                "success": False,
                "elapsed_time": 0
            }

        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}]
        )

        elapsed = time.time() - start_time
        return {
            "answer": response.content[0].text,
            "model": "Claude",
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Claude error: {e}")
        return {
            "answer": f"Error: {str(e)[:100]}",
            "model": "Claude",
            "success": False,
            "elapsed_time": elapsed
        }


def query_gemini(question: str, timeout: float = 15.0) -> Dict:
    """Query Gemini (Google)"""
    start_time = time.time()
    try:
        import google.generativeai as genai

        if not GEMINI_API_KEY:
            return {
                "answer": "No Gemini API key configured",
                "model": "Gemini",
                "success": False,
                "elapsed_time": 0
            }

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(question)

        elapsed = time.time() - start_time
        return {
            "answer": response.text,
            "model": "Gemini",
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Gemini error: {e}")
        return {
            "answer": f"Error: {str(e)[:100]}",
            "model": "Gemini",
            "success": False,
            "elapsed_time": elapsed
        }


def query_llama(question: str, timeout: float = 15.0) -> Dict:
    """Query Llama 3.3 70B via Groq"""
    start_time = time.time()
    try:
        from openai import OpenAI

        if not GROQ_API_KEY:
            return {
                "answer": "No Groq API key configured",
                "model": "Llama-3.3-70B",
                "success": False,
                "elapsed_time": 0
            }

        client = OpenAI(
            api_key=GROQ_API_KEY,
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


def query_gpt(question: str, timeout: float = 15.0) -> Dict:
    """Query GPT-4.1-mini (OpenAI) - uses existing config"""
    start_time = time.time()
    try:
        from openai import OpenAI

        openai_key = get_secret("OPENAI_API_KEY")
        if not openai_key:
            return {
                "answer": "No OpenAI API key configured",
                "model": "GPT-4.1-mini",
                "success": False,
                "elapsed_time": 0
            }

        client = OpenAI(api_key=openai_key)
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


# =============================================================================
# Parallel Query All Models
# =============================================================================

def query_all_models(question: str) -> List[Dict]:
    """Query all available models in parallel"""

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(query_gpt, question): "gpt",
            executor.submit(query_gemini, question): "gemini",
            executor.submit(query_llama, question): "llama",
        }

        # Add Claude only if API key is available
        if ANTHROPIC_API_KEY:
            futures[executor.submit(query_claude, question)] = "claude"

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
    model_order = ["GPT-4.1-mini", "Claude", "Gemini", "Llama-3.3-70B"]
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
        # $4.81 billion, $4.81B
        (r'\$\s*([\d.]+)\s*(?:billion|b)\b', 1.0),
        # $4,810 million, $4810M
        (r'\$\s*([\d.]+)\s*(?:million|m)\b', 0.001),
        # 4.81 billion dollars
        (r'([\d.]+)\s*(?:billion)\s*(?:dollars?)?', 1.0),
        # 4810 million dollars
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

    # Extract numbers from successful responses
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

    # Calculate average
    values = [n['value'] for n in numbers]
    avg = sum(values) / len(values)

    # Calculate standard deviation
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5

    # Find median (more robust to outliers)
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = sorted_values[mid]

    # Find closest to median
    closest = min(numbers, key=lambda x: abs(x['value'] - median))

    # Calculate spread (max deviation from median)
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


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Quick test
    test_q = "What was Airbnb's total revenue in fiscal year 2019? Give a specific dollar amount."

    print("Testing individual models...")

    print("\n--- GPT ---")
    r = query_gpt(test_q)
    print(f"Success: {r['success']}, Time: {r['elapsed_time']:.2f}s")
    print(r['answer'][:200] if r['success'] else r['answer'])

    print("\n--- Gemini ---")
    r = query_gemini(test_q)
    print(f"Success: {r['success']}, Time: {r['elapsed_time']:.2f}s")
    print(r['answer'][:200] if r['success'] else r['answer'])

    print("\n--- Llama ---")
    r = query_llama(test_q)
    print(f"Success: {r['success']}, Time: {r['elapsed_time']:.2f}s")
    print(r['answer'][:200] if r['success'] else r['answer'])
