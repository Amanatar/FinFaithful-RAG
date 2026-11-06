"""
FinFaithful-RAG: Decoupled Live Generator & Independent Judge Resolver
Decouples Generator Model and Judge Model into independent model IDs / TPD pools:
  - Generator Model: meta-llama/llama-4-scout-17b-16e-instruct (fallback qwen/qwen3.8-27b)
  - Judge Model: llama-3.1-8b-instant (fallback openai/gpt-oss-20b)
Eliminates LLM self-preference bias and avoids per-model TPD exhaustion on Groq.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(r"D:\FinFaithful-RAG\.env", override=True)

_CACHED_GEN_CLIENT: Optional[OpenAI] = None
_CACHED_GEN_PROVIDER: Optional[str] = None
_CACHED_GEN_MODEL: Optional[str] = None

_CACHED_JUDGE_CLIENT: Optional[OpenAI] = None
_CACHED_JUDGE_PROVIDER: Optional[str] = None
_CACHED_JUDGE_MODEL: Optional[str] = None


def strip_think_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning blocks from thinking models before scoring or JSON parsing."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def resolve_generator_llm() -> Tuple[OpenAI, str, str]:
    """
    Resolves and locks the live Generator Model.
    Priority:
      1. CEREBRAS_API_KEY -> https://api.cerebras.ai/v1 (llama3.1-70b / llama3.3-70b)
      2. GROQ_API_KEY -> meta-llama/llama-4-scout-17b-16e-instruct -> qwen/qwen3.8-27b
      3. NVIDIA_API_KEY -> meta/llama-3.1-70b-instruct
    Returns (client, provider_name, model_id).
    """
    global _CACHED_GEN_CLIENT, _CACHED_GEN_PROVIDER, _CACHED_GEN_MODEL
    if _CACHED_GEN_CLIENT is not None:
        return _CACHED_GEN_CLIENT, _CACHED_GEN_PROVIDER, _CACHED_GEN_MODEL

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    errors = []

    # 1. Cerebras
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    if cerebras_key and not cerebras_key.startswith("csk-placeholder") and len(cerebras_key) > 10:
        base_url = "https://api.cerebras.ai/v1"
        client = OpenAI(base_url=base_url, api_key=cerebras_key, timeout=20)
        for m in ["llama-3.3-70b", "llama3.1-70b", "llama3.1-8b"]:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "Respond OK"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                txt = strip_think_blocks(resp.choices[0].message.content or "")
                if txt:
                    _CACHED_GEN_CLIENT = client
                    _CACHED_GEN_PROVIDER = "cerebras"
                    _CACHED_GEN_MODEL = m
                    return _CACHED_GEN_CLIENT, _CACHED_GEN_PROVIDER, _CACHED_GEN_MODEL
            except Exception as e:
                errors.append(f"Cerebras ({m}): {type(e).__name__}: {e}")

    # 2. Groq Generator
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("gsk_placeholder") and len(groq_key) > 10:
        base_url = "https://api.groq.com/openai/v1"
        client = OpenAI(base_url=base_url, api_key=groq_key, timeout=20)
        groq_gen_candidates = [
            os.getenv("GROQ_GENERATOR_MODEL", ""),
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ]
        for m in [c for c in groq_gen_candidates if c]:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "Respond OK"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                txt = strip_think_blocks(resp.choices[0].message.content or "")
                if txt or resp.choices[0].message.content is not None:
                    _CACHED_GEN_CLIENT = client
                    _CACHED_GEN_PROVIDER = "groq"
                    _CACHED_GEN_MODEL = m
                    return _CACHED_GEN_CLIENT, _CACHED_GEN_PROVIDER, _CACHED_GEN_MODEL
            except Exception as e:
                errors.append(f"Groq Generator ({m}): {type(e).__name__}: {e}")

    raise RuntimeError(
        f"CRITICAL PRE-FLIGHT CHECK FAILED: No live Generator model accessible.\nErrors:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def resolve_judge_llm() -> Tuple[OpenAI, str, str]:
    """
    Resolves and locks the live Independent Judge Model.
    Priority:
      1. GROQ_API_KEY -> llama-3.1-8b-instant -> openai/gpt-oss-20b -> allam-2-7b
    Returns (client, provider_name, model_id).
    """
    global _CACHED_JUDGE_CLIENT, _CACHED_JUDGE_PROVIDER, _CACHED_JUDGE_MODEL
    if _CACHED_JUDGE_CLIENT is not None:
        return _CACHED_JUDGE_CLIENT, _CACHED_JUDGE_PROVIDER, _CACHED_JUDGE_MODEL

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    errors = []

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("gsk_placeholder") and len(groq_key) > 10:
        base_url = "https://api.groq.com/openai/v1"
        client = OpenAI(base_url=base_url, api_key=groq_key, timeout=20)
        groq_judge_candidates = [
            os.getenv("GROQ_JUDGE_MODEL", ""),
            "llama-3.1-8b-instant",
            "allam-2-7b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ]
        for m in [c for c in groq_judge_candidates if c]:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": '{"faithfulness_score": 5, "relevance_score": 5, "rationale": "OK"}'}],
                    max_tokens=30,
                    temperature=0.0,
                )
                content = resp.choices[0].message.content or ""
                if content or getattr(resp.choices[0].message, "reasoning", ""):
                    _CACHED_JUDGE_CLIENT = client
                    _CACHED_JUDGE_PROVIDER = "groq"
                    _CACHED_JUDGE_MODEL = m
                    return _CACHED_JUDGE_CLIENT, _CACHED_JUDGE_PROVIDER, _CACHED_JUDGE_MODEL
            except Exception as e:
                errors.append(f"Groq Judge ({m}): {type(e).__name__}: {e}")

    raise RuntimeError(
        f"CRITICAL PRE-FLIGHT CHECK FAILED: No live Judge model accessible.\nErrors:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def assert_live_models() -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Mandatory live pre-flight check at startup:
    Verifies BOTH Generator and Judge endpoints with live HTTP 200 completion.
    Returns ((gen_provider, gen_model), (judge_provider, judge_model)).
    """
    _, gen_prov, gen_model = resolve_generator_llm()
    _, judge_prov, judge_model = resolve_judge_llm()
    return (gen_prov, gen_model), (judge_prov, judge_model)


# Backwards compatibility helper
def resolve_live_llm() -> Tuple[OpenAI, str, str, str]:
    client, prov, model = resolve_generator_llm()
    base_url = "https://api.groq.com/openai/v1"
    return client, prov, base_url, model


def assert_live_llm() -> Tuple[str, str]:
    (gen_prov, gen_model), _ = assert_live_models()
    return gen_prov, gen_model
