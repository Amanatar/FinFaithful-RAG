"""
FinFaithful-RAG: Grounded Generation Engine with Lineage Verification (Phase 4 / 4.5)

Enforces:
  1. Strict Citation Lineage: [Ticker | FY {fiscal_year} | {item_section} | Chunk {chunk_id}]
  2. Zero Hallucination: Grounded strictly in retrieved context.
  3. Explicit Abstention: [INSUFFICIENT_FILING_EVIDENCE] on unanswerable/forward-looking traps.
  4. OpenAI-compatible live endpoint execution (Groq, Cerebras) with <think> stripping.
  5. Compact context trimming (<=850 chars per chunk, max_tokens=220) for strict TPD compliance.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from llm_client import resolve_generator_llm, strip_think_blocks

SYSTEM_PROMPT = """You are a Staff Financial NLP & Retrieval Systems Auditor AI analyzing SEC Form 10-K annual reports.
Your answers must strictly adhere to four mandatory financial auditing rules:

1. STRICT CITATION LINEAGE: Every numerical figure, monetary amount, growth percentage, segment metric, or accounting claim MUST carry an inline bracket citation in the exact format:
   [Ticker | FY {fiscal_year} | {item_section} | Chunk {chunk_id}]
   matching one of the provided retrieved context chunks. Never provide a number without an inline citation immediately following it.

2. ZERO HALLUCINATION / ZERO PARAMETRIC EXTRAPOLATION: Never extrapolate, speculate, or invent figures from parametric memory. If a requested metric, fiscal quarter, segment, or future year is not explicitly provided in the retrieved 10-K excerpts, you must state that it is missing.

3. EXPLICIT ABSTENTION: When the retrieved chunks do not contain sufficient evidence to answer the question (or when asked for non-existent future periods, undisclosed projections, or unverified claims), you MUST explicitly output:
   [INSUFFICIENT_FILING_EVIDENCE]
   followed by a concise factual explanation detailing what is and what is not disclosed in the retrieved filings.

4. DIRECT RESPONSE: Provide the direct factual response immediately without thinking tags, internal chain-of-thought, or <think> blocks.
"""


DIRECT_LLM_SYSTEM_PROMPT = """You are a financial analyst analyzing SEC Form 10-K filings.
Answer the financial question directly and concisely based on your pre-trained financial knowledge. State specific numbers, dates, and metrics where known.
If a requested metric is a non-existent metric, forward-looking projection not filed in historical 10-Ks, or confidential trade secret, state clearly that it is not disclosed. Do not output bracketed citations as you do not have external filing chunks."""


def query_aware_trim_chunk(chunk_text: str, query: str, max_chars: int = 1400) -> str:
    """
    Query-aware contextual trimmer (<= max_chars):
    - Preserves provenance line [Ticker | FY | Section | Chunk] and natural language summary.
    - If a Markdown table is present:
        * Preserves table title and column header rows (| Header | Year | and | :--- | :--- |).
        * Extracts keywords from query (numbers, monetary tokens, words >= 3 chars).
        * Prioritizes key bottom-line summary rows (Total, Net income, Operating income, Diluted, Per share, Cash, Debt).
        * Keeps matching table rows so bottom-line figures are never sliced off.
    - Enforces <= max_chars hard budget.
    """
    if not chunk_text or len(chunk_text) <= max_chars:
        return (chunk_text or "").strip()

    lines = chunk_text.splitlines()
    if not lines:
        return chunk_text[:max_chars].strip()

    q_clean = re.sub(r"[^\w\s]", " ", query.lower())
    q_words = {w for w in q_clean.split() if len(w) >= 3}
    q_numbers = set(re.findall(r"\b\d+\b", query))

    priority_anchors = {
        "total", "net income", "operating income", "operating revenue", "gross profit",
        "diluted", "basic", "per share", "total revenue", "net sales", "total assets",
        "long-term debt", "commercial paper", "cash and cash equivalents", "interest income",
        "segment", "americas", "europe", "greater china", "cloud", "intelligent cloud",
        "more personal computing", "data center", "gaming", "automotive"
    }

    hdr, tbl_hdr, match_rows, other_rows = [], [], [], []
    tbl_captured = False

    for l in lines:
        s = l.strip()
        if s.startswith("[Ticker:") or s.startswith("[AAPL") or s.startswith("[MSFT") or s.startswith("[NVDA") or s.startswith("[JPM") or s.startswith("[GOOGL"):
            hdr.append(l)
            continue
        if any(w in s for w in ["Form 10-K", "table covering", "Statement:"]) and not s.startswith("|"):
            hdr.append(l)
            continue

        if s.startswith("|"):
            if not tbl_captured:
                tbl_hdr.append(l)
                if "--" in l:
                    tbl_captured = True
                continue
            low = s.lower()
            w = set(re.sub(r"[^\w\s]", " ", low).split())
            nums = set(re.findall(r"\b\d+\b", s))
            if (w & q_words) or (nums & q_numbers) or any(a in low for a in priority_anchors):
                match_rows.append(l)
            else:
                other_rows.append(l)
        else:
            if not tbl_captured:
                hdr.append(l)
            else:
                other_rows.append(l)

    out = hdr + tbl_hdr + match_rows
    cur = sum(len(x) + 1 for x in out)
    for l in other_rows:
        if cur + len(l) + 1 <= max_chars:
            out.append(l)
            cur += len(l) + 1
        else:
            break

    res = "\n".join(out).strip()
    if len(res) > max_chars:
        res = res[:max_chars].rsplit("\n", 1)[0]
    return res


def verify_citations(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight citation verifier function that checks whether cited ChunkIDs
    actually exist in the retrieved context and flags phantom citations.
    """
    clean_ans = strip_think_blocks(answer)
    raw_citations = re.findall(r"\[([^\]]+)\]", clean_ans)
    has_abstention = "[INSUFFICIENT_FILING_EVIDENCE]" in clean_ans

    valid_keys = set()
    valid_cids = set()
    for c in retrieved_chunks:
        ticker = str(c.get("ticker", "")).upper()
        fy = str(c.get("fiscal_year", ""))
        sec = str(c.get("item_section", "")).upper()
        cid = str(c.get("chunk_id", ""))
        gid = str(c.get("global_chunk_id", "")).upper()

        valid_keys.add(f"{ticker}_{cid}")
        valid_keys.add(f"{ticker}_{fy}_{cid}")
        valid_keys.add(gid)
        valid_cids.add(cid)
        valid_cids.add(gid)

    found_citations: List[str] = []
    valid_citations: List[str] = []
    phantom_citations: List[str] = []

    for cit in raw_citations:
        cit_clean = cit.strip()
        if cit_clean in ("INSUFFICIENT_FILING_EVIDENCE", "FINANCIAL_TABLE_START", "FINANCIAL_TABLE_END"):
            continue

        found_citations.append(cit_clean)

        if "|" in cit_clean:
            parts = [p.strip() for p in cit_clean.split("|")]
            cit_ticker = parts[0].upper()
            chunk_part = parts[-1].upper().replace("CHUNK", "").strip()

            matched = False
            for c in retrieved_chunks:
                c_ticker = str(c.get("ticker", "")).upper()
                c_id = str(c.get("chunk_id", "")).strip()
                c_gid = str(c.get("global_chunk_id", "")).upper()
                if cit_ticker == c_ticker and (chunk_part == c_id or chunk_part in c_gid):
                    matched = True
                    break

            if matched:
                valid_citations.append(cit_clean)
            else:
                phantom_citations.append(cit_clean)
        elif cit_clean.upper() in valid_cids or cit_clean.upper() in valid_keys:
            valid_citations.append(cit_clean)
        else:
            matched = False
            for c in retrieved_chunks:
                if str(c.get("chunk_id", "")) in cit_clean or str(c.get("global_chunk_id", "")).upper() in cit_clean.upper():
                    matched = True
                    break
            if matched:
                valid_citations.append(cit_clean)
            else:
                phantom_citations.append(cit_clean)

    is_valid = (len(phantom_citations) == 0) and (len(valid_citations) > 0 or has_abstention or len(retrieved_chunks) == 0)

    return {
        "is_valid": is_valid,
        "has_abstention": has_abstention,
        "total_citations_found": len(found_citations),
        "valid_citations": valid_citations,
        "phantom_citations": phantom_citations,
    }


def _call_generator_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 320,
    temperature: float = 0.0,
) -> str:
    """Calls locked live generator model with automatic failover to gpt-oss-120b on TPD exhaustion."""
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if "gpt-oss" in model:
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        return strip_think_blocks(raw)
    except Exception as e:
        err_msg = str(e)
        if "tokens per day (TPD)" in err_msg and model != "openai/gpt-oss-20b":
            print(f"[FAILOVER] TPD exhausted on {model}. Failing over to openai/gpt-oss-20b...", flush=True)
            fallback_kwargs = {
                "model": "openai/gpt-oss-20b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": {"reasoning_effort": "low"},
            }
            resp = client.chat.completions.create(**fallback_kwargs)
            raw = resp.choices[0].message.content or ""
            return strip_think_blocks(raw)
        raise


def generate(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    mode: str = "hybrid_rerank",
    scope_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes grounded generation with strict citation lineage & abstention verification.
    Uses the locked live Generator Model with query-aware contextual trimming (<=1,400 chars).
    """
    t0 = time.perf_counter()

    client, provider_name, model_id = resolve_generator_llm()

    # Ungrounded direct LLM baseline uses direct financial prompt without retrieval context or citation constraints
    if mode == "direct_llm":
        user_prompt = (
            f"Question: {query}\n\n"
            f"Answer the financial question directly and concisely based on your pre-trained financial knowledge. "
            f"State specific numbers, dates, and metrics where known. If the metric is a non-existent metric, "
            f"unfiled future projection, or confidential trade secret, state clearly that it is not disclosed."
        )
        messages = [
            {"role": "system", "content": DIRECT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        answer = _call_generator_llm(
            client=client,
            model=model_id,
            messages=messages,
            max_tokens=220,
            temperature=0.0,
        )
        gen_latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query,
            "mode": mode,
            "engine": f"{provider_name}:{model_id}",
            "answer": answer,
            "citation_audit": {
                "valid_citations": [],
                "phantom_citations": [],
                "total_citations_found": 0,
            },
            "generation_latency_ms": gen_latency_ms,
        }

    # Format query-aware context blocks (top 5 chunks max, trimmed to <= 850 characters each)
    context_blocks = []
    for c in retrieved_chunks[:5]:
        c_header = f"[{c['ticker']} | FY {c['fiscal_year']} | {c['item_section']} | Chunk {c['chunk_id']}]"
        raw_text = c.get("text", "")
        c_text = query_aware_trim_chunk(raw_text, query, max_chars=850)
        context_blocks.append(f"--- EXCERPT {c_header} ---\n{c_text}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "NO CONTEXT RETRIEVED (DIRECT LLM MODE)."

    scope_notice = ""
    # Only supply scope warnings for retrieval modes; direct_llm baseline must not receive cheat hints
    if mode != "direct_llm" and scope_check and scope_check.get("is_unanswerable_trap"):
        scope_notice = (
            f"\n[SCOPE WARNING: This query requests figures that are unannounced, forward-looking, or confidential trade secrets "
            f"({scope_check.get('abstention_reason')}). If the retrieved excerpts do not contain explicit certified line-item evidence, "
            f"output [INSUFFICIENT_FILING_EVIDENCE] immediately.]\n"
        )

    user_prompt = f"""Question: {query}
{scope_notice}
Retrieved SEC 10-K Excerpts:
{context_str}

Provide a concise, factual answer adhering strictly to the citation and abstention rules."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    answer = _call_generator_llm(
        client=client,
        model=model_id,
        messages=messages,
        max_tokens=220,
        temperature=0.0,
    )

    gen_latency_ms = (time.perf_counter() - t0) * 1000.0

    citation_audit = verify_citations(answer, retrieved_chunks[:5])

    return {
        "query": query,
        "mode": mode,
        "engine": f"{provider_name}:{model_id}",
        "answer": answer,
        "citation_audit": citation_audit,
        "generation_latency_ms": gen_latency_ms,
    }
