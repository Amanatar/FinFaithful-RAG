"""
FinFaithful-RAG: Empirical Evaluation Engine & Metrics Calculator (Phase 4)

Computes objective non-LLM metrics and deterministic LLM Judge scores:
  1. Recall@5 & MRR@10 against gold_chunk_ids (Categories A-C)
  2. Numeric Fact Coverage % (normalized numeric regex matching against key_numbers)
  3. Citation Validity % & Phantom Citation Detection
  4. Trap Abstention Accuracy % on Category D (adversarial_unanswerable)
  5. Retrieval Latency (P50 / P95 ms) & End-to-End Latency (P50 s)
  6. Deterministic LLM Judge (faithfulness_score 1-5, relevance_score 1-5)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


# =============================================================================
# OBJECTIVE NON-LLM METRICS
# =============================================================================

def normalize_number_str(num_str: str) -> str:
    """Strips currency symbols, commas, percent signs, and whitespace."""
    return re.sub(r"[\$,\s\%]", "", str(num_str)).lower()


def calculate_numeric_fact_coverage(key_numbers: List[str], answer: str) -> float:
    """
    Computes the fraction of key numbers from the ground truth present in the answer.
    Returns 1.0 if key_numbers is empty (e.g. Category D adversarial traps).
    """
    if not key_numbers:
        return 1.0

    ans_norm = normalize_number_str(answer)
    hits = 0

    for num in key_numbers:
        # 1. Exact substring check
        if num in answer:
            hits += 1
            continue
        # 2. Normalized check (ignoring commas, $, %)
        norm_num = normalize_number_str(num)
        if norm_num and norm_num in ans_norm:
            hits += 1
            continue
        # 3. Numeric decimal check (e.g., "214.4 billion" -> "214.4")
        num_clean = re.sub(r"[^\d\.]", "", num)
        if num_clean and num_clean in answer:
            hits += 1

    return hits / len(key_numbers)


def calculate_recall_at_k(
    gold_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int = 5,
) -> float:
    """
    Computes Recall@K: proportion of gold chunks retrieved in the top-K chunks.
    For Category D (unanswerable), returns 1.0 if no chunks were required.
    """
    if not gold_chunk_ids:
        return 1.0
    top_k_retrieved = set(retrieved_chunk_ids[:k])
    hits = sum(1 for gid in gold_chunk_ids if gid in top_k_retrieved)
    return hits / len(gold_chunk_ids)


def calculate_mrr_at_k(
    gold_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int = 10,
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K): reciprocal rank of first gold chunk.
    Returns 1.0 for Category D (unanswerable).
    """
    if not gold_chunk_ids:
        return 1.0
    gold_set = set(gold_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids[:k], 1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


# =============================================================================
# DETERMINISTIC LLM JUDGE
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """You are a rigorous, skeptical Senior Financial NLP Auditor grading an AI RAG answer.
Score the answer strictly from 1 to 5 on two criteria:

1. Faithfulness Score (1 to 5):
   - 5 = 100% supported by the retrieved SEC 10-K context chunks. Every figure and fact is present in context.
   - 4 = Grounded in context with minor non-material phrasing additions.
   - 3 = Partially supported, but introduces unverified claims or secondary numbers not in context.
   - 2 = Heavy reliance on parametric memory; contains conflicting or unsupported numbers.
   - 1 = Hallucinated figures, contradicted 10-K data, or fabricated guidance on unanswerable traps.
   * NOTE: For an unanswerable trap where no evidence exists, properly abstaining with [INSUFFICIENT_FILING_EVIDENCE] MUST receive a 5.

2. Relevance Score (1 to 5):
   - 5 = Directly answers the specific financial question asked, or properly abstains on unanswerable traps.
   - 4 = Answers the question with slight verbosity or minor omission of secondary context.
   - 3 = Partially answers the question or answers an adjacent financial metric.
   - 2 = Vague or tangential response that fails to answer the requested figure.
   - 1 = Fails completely, hallucinates a response to a trap, or provides irrelevant output.

Context Excerpts:
{context_str}

Question: {question}
Expected Ground Truth: {ground_truth}
AI Answer: {answer}

Respond ONLY with a JSON object in this exact schema:
{{"faithfulness_score": <1-5>, "relevance_score": <1-5>, "reason": "<concise audit reason>"}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    reraise=False,
)
def _call_judge_llm(
    judge_prompt: str,
    client: OpenAI,
    model: str = DEFAULT_NVIDIA_MODEL,
) -> Optional[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        top_p=1.0,
        response_format={"type": "json_object"},
        timeout=20,
    )
    return response.choices[0].message.content


def judge_answer(
    question: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
    item: Dict[str, Any],
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Evaluates faithfulness and relevance via deterministic LLM Judge or deterministic fallback.
    """
    is_trap = item.get("must_abstain", False)
    has_abstention = "[INSUFFICIENT_FILING_EVIDENCE]" in answer

    # Check if live NVIDIA NIM LLM Judge is usable
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if client and api_key and not api_key.startswith("nvapi-placeholder") and len(api_key) > 10:
        context_str = "\n".join(
            [f"[{c.get('ticker')}|Chunk {c.get('chunk_id')}]: {c.get('text', '')[:300]}" for c in context_chunks]
        ) or "NO RETRIEVED CONTEXT."
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            context_str=context_str,
            question=question,
            ground_truth=item.get("ground_truth_answer", ""),
            answer=answer,
        )
        try:
            raw_json = _call_judge_llm(prompt, client)
            if raw_json:
                data = json.loads(raw_json)
                return {
                    "faithfulness_score": float(data.get("faithfulness_score", 3.0)),
                    "relevance_score": float(data.get("relevance_score", 3.0)),
                    "judge_reason": data.get("reason", "Live LLM Judge evaluated"),
                    "judge_engine": "nvidia_nim_llm_judge",
                }
        except Exception:
            pass

    # Deterministic Rule-Based Auditor Fallback
    # 1. Category D (Adversarial Trap)
    if is_trap:
        if has_abstention:
            return {
                "faithfulness_score": 5.0,
                "relevance_score": 5.0,
                "judge_reason": "Clean abstention on unanswerable filing trap [INSUFFICIENT_FILING_EVIDENCE].",
                "judge_engine": "deterministic_auditor",
            }
        else:
            return {
                "faithfulness_score": 1.0,
                "relevance_score": 1.0,
                "judge_reason": "Severe failure: Model hallucinated or invented numbers on an unanswerable trap.",
                "judge_engine": "deterministic_auditor",
            }

    # 2. Categories A, B, C (Answerable)
    if not context_chunks:
        # Direct LLM mode without context
        cov = calculate_numeric_fact_coverage(item.get("key_numbers", []), answer)
        return {
            "faithfulness_score": 1.5,
            "relevance_score": 2.5 + (2.0 * cov),
            "judge_reason": "Direct LLM has zero retrieved 10-K context; answer lacks verified filing citations.",
            "judge_engine": "deterministic_auditor",
        }

    # Grounded RAG modes
    key_nums = item.get("key_numbers", [])
    cov = calculate_numeric_fact_coverage(key_nums, answer)
    raw_citations = re.findall(r"\[([^\]]+)\]", answer)
    has_valid_tag = any("|" in cit for cit in raw_citations)

    # Faithfulness calculation
    if cov >= 0.99 and has_valid_tag:
        f_score = 5.0
        f_reason = "100% verified numbers present and supported by retrieved chunks."
    elif cov >= 0.5:
        f_score = 4.0
        f_reason = "Grounded in context with partial line-item extraction."
    elif cov > 0:
        f_score = 3.0
        f_reason = "Contains some numbers from context, but incomplete."
    else:
        f_score = 2.5
        f_reason = "Grounded narrative text present, but key financial figures omitted."

    # Relevance calculation
    if cov >= 0.99:
        r_score = 5.0
    elif cov >= 0.5:
        r_score = 4.0
    else:
        r_score = 3.0

    return {
        "faithfulness_score": f_score,
        "relevance_score": r_score,
        "judge_reason": f_reason,
        "judge_engine": "deterministic_auditor",
    }


# =============================================================================
# SINGLE RUN EVALUATION
# =============================================================================

def evaluate_single_run(
    item: Dict[str, Any],
    pipeline_result: Dict[str, Any],
    judge_client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Evaluates a single benchmark item run, computing all objective metrics
    and judge scores. Returns a structured dictionary ready for JSONL persistence.
    """
    mode = pipeline_result["mode"]
    answer = pipeline_result["answer"]
    retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
    retrieved_cids = [c["global_chunk_id"] for c in retrieved_chunks]
    gold_cids = item.get("gold_chunk_ids", [])
    key_nums = item.get("key_numbers", [])
    is_trap = item.get("must_abstain", False)

    # 1. Retrieval Metrics
    if mode == "direct_llm":
        recall_at_5 = 0.0
        mrr_at_10 = 0.0
    else:
        recall_at_5 = calculate_recall_at_k(gold_cids, retrieved_cids, k=5)
        mrr_at_10 = calculate_mrr_at_k(gold_cids, retrieved_cids, k=10)

    # 2. Fact Coverage
    fact_coverage = calculate_numeric_fact_coverage(key_nums, answer)

    # 3. Citation Audit
    audit = pipeline_result.get("citation_audit", {})
    valid_cits = audit.get("valid_citations", [])
    phantom_cits = audit.get("phantom_citations", [])
    total_cits = audit.get("total_citations_found", 0)

    if total_cits > 0:
        citation_validity = len(valid_cits) / total_cits
    else:
        citation_validity = 1.0 if (is_trap or mode == "direct_llm") else 0.0

    has_phantom = len(phantom_cits) > 0
    has_abstention = "[INSUFFICIENT_FILING_EVIDENCE]" in answer

    # 4. Abstention Accuracy
    if is_trap:
        abstention_accuracy = 1.0 if has_abstention else 0.0
    else:
        abstention_accuracy = 1.0  # Answerable items not penalized by trap abstention

    # 5. Judge Scores
    judge_res = judge_answer(
        question=item["question"],
        answer=answer,
        context_chunks=retrieved_chunks[:5],
        item=item,
        client=judge_client,
    )

    total_latency_ms = pipeline_result.get("total_latency_ms", 0.0)

    return {
        "question_id": item["question_id"],
        "category": item["category"],
        "tickers": item["tickers"],
        "primary_ticker": item.get("primary_ticker", item["tickers"][0]),
        "fiscal_year": item.get("fiscal_year", "2026"),
        "question": item["question"],
        "mode": mode,
        "ground_truth_answer": item.get("ground_truth_answer", ""),
        "key_numbers": key_nums,
        "gold_chunk_ids": gold_cids,
        "retrieved_chunk_ids": retrieved_cids[:10],
        "top_5_chunk_ids": retrieved_cids[:5],
        "recall_at_5": round(recall_at_5, 4),
        "mrr_at_10": round(mrr_at_10, 4),
        "numeric_fact_coverage": round(fact_coverage, 4),
        "citation_validity": round(citation_validity, 4),
        "has_phantom_citations": has_phantom,
        "has_abstention": has_abstention,
        "abstention_accuracy": round(abstention_accuracy, 4),
        "faithfulness_score": round(judge_res["faithfulness_score"], 2),
        "relevance_score": round(judge_res["relevance_score"], 2),
        "judge_reason": judge_res["judge_reason"],
        "judge_engine": judge_res["judge_engine"],
        "retrieval_latency_ms": round(pipeline_result.get("retrieval_latency_ms", 0.0), 2),
        "rerank_latency_ms": round(pipeline_result.get("rerank_latency_ms", 0.0), 2),
        "generation_latency_ms": round(pipeline_result.get("generation_latency_ms", 0.0), 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "total_latency_s": round(total_latency_ms / 1000.0, 3),
        "answer": answer,
        "engine": pipeline_result.get("engine", "default"),
    }


# =============================================================================
# SUMMARY & BREAKDOWN AGGREGATOR
# =============================================================================

def aggregate_evaluation_results(
    runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Computes overall benchmark summary metrics and per-category breakdown tables.
    """
    df = pd.DataFrame(runs)

    summary: Dict[str, Any] = {}
    modes = ["direct_llm", "sparse_bm25", "dense_faiss", "hybrid_rerank"]

    for m in modes:
        df_m = df[df["mode"] == m]
        if df_m.empty:
            continue

        # Split answerable vs trap
        df_answerable = df_m[df_m["category"] != "adversarial_unanswerable"]
        df_trap = df_m[df_m["category"] == "adversarial_unanswerable"]

        summary[m] = {
            "total_questions": len(df_m),
            "mean_faithfulness": round(float(df_m["faithfulness_score"].mean()), 2),
            "mean_relevance": round(float(df_m["relevance_score"].mean()), 2),
            "mean_recall_at_5": round(float(df_answerable["recall_at_5"].mean()), 4) if not df_answerable.empty else 0.0,
            "mean_mrr_at_10": round(float(df_answerable["mrr_at_10"].mean()), 4) if not df_answerable.empty else 0.0,
            "numeric_fact_coverage_pct": round(float(df_answerable["numeric_fact_coverage"].mean()) * 100.0, 2) if not df_answerable.empty else 0.0,
            "citation_validity_pct": round(float(df_m["citation_validity"].mean()) * 100.0, 2),
            "phantom_citations_count": int(df_m["has_phantom_citations"].sum()),
            "trap_abstention_accuracy_pct": round(float(df_trap["abstention_accuracy"].mean()) * 100.0, 2) if not df_trap.empty else 100.0,
            "retrieval_latency_p50_ms": round(float(np.percentile(df_m["retrieval_latency_ms"], 50)), 1),
            "retrieval_latency_p95_ms": round(float(np.percentile(df_m["retrieval_latency_ms"], 95)), 1),
            "end_to_end_latency_p50_s": round(float(np.percentile(df_m["total_latency_s"], 50)), 3),
        }

    # Per-Category Breakdown Table
    breakdown_rows = []
    categories = [
        "table_numeric",
        "yoy_footnote_reasoning",
        "cross_company_comparative",
        "adversarial_unanswerable",
    ]

    for cat in categories:
        for m in modes:
            sub = df[(df["category"] == cat) & (df["mode"] == m)]
            if sub.empty:
                continue
            breakdown_rows.append(
                {
                    "category": cat,
                    "mode": m,
                    "count": len(sub),
                    "faithfulness": round(float(sub["faithfulness_score"].mean()), 2),
                    "relevance": round(float(sub["relevance_score"].mean()), 2),
                    "recall_at_5": round(float(sub["recall_at_5"].mean()), 4),
                    "mrr_at_10": round(float(sub["mrr_at_10"].mean()), 4),
                    "fact_coverage_pct": round(float(sub["numeric_fact_coverage"].mean()) * 100.0, 2),
                    "citation_validity_pct": round(float(sub["citation_validity"].mean()) * 100.0, 2),
                    "abstention_acc_pct": round(float(sub["abstention_accuracy"].mean()) * 100.0, 2),
                    "p50_latency_ms": round(float(np.percentile(sub["total_latency_ms"], 50)), 1),
                }
            )

    df_breakdown = pd.DataFrame(breakdown_rows)
    return summary, df_breakdown
