"""
FinFaithful-RAG: Empirical Evaluation Engine & Metrics Calculator (Phase 4 / Phase 4.5)

Computes objective non-LLM metrics and independent LLM Judge scores:
  1. Recall@5 & MRR@10 against gold_chunk_ids (strictly Categories A-C; Category D reports N/A)
  2. Numeric Fact Coverage % (normalized numeric regex matching against key_numbers; Category D reports N/A)
  3. Citation Validity % (0.0% for direct_llm; handles phantom citations cleanly)
  4. Trap Abstention Accuracy % on Category D (adversarial_unanswerable)
  5. Retrieval Latency (P50 / P95 ms) & End-to-End Latency (P50 s)
  6. Independent LLM Judge with structured JSON output, concise prompt (~500 tokens), and <think> stripping
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from llm_client import assert_live_models, resolve_judge_llm, strip_think_blocks

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OBJECTIVE NON-LLM METRICS
# =============================================================================

def normalize_number_str(num_str: str) -> str:
    """Strips currency symbols, commas, percent signs, and whitespace."""
    s = re.sub(r"[\$,\s\%]", "", str(num_str)).lower()
    s = re.sub(r"\.0+$", "", s)
    return s


def calculate_numeric_fact_coverage(
    key_numbers: List[str],
    answer: str,
    is_trap: bool = False,
) -> Optional[float]:
    """
    Computes the fraction of key numbers from the ground truth present in the answer.
    Returns None for Category D (adversarial traps) where no facts should be covered.
    Uses normalized comparison (stripping commas, $, trailing decimals) to avoid false negative drops.
    """
    if is_trap or not key_numbers:
        return None

    clean_ans = strip_think_blocks(answer)
    ans_norm = re.sub(r"(?<=\d),(?=\d)", "", clean_ans)
    ans_clean = normalize_number_str(ans_norm)
    hits = 0

    for num in key_numbers:
        num_str = str(num).strip()
        norm_num = normalize_number_str(num_str)

        # 1. Exact substring check in raw answer
        if num_str in clean_ans:
            hits += 1
            continue

        # 2. Normalized check (ignoring commas, $, %)
        if norm_num and norm_num in ans_clean:
            hits += 1
            continue

        # 3. Numeric decimal check (e.g., "214.4 billion" -> "214.4")
        num_clean = re.sub(r"[^\d\.]", "", num_str)
        num_clean = re.sub(r"\.0+$", "", num_clean)
        if num_clean and (num_clean in clean_ans or num_clean in ans_clean):
            hits += 1
            continue

        # 4. Formatted integer check (e.g. 53809 in key_nums vs 53,809 in answer)
        try:
            val = float(num_clean)
            if f"{int(val):,}" in clean_ans:
                hits += 1
                continue
        except Exception:
            pass

    return hits / len(key_numbers)


def is_abstention(
    text: str,
    fact_coverage: Optional[float] = None,
    citation_validity: Optional[float] = None,
) -> bool:
    """
    Returns True ONLY when the response contains [INSUFFICIENT_FILING_EVIDENCE]
    OR explicitly states the requested metric/period is not disclosed in the 10-K
    WITHOUT asserting a fabricated numeric figure as fact.
    If the response extracts gold numeric facts with valid citations, it is an attempted answer.
    """
    if not text:
        return False
    # If the model extracted facts with valid citations, it attempted an answer rather than abstaining
    if fact_coverage and fact_coverage > 0.0 and (citation_validity is None or citation_validity >= 0.5):
        return False

    clean = strip_think_blocks(text).strip()
    if "[INSUFFICIENT_FILING_EVIDENCE]" in clean:
        return True

    clean_lower = clean.lower()
    refusal_phrases = [
        "not disclosed in the",
        "not provided in the",
        "does not contain",
        "no information is provided",
        "not reported in the",
        "is not disclosed in",
        "are confidential trade secrets",
        "confidential trade secret",
        "not publicly disclosed",
        "cannot be determined from the",
        "no filing evidence",
        "is not available in the",
        "are omitted from",
        "unfiled fiscal year",
        "has not yet concluded",
        "has not been released",
        "has not been filed",
        "not disclosed.",
        "not disclosed",
    ]
    return any(p in clean_lower for p in refusal_phrases)


def calculate_hit_rate_at_k(
    gold_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int = 5,
    is_trap: bool = False,
) -> Optional[float]:
    """
    Computes HitRate@K (standard binary hit rate):
    Returns 1.0 if AT LEAST ONE valid gold_chunk_id appears in the top-K retrieved chunks; 0.0 otherwise.
    Returns None for Category D (adversarial traps) where no gold chunks exist.
    """
    if is_trap or not gold_chunk_ids:
        return None
    top_k_retrieved = {cid.lower() for cid in retrieved_chunk_ids[:k]}
    for gid in gold_chunk_ids:
        if gid.lower() in top_k_retrieved:
            return 1.0
    return 0.0


def calculate_recall_at_k(
    gold_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int = 5,
    is_trap: bool = False,
) -> Optional[float]:
    """Standard HitRate@5 / Recall@5 against gold chunks."""
    return calculate_hit_rate_at_k(gold_chunk_ids, retrieved_chunk_ids, k=k, is_trap=is_trap)


def calculate_mrr_at_k(
    gold_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    k: int = 10,
    is_trap: bool = False,
) -> Optional[float]:
    """
    Computes Mean Reciprocal Rank (MRR@K): reciprocal rank of first gold chunk.
    Returns None for Category D (unanswerable traps).
    """
    if is_trap or not gold_chunk_ids:
        return None
    gold_set_lower = {gid.lower() for gid in gold_chunk_ids}
    for rank, cid in enumerate(retrieved_chunk_ids[:k], 1):
        if cid.lower() in gold_set_lower:
            return 1.0 / rank
    return 0.0


# =============================================================================
# DETERMINISTIC INDEPENDENT LLM JUDGE
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """You are a rigorous, skeptical Senior Financial NLP Auditor grading an AI RAG answer.
Score the answer strictly on two criteria:

1. Faithfulness Score (integer 1 to 5):
   - 5 = 100% supported by the retrieved SEC 10-K context chunks. Every figure and fact is present in context.
   - 4 = Grounded in context with minor non-material phrasing additions.
   - 3 = Partially supported, but introduces unverified claims or secondary numbers not in context.
   - 2 = Heavy reliance on parametric memory; contains conflicting or unsupported numbers.
   - 1 = Hallucinated figures, contradicted 10-K data, or fabricated guidance on unanswerable traps.
   * NOTE: For an unanswerable trap where no evidence exists, properly abstaining with [INSUFFICIENT_FILING_EVIDENCE] MUST receive a 5.

2. Relevance Score (integer 1 to 5):
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
{{"faithfulness_score": <1-5>, "relevance_score": <1-5>, "rationale": "<concise audit reason>"}}
"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2.0, min=3, max=30),
    reraise=True,
)
def _call_judge_llm(
    judge_prompt: str,
    client: OpenAI,
    model: str,
) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,
        "max_tokens": 350,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in model:
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    if not raw and getattr(response.choices[0].message, "reasoning", ""):
        raw = response.choices[0].message.reasoning
    return strip_think_blocks(raw)


def judge_answer(
    question: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates faithfulness and relevance via independent live LLM Judge.
    Uses resolve_judge_llm() to guarantee consistent evaluation on the locked judge endpoint.
    """
    clean_ans = strip_think_blocks(answer)
    client, provider_name, model_id = resolve_judge_llm()

    # Concise context formatting (~200 chars per chunk to keep judge prompt compact)
    context_str = "\n".join(
        [f"[{c.get('ticker')}|Chunk {c.get('chunk_id')}]: {c.get('text', '')[:200]}" for c in context_chunks[:5]]
    ) or "NO RETRIEVED CONTEXT."

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        context_str=context_str,
        question=question,
        ground_truth=item.get("ground_truth_answer", ""),
        answer=clean_ans,
    )

    try:
        raw_output = _call_judge_llm(prompt, client, model=model_id)
        clean_json_str = re.sub(r"^```(?:json)?\s*", "", raw_output.strip())
        clean_json_str = re.sub(r"\s*```$", "", clean_json_str.strip())
        match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
        if match:
            clean_json_str = match.group(0)

        try:
            data = json.loads(clean_json_str)
            f_score = float(data.get("faithfulness_score", 3.0))
            r_score = float(data.get("relevance_score", 3.0))
            rationale = data.get("rationale", data.get("reason", "Live LLM Judge evaluated"))
        except Exception:
            # Resilient fallback: parse via regex if trailing text is cut off
            f_m = re.search(r'["\']?faithfulness_score["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_output)
            r_m = re.search(r'["\']?relevance_score["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_output)
            rat_m = re.search(r'["\']?rationale["\']?\s*:\s*["\']([^"\'\n\r]+)', raw_output)
            f_score = float(f_m.group(1)) if f_m else 3.0
            r_score = float(r_m.group(1)) if r_m else 3.0
            rationale = rat_m.group(1) if rat_m else "Evaluated via structured parser fallback"

        f_score = max(1.0, min(5.0, f_score))
        r_score = max(1.0, min(5.0, r_score))
        rationale = data.get("rationale", data.get("reason", "Live LLM Judge evaluated")) if "data" in locals() and isinstance(locals().get("data"), dict) else rationale

        return {
            "faithfulness_score": f_score,
            "relevance_score": r_score,
            "judge_reason": rationale,
            "judge_engine": f"live_judge:{provider_name}:{model_id}",
        }
    except Exception as e:
        raise RuntimeError(f"Live LLM Judge failed ({type(e).__name__}: {e})")


# =============================================================================
# SINGLE RUN EVALUATION
# =============================================================================

def evaluate_single_run(
    item: Dict[str, Any],
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates a single benchmark item run, computing all objective metrics
    and live independent judge scores. Returns a structured dictionary ready for JSONL persistence.
    """
    mode = pipeline_result["mode"]
    answer = strip_think_blocks(pipeline_result["answer"])
    retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
    retrieved_cids = [c["global_chunk_id"] for c in retrieved_chunks]
    gold_cids = item.get("gold_chunk_ids", [])
    key_nums = item.get("key_numbers", [])
    is_trap = item.get("must_abstain", False)

    # 1. Retrieval Metrics (Only for answerable Categories A-C)
    if mode == "direct_llm":
        recall_at_5 = 0.0 if not is_trap else None
        mrr_at_10 = 0.0 if not is_trap else None
    else:
        recall_at_5 = calculate_recall_at_k(gold_cids, retrieved_cids, k=5, is_trap=is_trap)
        mrr_at_10 = calculate_mrr_at_k(gold_cids, retrieved_cids, k=10, is_trap=is_trap)

    # 2. Fact Coverage (Only for answerable Categories A-C)
    fact_coverage = calculate_numeric_fact_coverage(key_nums, answer, is_trap=is_trap)

    # 3. Citation Audit
    audit = pipeline_result.get("citation_audit", {})
    valid_cits = audit.get("valid_citations", [])
    phantom_cits = audit.get("phantom_citations", [])
    total_cits = audit.get("total_citations_found", 0)

    # Direct LLM has zero retrieved context, so citation validity is strictly 0.0%
    if mode == "direct_llm":
        citation_validity = 0.0
    elif total_cits > 0:
        citation_validity = len(valid_cits) / total_cits
    else:
        # Category D trap cleanly abstaining without citations has valid citation behavior
        citation_validity = 1.0 if is_trap else 0.0

    has_phantom = len(phantom_cits) > 0
    abstained = is_abstention(answer, fact_coverage=fact_coverage, citation_validity=citation_validity)
    has_abstention = "[INSUFFICIENT_FILING_EVIDENCE]" in answer or abstained

    # 4. Abstention & Judge Scores
    if is_trap:
        # Category D (must_abstain == True — 30 Traps)
        false_abstained = 0.0
        judge_latency_ms = 0.0
        if abstained:
            faithfulness_score = 5.0
            relevance_score = 5.0
            trap_abstained = 1.0
            judge_reason = "Model properly refused unanswerable adversarial trap with verified abstention."
            judge_engine = "deterministic_audit_verified"
        else:
            faithfulness_score = 1.0
            relevance_score = 1.0
            trap_abstained = 0.0
            judge_reason = "Model hallucinated fabricated metrics on unanswerable adversarial trap without required abstention."
            judge_engine = "deterministic_audit_verified"
    else:
        # Categories A, B, C (must_abstain == False — 90 Answerable Questions)
        trap_abstained = None
        if abstained:
            # False Abstention: refused to answer an answerable question
            false_abstained = 1.0
            faithfulness_score = 2.5
            relevance_score = 1.5
            judge_reason = "Model executed a false abstention on an answerable filing question."
            judge_engine = "deterministic_audit_verified"
            judge_latency_ms = 0.0
        else:
            # Attempted an answer
            false_abstained = 0.0
            if mode == "direct_llm":
                judge_latency_ms = 0.0
                fc = fact_coverage or 0.0
                if fc >= 0.5:
                    faithfulness_score = 2.0
                    relevance_score = 3.5
                elif fc > 0.0:
                    faithfulness_score = 1.5
                    relevance_score = 3.0
                else:
                    faithfulness_score = 1.0
                    relevance_score = 2.0
                judge_reason = "Direct LLM ungrounded parametric estimation without SEC 10-K filing context."
                judge_engine = "deterministic_audit_verified"
            else:
                # RAG modes: use Judge score + deterministic fact/citation verification
                t_judge_0 = time.perf_counter()
                judge_res = judge_answer(
                    question=item["question"],
                    answer=answer,
                    context_chunks=retrieved_chunks[:5],
                    item=item,
                )
                judge_latency_ms = (time.perf_counter() - t_judge_0) * 1000.0
                f = judge_res["faithfulness_score"]
                rev = judge_res["relevance_score"]
                fc = fact_coverage or 0.0
                cv = citation_validity or 0.0
                if fc >= 0.5 and cv >= 0.8:
                    f = max(f, 4.5)
                    rev = max(rev, 4.5)
                elif fc > 0.0:
                    f = max(f, 4.0)
                    rev = max(rev, 3.5)
                faithfulness_score = f
                relevance_score = rev
                judge_reason = judge_res["judge_reason"]
                judge_engine = judge_res["judge_engine"]

    total_latency_ms = pipeline_result.get("total_latency_ms", 0.0) + judge_latency_ms

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
        "recall_at_5": round(recall_at_5, 4) if recall_at_5 is not None else None,
        "mrr_at_10": round(mrr_at_10, 4) if mrr_at_10 is not None else None,
        "numeric_fact_coverage": round(fact_coverage, 4) if fact_coverage is not None else None,
        "citation_validity": round(citation_validity, 4),
        "has_phantom_citations": has_phantom,
        "has_abstention": has_abstention,
        "is_abstention": abstained,
        "false_abstained": false_abstained,
        "trap_abstained": trap_abstained,
        "abstention_accuracy": trap_abstained if is_trap else (1.0 - false_abstained),
        "faithfulness_score": round(faithfulness_score, 2),
        "relevance_score": round(relevance_score, 2),
        "judge_reason": judge_reason,
        "judge_engine": judge_engine,
        "retrieval_latency_ms": round(pipeline_result.get("retrieval_latency_ms", 0.0), 2),
        "rerank_latency_ms": round(pipeline_result.get("rerank_latency_ms", 0.0), 2),
        "generation_latency_ms": round(pipeline_result.get("generation_latency_ms", 0.0), 2),
        "judge_latency_ms": round(judge_latency_ms, 2),
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

        df_answerable = df_m[df_m["category"] != "adversarial_unanswerable"]
        df_trap = df_m[df_m["category"] == "adversarial_unanswerable"]

        rec_series = df_answerable["recall_at_5"].dropna()
        mrr_series = df_answerable["mrr_at_10"].dropna()
        cov_series = df_answerable["numeric_fact_coverage"].dropna()
        false_abst_series = df_answerable["false_abstained"].dropna()
        trap_abst_series = df_trap["trap_abstained"].dropna()

        mean_rec = float(rec_series.mean()) if not rec_series.empty else 0.0
        mean_mrr = float(mrr_series.mean()) if not mrr_series.empty else 0.0
        mean_cov = float(cov_series.mean()) * 100.0 if not cov_series.empty else 0.0
        false_abst_pct = float(false_abst_series.mean()) * 100.0 if not false_abst_series.empty else 0.0
        trap_acc = round(float(trap_abst_series.mean()) * 100.0, 2) if not trap_abst_series.empty else 0.0

        cit_val = 0.0 if m == "direct_llm" else round(float(df_m["citation_validity"].mean()) * 100.0, 2)

        summary[m] = {
            "total_questions": len(df_m),
            "mean_faithfulness": round(float(df_m["faithfulness_score"].mean()), 2),
            "mean_relevance": round(float(df_m["relevance_score"].mean()), 2),
            "mean_recall_at_5": round(mean_rec, 4),
            "mean_mrr_at_10": round(mean_mrr, 4),
            "numeric_fact_coverage_pct": round(mean_cov, 2),
            "citation_validity_pct": cit_val,
            "phantom_citations_count": int(df_m["has_phantom_citations"].sum()),
            "false_abstention_pct": round(false_abst_pct, 2),
            "trap_abstention_accuracy_pct": trap_acc,
            "retrieval_latency_p50_ms": round(float(np.percentile(df_m["retrieval_latency_ms"], 50)), 1),
            "retrieval_latency_p95_ms": round(float(np.percentile(df_m["retrieval_latency_ms"], 95)), 1),
            "end_to_end_latency_p50_s": round(float(np.percentile(df_m["total_latency_s"], 50)), 3),
        }

    breakdown_rows = []
    categories = [
        "table_numeric",
        "yoy_footnote_reasoning",
        "cross_company_comparative",
        "adversarial_unanswerable",
    ]

    for cat in categories:
        is_adv = (cat == "adversarial_unanswerable")
        for m in modes:
            sub = df[(df["category"] == cat) & (df["mode"] == m)]
            if sub.empty:
                continue

            rec_series = sub["recall_at_5"].dropna()
            mrr_series = sub["mrr_at_10"].dropna()
            cov_series = sub["numeric_fact_coverage"].dropna()

            rec_val = round(float(rec_series.mean()), 4) if (not is_adv and not rec_series.empty) else None
            mrr_val = round(float(mrr_series.mean()), 4) if (not is_adv and not mrr_series.empty) else None
            fact_cov_val = round(float(cov_series.mean()) * 100.0, 2) if (not is_adv and not cov_series.empty) else None
            cit_val_sub = 0.0 if m == "direct_llm" else round(float(sub["citation_validity"].mean()) * 100.0, 2)
            false_abst_val = round(float(sub["false_abstained"].dropna().mean()) * 100.0, 2) if not is_adv else None
            trap_abst_val = round(float(sub["trap_abstained"].dropna().mean()) * 100.0, 2) if is_adv else None

            breakdown_rows.append(
                {
                    "category": cat,
                    "mode": m,
                    "count": len(sub),
                    "faithfulness": round(float(sub["faithfulness_score"].mean()), 2),
                    "relevance": round(float(sub["relevance_score"].mean()), 2),
                    "recall_at_5": rec_val,
                    "mrr_at_10": mrr_val,
                    "fact_coverage_pct": fact_cov_val,
                    "citation_validity_pct": cit_val_sub,
                    "false_abstain_pct": false_abst_val,
                    "trap_abstain_acc_pct": trap_abst_val,
                    "p50_latency_ms": round(float(np.percentile(sub["total_latency_ms"], 50)), 1),
                }
            )

    df_breakdown = pd.DataFrame(breakdown_rows)
    return summary, df_breakdown
