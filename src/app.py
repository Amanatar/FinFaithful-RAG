"""
FinFaithful-RAG: Production Streamlit Workbench (Phase 4.5 Release)
3-Tab Architecture:
  Tab 1: Live Cited Q&A + Preset Adversarial Trap Demo
  Tab 2: 120-Question Empirical Benchmark Leaderboard
  Tab 3: Failure-Mode Audit Explorer & Lineage Inspector
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env", override=True)

from pipeline_manager import query_pipeline
from ticker_lookup import get_company_name

st.set_page_config(
    page_title="FinFaithful-RAG: Financial NLP Workbench",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .citation-badge {
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .abstention-banner {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_JSON = RESULTS_DIR / "summary_metrics.json"
CATEGORY_CSV = RESULTS_DIR / "category_breakdown.csv"
BENCHMARK_JSONL = RESULTS_DIR / "benchmark_runs_120.jsonl"


# =============================================================================
# HELPER DATA LOADERS
# =============================================================================

@st.cache_data
def load_summary_metrics() -> Dict[str, Any]:
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_category_breakdown() -> pd.DataFrame:
    if CATEGORY_CSV.exists():
        return pd.read_csv(CATEGORY_CSV)
    return pd.DataFrame()


@st.cache_data
def load_benchmark_runs() -> List[Dict[str, Any]]:
    if BENCHMARK_JSONL.exists():
        runs = []
        with open(BENCHMARK_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        runs.append(json.loads(line))
                    except Exception:
                        pass
        return runs
    return []


def _safe_get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return ""


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### 🏦 **FinFaithful-RAG**")
    st.caption("Auditable, Table-Preserving SEC 10-K RAG Engine")
    st.divider()

    st.markdown("#### **Corpus Overview**")
    st.markdown(
        """
        - **Tickers**: `AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`
        - **Index Artifacts**: FAISS (3.6 MB) + BM25 (5.0 MB)
        - **Total Chunks**: 2,469 audited segments
        - **Embeddings**: `all-MiniLM-L6-v2`
        - **Reranker**: `ms-marco-MiniLM-L-6-v2`
        """
    )
    st.divider()

    # Detect Credentials
    groq_k = _safe_get_secret("GROQ_API_KEY")
    hf_k = _safe_get_secret("HF_TOKEN")
    nv_k = _safe_get_secret("NVIDIA_API_KEY")

    active_provider = "None (Inspection Only)"
    if groq_k:
        active_provider = "Groq (`llama-3.3-70b-versatile`)"
    elif hf_k:
        active_provider = "Hugging Face Router (`Llama-3.1-8B`)"
    elif nv_k:
        active_provider = "NVIDIA NIM (`llama-3.1-70b-instruct`)"

    st.markdown(f"**Live Engine**: {active_provider}")
    st.caption("Configured via `.env` or Streamlit Cloud `st.secrets`.")


# =============================================================================
# MAIN TABS WORKBENCH
# =============================================================================

st.markdown('<div class="main-title">FinFaithful-RAG Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production SEC Form 10-K Retrieval with Strict Lineage Verification & Hallucination Guardrails</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Cited Q&A & Adversarial Traps",
    "📊 120-Question Benchmark Leaderboard",
    "🔬 Failure-Mode Audit Explorer",
])


# =============================================================================
# TAB 1: LIVE CITED Q&A
# =============================================================================

with tab1:
    col_mode, col_preset = st.columns([1, 2])

    with col_mode:
        selected_mode = st.selectbox(
            "Retrieval Pipeline Mode",
            ["hybrid_rerank", "dense_faiss", "sparse_bm25", "direct_llm"],
            index=0,
            help="hybrid_rerank = Top-20 BM25 + Top-20 FAISS fused via RRF & Cross-Encoder reranked to top-5.",
        )

    with col_preset:
        preset_choice = st.selectbox(
            "Quick-Fill Example Queries",
            [
                "-- Select a curated financial question or adversarial trap --",
                "[Item 8 Table] What were Apple's total net sales in fiscal 2026?",
                "[Item 8 Table] What were Microsoft's Intelligent Cloud segment revenues in fiscal 2026?",
                "[Comparative] Compare research and development expenses between Nvidia and Microsoft in fiscal 2026.",
                "[Adversarial Trap] What is Nvidia's projected Q3 2028 Blackwell architecture gross margin guidance?",
                "[Adversarial Trap] What was Jamie Dimon's personal ATM cash withdrawal limit on December 24?",
                "[Adversarial Trap] What was Alphabet's daily employee voluntary attrition rate in fiscal 2025?",
            ],
        )

    initial_q = ""
    if preset_choice and not preset_choice.startswith("--"):
        initial_q = re.sub(r"^\[.*?\]\s*", "", preset_choice)

    user_query = st.text_input(
        "Enter Financial Query or Audit Question:",
        value=initial_q,
        placeholder="e.g., What was JPMorgan Chase's Net Interest Income in fiscal 2025?",
    )

    if st.button("Run Grounded Financial Audit", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner(f"Executing retrieval & generation in `{selected_mode}` mode..."):
                try:
                    res = query_pipeline(
                        query=user_query,
                        mode=selected_mode,
                        top_k=5,
                    )

                    ans = res.get("answer", "")
                    ret_chunks = res.get("retrieved_chunks", [])
                    tot_lat = res.get("total_latency_ms", 0.0)

                    # Check for abstention
                    if "[INSUFFICIENT_FILING_EVIDENCE]" in ans:
                        st.markdown(
                            f"""
                            <div class="abstention-banner">
                                <strong>🛡️ Guardrail Abstention Triggered:</strong><br>
                                {ans}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("### Audited Financial Answer")
                        st.markdown(ans)

                    # Performance Metrics
                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Latency", f"{tot_lat:.1f} ms")
                    m2.metric("Retrieval Latency", f"{res.get('retrieval_latency_ms', 0.0):.1f} ms")
                    m3.metric("Rerank Latency", f"{res.get('rerank_latency_ms', 0.0):.1f} ms")
                    m4.metric("Generation Latency", f"{res.get('generation_latency_ms', 0.0):.1f} ms")

                    # Retrieved Chunks Accordion
                    with st.expander(f"📑 Retrieved 10-K Evidence Chunks ({len(ret_chunks)} chunks)"):
                        for idx, c in enumerate(ret_chunks, 1):
                            t_tag = f"[{c.get('ticker')} | FY {c.get('fiscal_year')} | {c.get('item_section')} | Chunk {c.get('chunk_id')}]"
                            score_info = f"Score: {c.get('score', 0.0):.4f}" if "score" in c else ""
                            st.markdown(f"**Chunk {idx}: `{t_tag}`** {score_info}")
                            st.markdown(c.get("text", ""))
                            st.divider()

                except Exception as e:
                    st.error(f"Execution Error: {e}")


# =============================================================================
# TAB 2: 120-QUESTION BENCHMARK LEADERBOARD
# =============================================================================

with tab2:
    st.markdown("### 🏆 120-Question SEC 10-K Ground-Truth Benchmark Leaderboard")
    st.markdown(
        "Empirical 4-method comparison evaluated across 120 ground-truth questions and 5 reporting structures (`AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`)."
    )

    summary_data = load_summary_metrics()
    df_cat = load_category_breakdown()

    if summary_data:
        rows = []
        for m, s in summary_data.items():
            rows.append(
                {
                    "Mode": m,
                    "Faithfulness (1-5)": s.get("mean_faithfulness"),
                    "Relevance (1-5)": s.get("mean_relevance"),
                    "Recall@5": s.get("mean_recall_at_5"),
                    "MRR@10": s.get("mean_mrr_at_10"),
                    "Fact Coverage %": f"{s.get('numeric_fact_coverage_pct', 0.0):.1f}%",
                    "Citation Validity %": f"{s.get('citation_validity_pct', 0.0):.1f}%",
                    "Trap Abstention %": f"{s.get('trap_abstention_accuracy_pct', 0.0):.1f}%",
                    "P50 Retrieval (ms)": f"{s.get('retrieval_latency_p50_ms', 0.0):.1f}",
                    "P50 Total (s)": f"{s.get('end_to_end_latency_p50_s', 0.0):.3f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Benchmark summary results will appear here once `results/summary_metrics.json` is generated.")

    st.markdown("#### 📂 Category Performance Breakdown")
    if not df_cat.empty:
        st.dataframe(df_cat, use_container_width=True)
    else:
        st.caption("Category breakdown data available in `results/category_breakdown.csv`.")


# =============================================================================
# TAB 3: FAILURE-MODE AUDIT EXPLORER
# =============================================================================

with tab3:
    st.markdown("### 🔬 Failure-Mode Audit Explorer & Lineage Inspector")
    st.markdown(
        "Inspect individual benchmark query runs where `hybrid_rerank` scored < 5.0 Faithfulness or missed gold chunk IDs."
    )

    all_runs = load_benchmark_runs()
    if all_runs:
        df_runs = pd.DataFrame(all_runs)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_mode = st.selectbox("Filter Mode", ["hybrid_rerank", "dense_faiss", "sparse_bm25", "direct_llm"])
        with col_f2:
            filter_cat = st.selectbox(
                "Filter Category",
                ["All", "table_numeric", "yoy_footnote_reasoning", "cross_company_comparative", "adversarial_unanswerable"],
            )

        sub_df = df_runs[df_runs["mode"] == filter_mode]
        if filter_cat != "All":
            sub_df = sub_df[sub_df["category"] == filter_cat]

        st.caption(f"Showing {len(sub_df)} runs matching filter criteria.")

        for _, item in sub_df.head(15).iterrows():
            with st.expander(f"[{item['category']}] `{item['question_id']}`: {item['question'][:80]}..."):
                st.markdown(f"**Full Question**: {item['question']}")
                st.markdown(f"**Ground Truth Answer**: {item.get('ground_truth_answer', 'N/A')}")
                st.markdown(f"**Expected Numbers**: `{item.get('key_numbers', [])}`")
                st.markdown(f"**Gold Chunk IDs**: `{item.get('gold_chunk_ids', [])}`")
                st.markdown(f"**Retrieved Chunk IDs**: `{item.get('top_5_chunk_ids', [])}`")
                st.markdown(f"**Observed Recall@5**: `{item.get('recall_at_5')}` | **Fact Coverage**: `{item.get('numeric_fact_coverage')}`")
                st.markdown(f"**Faithfulness Score**: `{item.get('faithfulness_score')}` | **Relevance**: `{item.get('relevance_score')}`")
                st.markdown(f"**Judge Rationale**: *{item.get('judge_reason', 'N/A')}*")
                st.markdown(f"**Generated Answer**:\n> {item.get('answer', '')}")
    else:
        st.info("No run logs found in `results/benchmark_runs_120.jsonl`.")
