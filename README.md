# FinFaithful-RAG: SEC 10-K Grounded Retrieval & Faithfulness Benchmark

<div align="center">

[![CI Verified](https://github.com/Amanatar/FinFaithful-RAG/actions/workflows/ci-verify.yml/badge.svg)](https://github.com/Amanatar/FinFaithful-RAG/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Ask an ungrounded LLM for audited financial line items from an SEC Form 10-K filing — YoY segment revenue growth, debt maturities, capitalization footnotes, or confidential unit economics — and it will confidently fabricate numbers or extrapolate from outdated parametric memory. On our 120-question Form 10-K benchmark, ungrounded direct LLM prompting achieved **0.0% verified citation validity** and failed completely on adversarial hallucination traps.

**FinFaithful-RAG** is an audited, production-grade financial retrieval and verification engine built over real multi-ticker SEC Form 10-K filings (`AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`). It benchmarks lexical, dense vector, and cross-encoder reranked hybrid retrieval against an independent, decoupled LLM Judge grading pipeline across **120 ground-truth questions** and **480 live evaluations**.

---

## 🏆 Key Empirical Benchmark Findings

Across 120 audited questions spanning 4 strict categories (30 questions each: Table Numeric, YoY Footnote Synthesis, Cross-Company Comparative, and Adversarial Traps):

```
Retrieval HitRate@5 Progression (Categories A–C, 90 Questions):
  ├── Dense FAISS (all-MiniLM-L6-v2) : 44.44% | MRR@10: 0.1839
  ├── Sparse BM25 (BM25Okapi)         : 54.44% | MRR@10: 0.3307
  └── Hybrid RRF + Cross-Encoder     : 72.22% | MRR@10: 0.3744  (Benchmark Target Met >= 0.70)
```

### Main 4-Pipeline Comparison Table

| Pipeline Mode | Faithfulness (1-5) | Relevance (1-5) | HitRate@5 | MRR@10 | Numeric Fact Coverage % | Citation Validity % | False Abstention % | Trap Abstention Acc % | Retrieval P50 (ms) | Total Latency P50 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM (Ungrounded)** | 2.66 | 2.17 | 0.0000 | 0.0000 | 0.0% | 0.0% | 82.2% | 73.3% | 0.0 ms | 1.018 s |
| **Dense FAISS (all-MiniLM)** | 3.85 | 3.42 | 0.4444 | 0.1839 | 33.7% | 70.0% | 60.0% | 100.0% | 62.2 ms | 5.949 s |
| **Sparse BM25 (Lexical)** | 4.05 | 3.78 | 0.5444 | 0.3307 | 47.0% | 88.1% | 41.1% | 100.0% | 33.4 ms | 10.063 s |
| **Hybrid RRF + Cross-Encoder** | **4.12** | **3.83** | **0.7222** | **0.3744** | **48.7%** | **89.2%** | **42.2%** | **100.0%** | 122.6 ms | 10.698 s |

> *Full unrounded evaluation metrics and failure postmortems are documented in [RESULTS.md](RESULTS.md).*

---

## 🏛️ System Architecture

```text
SEC EDGAR API (Official Submissions & Archives)
       │
       ▼
Table-Preserving Section Parser (Item 1, 1A, 7 MD&A, 7A, 8 Financial Statements)
       │
       ├──► HTML Table-to-Markdown Converter (replicates column years + statement titles)
       └──► Row Contextualizer: "[Ticker] ([Company]) [Statement] covering [Rows] for [Years]"
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
Dense FAISS Index (all-MiniLM-L6-v2)      Sparse BM25 Index (BM25Okapi)
(384-dim IndexFlatIP normalized)          (Section-boosted Item 8/7 tables x1.35)
       │                                          │
       └───────────────────┬──────────────────────┘
                           ▼
               Reciprocal Rank Fusion (RRF: k=60)
               + Per-Ticker Balanced Allocation (K/M)
                           │
                           ▼
        Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
        (Precision top-5 selection on CPU)
                           │
                           ▼
          Decoupled Live Generation & Audit Engine
  ┌────────────────────────────────────────────────────────┐
  │ Generator Model: groq:qwen/qwen3.8-27b                │
  │ Inline Bracket Citations: [Ticker|FY|Item|ChunkID]     │
  │ Explicit Abstention: [INSUFFICIENT_FILING_EVIDENCE]    │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │ Independent Judge Model: groq:openai/gpt-oss-20b               │
  │ Structured Rubric JSON (Faithfulness 1-5, Rel 1-5)    │
  │ <think> reasoning block stripping before audit scoring │
  └────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Financial NLP Innovations

### 1. Table-Aware Header Replication & Row Contextualization
Audited financial statements (Consolidated Balance Sheets, Statements of Operations, Segment Disaggregation) are often flattened into meaningless text cells in traditional RAG. FinFaithful-RAG converts tables into clean pipe-delimited Markdown, dynamically replicating the table title, fiscal year column headers, and separator rows across every chunk fragment. Furthermore, each chunk is prepended with a dense natural language index summary:
```text
[Apple Inc. (AAPL) Form 10-K Consolidated Statements of Operations table covering iPhone, Mac, iPad, Wearables, Services, Total net sales for fiscal periods 2026, 2025, 2024]
```
This enables dense vector embeddings to directly match natural language queries like *"What was Apple's total iPhone net sales in fiscal 2026?"* with rank-1 precision.

### 2. Multi-Ticker Balanced Quota Partitioning
When querying comparative financial metrics (e.g. *"Compare Cloud revenue growth between Microsoft and Alphabet"*), naive RRF fusion suffers from entity starvation where the company with higher lexical density dominates the top-5 pool. FinFaithful-RAG dynamically partitions candidate retrieval quotas:
$$\text{quota per ticker} = \left\lceil \frac{K}{M} \right\rceil$$
guaranteeing balanced multi-company context allocation.

### 3. Decoupled Live Generation & Independent Audit
To avoid per-model Token-Per-Day (TPD) rate-limit exhaustion and eliminate LLM self-preference bias, generation and evaluation are decoupled into two distinct model architectures:
- **Locked Generator**: `qwen/qwen3.8-27b` conditioned with a strict system prompt requiring inline bracket citations `[Ticker | FY {year} | {section} | Chunk {id}]`.
- **Locked Independent Judge**: `openai/gpt-oss-20b` running a deterministic JSON rubric (`max_tokens=250`, `temperature=0.0`) with automatic `<think>...</think>` block stripping.

---

## 💻 Streamlit 3-Tab Workbench

FinFaithful-RAG includes a production Streamlit application designed for deployment directly on **Streamlit Community Cloud (`share.streamlit.io`)** with committed index artifacts under 35 MB:

1. **Tab 1 — Live Cited Q&A & Adversarial Traps:**
   Interactive querying across `AAPL`, `MSFT`, `NVDA`, `JPM`, and `GOOGL` with real-time bracket citation validation, latency counters, and preset adversarial trap demonstrations.
2. **Tab 2 — 120-Question Benchmark Leaderboard:**
   Interactive performance dashboards detailing Recall@5, MRR@10, Faithfulness, Fact Coverage %, and P50 latencies across all 4 categories.
3. **Tab 3 — Failure-Mode Audit Explorer:**
   Deep inspection tool allowing financial auditors to drill into specific queries, examining gold chunks vs retrieved chunks, generated answers, and judge rationales.

```bash
streamlit run src/app.py
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Amanatar/FinFaithful-RAG.git
cd FinFaithful-RAG

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `.env` file (strictly gitignored):

```env
# Free Groq API Key (instant at https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_key_here

# HuggingFace Token (for model weights & rate-limit elevation)
HF_TOKEN=hf_your_token_here
```

### 3. Run Benchmark Suite

```bash
# Verify live endpoint connectivity
python -c "import sys; sys.path.insert(0, 'src'); from llm_client import assert_live_models; print(assert_live_models())"

# Execute full 480 evaluations
python src/experiment_runner.py --dataset data/eval/sec_10k_benchmark_120.jsonl --all-modes
```

### 4. Run CI Test Suite

```bash
pytest tests/test_vm_execution.py
```

---

## 📄 License & Attribution

Distributed under the **MIT License**. Created by [Aman Atar](https://github.com/Amanatar).
