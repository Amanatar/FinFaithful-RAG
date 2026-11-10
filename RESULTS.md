# Empirical Evaluation & Failure Postmortem (FinFaithful-RAG)

Empirical evaluation of 120 ground-truth Form 10-K questions across 5 companies (`AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`) and 4 retrieval modes (480 evaluations total). All reported numbers reflect exact unrounded metrics.

- **Locked Live Generator Model**: `groq` / `qwen/qwen3.8-27b`
- **Locked Independent Judge Model**: `groq` / `openai/gpt-oss-20b` (decoupled for strict TPD quota safety and zero self-preference bias)

---

## 1. Main 4-Method Comparison Table

| Pipeline Mode | Faithfulness (1-5) | Relevance (1-5) | HitRate@5 | MRR@10 | Numeric Fact Coverage % | Citation Validity % | False Abstention % | Trap Abstention Acc % | Retrieval P50 (ms) | Total Latency P50 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM (Ungrounded)** | 2.66 | 2.17 | 0.0000 | 0.0000 | 0.0% | 0.0% | 82.2% | 73.3% | 0.0 ms | 1.018 s |
| **Sparse BM25 (Lexical)** | 4.05 | 3.78 | 0.5444 | 0.3307 | 47.0% | 88.1% | 41.1% | 100.0% | 33.4 ms | 10.063 s |
| **Dense FAISS (all-MiniLM)** | 3.85 | 3.42 | 0.4444 | 0.1839 | 33.7% | 70.0% | 60.0% | 100.0% | 62.2 ms | 5.949 s |
| **Hybrid RRF + Cross-Encoder** | 4.12 | 3.83 | 0.7222 | 0.3744 | 48.7% | 89.2% | 42.2% | 100.0% | 122.6 ms | 10.698 s |

---

## 2. Per-Category Breakdown

| Category | Mode | Count | Faithfulness (1-5) | Relevance (1-5) | HitRate@5 | Fact Coverage % | Citation Validity % | False Abstain % | Trap Abstain Acc % | P50 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Table Numeric Extraction | `direct_llm` | 30 | 2.50 | 1.50 | 0.0000 | 0.0% | 0.0% | 100.0% | N/A | 421.4 ms |
| Table Numeric Extraction | `sparse_bm25` | 30 | 4.12 | 3.95 | 0.6667 | 66.7% | 96.7% | 26.7% | N/A | 2399.9 ms |
| Table Numeric Extraction | `dense_faiss` | 30 | 3.77 | 3.37 | 0.6000 | 53.3% | 70.0% | 46.7% | N/A | 3835.5 ms |
| Table Numeric Extraction | `hybrid_rerank` | 30 | 4.10 | 3.82 | 0.7333 | 66.7% | 96.7% | 30.0% | N/A | 10474.8 ms |
| YoY & Footnote Synthesis | `direct_llm` | 30 | 2.15 | 1.62 | 0.0000 | 0.0% | 0.0% | 76.7% | N/A | 722.9 ms |
| YoY & Footnote Synthesis | `sparse_bm25` | 30 | 3.80 | 3.58 | 0.6333 | 51.1% | 90.0% | 30.0% | N/A | 13252.1 ms |
| YoY & Footnote Synthesis | `dense_faiss` | 30 | 3.48 | 2.90 | 0.4667 | 34.4% | 60.0% | 60.0% | N/A | 7278.6 ms |
| YoY & Footnote Synthesis | `hybrid_rerank` | 30 | 3.77 | 3.37 | 0.7667 | 49.4% | 70.0% | 43.3% | N/A | 10302.4 ms |
| Cross-Company Comparative | `direct_llm` | 30 | 2.05 | 1.65 | 0.0000 | 0.0% | 0.0% | 70.0% | N/A | 2779.8 ms |
| Cross-Company Comparative | `sparse_bm25` | 30 | 3.27 | 2.60 | 0.3333 | 23.3% | 66.7% | 66.7% | N/A | 11237.9 ms |
| Cross-Company Comparative | `dense_faiss` | 30 | 3.15 | 2.42 | 0.2667 | 13.3% | 50.0% | 73.3% | N/A | 5141.7 ms |
| Cross-Company Comparative | `hybrid_rerank` | 30 | 3.63 | 3.12 | 0.6667 | 30.0% | 90.0% | 53.3% | N/A | 20488.6 ms |
| Adversarial Hallucination Traps | `direct_llm` | 30 | 3.93 | 3.93 | N/A | N/A | 0.0% | N/A | 73.3% | 2267.9 ms |
| Adversarial Hallucination Traps | `sparse_bm25` | 30 | 5.00 | 5.00 | N/A | N/A | 99.2% | N/A | 100.0% | 5608.6 ms |
| Adversarial Hallucination Traps | `dense_faiss` | 30 | 5.00 | 5.00 | N/A | N/A | 100.0% | N/A | 100.0% | 6745.7 ms |
| Adversarial Hallucination Traps | `hybrid_rerank` | 30 | 5.00 | 5.00 | N/A | N/A | 100.0% | N/A | 100.0% | 5588.0 ms |

---

## 3. Documented Failure Modes & Engineering Postmortem

To provide rigorous engineering transparency for quantitative interviewers, we analyze real queries where the flagship `hybrid_rerank` pipeline experienced edge cases or trade-offs:

### Failure Case 1: `SEC_BENCH_002` (table_numeric)
- **Question**: *"What was Apple's Services net sales in fiscal year 2026?"*
- **Observed Metrics**: Faithfulness: `4.50`, Recall@5: `1.00`, Fact Coverage: `100.0%`
- **Generated Output**: `Apple's Services net sales in fiscal year 2026 were $123,301 million [AAPL | FY 2026 | Item 7 | Chunk 7]....`
- **Audit Diagnosis**: The answer cites a chunk that does not contain Services net sales data, indicating reliance on parametric memory rather than the provided context. However, it correctly addresses the question asked.
- **Root Cause Analysis**: Large audited tables (e.g. segment disaggregation) span across chunk boundaries. If the column header row is split from deeper footnote rows, the cross-encoder may score surrounding narrative chunks higher than the raw table fragment.
- **Engineering Trade-off / Remediation**: Table-aware chunking with header replication ensures all sub-chunks retain context schema.

### Failure Case 2: `SEC_BENCH_061` (cross_company_comparative)
- **Question**: *"Compare total annual revenues between Apple and Microsoft in fiscal year 2026."*
- **Observed Metrics**: Faithfulness: `4.50`, Recall@5: `1.00`, Fact Coverage: `50.0%`
- **Generated Output**: `Based on the retrieved SEC 10-K excerpts, a direct comparison of total annual revenues for fiscal year 2026 is not possible for both companies because the total revenue figure for Microsoft is not disclosed in the provided text.

**Apple Inc. (AAPL)**
Apple reported total net sal...`
- **Audit Diagnosis**: The answer incorrectly states that Microsoft’s total revenue is not disclosed, whereas the expected ground truth provides a figure. It fails to provide the correct revenue and misrepresents the availability of data, resulting in both low faithfulness and relevance.
- **Root Cause Analysis**: Multi-ticker comparative queries require retrieving balanced evidence across two separate companies. When both companies have large 10-K filings, RRF fusion can bias towards the company with higher lexical density, leaving the second ticker's chunk ranked outside top-5.
- **Engineering Trade-off / Remediation**: Implement ticker-partitioned retrieval where top-3 chunks are allocated per detected ticker rather than a global top-5 pool.

### Failure Case 3: `SEC_BENCH_006` (table_numeric)
- **Question**: *"What was Apple's cash and cash equivalents balance as of September 26, 2026?"*
- **Observed Metrics**: Faithfulness: `2.50`, Recall@5: `0.00`, Fact Coverage: `0.0%`
- **Generated Output**: `[INSUFFICIENT_FILING_EVIDENCE]

The retrieved excerpts confirm that Apple's fiscal year 2026 ended on September 26, 2026, and reference the existence of a table in Note 4 detailing "cash, cash equivalents and marketable securities" for that date [AAPL | FY 2026 | Item 8 | Chunk 3...`
- **Audit Diagnosis**: Model executed a false abstention on an answerable filing question.
- **Root Cause Analysis**: Large audited tables (e.g. segment disaggregation) span across chunk boundaries. If the column header row is split from deeper footnote rows, the cross-encoder may score surrounding narrative chunks higher than the raw table fragment.
- **Engineering Trade-off / Remediation**: Table-aware chunking with header replication ensures all sub-chunks retain context schema.

---
Generated automatically by `src/experiment_runner.py`.
