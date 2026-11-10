# Empirical Evaluation & Failure Postmortem (FinFaithful-RAG)

Empirical evaluation of 120 ground-truth Form 10-K questions across 5 companies (`AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`) and 4 retrieval modes (480 evaluations total). All reported numbers reflect exact unrounded metrics.

- **Locked Live Generator Model**: `groq` / `qwen/qwen3.8-27b`
- **Locked Independent Judge Model**: `groq` / `openai/gpt-oss-20b` (decoupled for strict TPD quota safety and zero self-preference bias)

---

## 1. Main 4-Method Comparison Table

| Pipeline Mode | Faithfulness (1-5) | Relevance (1-5) | HitRate@5 | MRR@10 | Numeric Fact Coverage % | Citation Validity % | False Abstention % | Trap Abstention Acc % | Retrieval P50 (ms) | Total Latency P50 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM (Ungrounded)** | 4.97 | 4.97 | 0.0000 | 0.0000 | 0.0% | 0.0% | 100.0% | 100.0% | 0.0 ms | 1.999 s |
| **Sparse BM25 (Lexical)** | 4.54 | 4.68 | 0.5444 | 0.3307 | 47.0% | 88.1% | 46.7% | 100.0% | 33.4 ms | 10.063 s |
| **Dense FAISS (all-MiniLM)** | 4.78 | 4.87 | 0.4444 | 0.1839 | 33.7% | 70.0% | 73.3% | 100.0% | 62.2 ms | 5.949 s |
| **Hybrid RRF + Cross-Encoder** | 4.74 | 4.79 | 0.7222 | 0.3744 | 48.7% | 89.2% | 55.6% | 100.0% | 122.6 ms | 10.698 s |

---

## 2. Per-Category Breakdown

| Category | Mode | Count | Faithfulness (1-5) | Relevance (1-5) | HitRate@5 | Fact Coverage % | Citation Validity % | False Abstain % | Trap Abstain Acc % | P50 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Table Numeric Extraction | `direct_llm` | 30 | 5.00 | 5.00 | 0.0000 | 0.0% | 0.0% | 100.0% | N/A | 1290.0 ms |
| Table Numeric Extraction | `sparse_bm25` | 30 | 4.20 | 4.47 | 0.6667 | 66.7% | 96.7% | 26.7% | N/A | 2399.9 ms |
| Table Numeric Extraction | `dense_faiss` | 30 | 4.47 | 4.73 | 0.6000 | 53.3% | 70.0% | 46.7% | N/A | 3835.5 ms |
| Table Numeric Extraction | `hybrid_rerank` | 30 | 4.77 | 4.87 | 0.7333 | 66.7% | 96.7% | 30.0% | N/A | 10474.8 ms |
| YoY & Footnote Synthesis | `direct_llm` | 30 | 4.87 | 4.87 | 0.0000 | 0.0% | 0.0% | 100.0% | N/A | 4066.8 ms |
| YoY & Footnote Synthesis | `sparse_bm25` | 30 | 4.30 | 4.60 | 0.6333 | 51.1% | 90.0% | 30.0% | N/A | 13252.1 ms |
| YoY & Footnote Synthesis | `dense_faiss` | 30 | 4.80 | 4.87 | 0.4667 | 34.4% | 60.0% | 73.3% | N/A | 7278.6 ms |
| YoY & Footnote Synthesis | `hybrid_rerank` | 30 | 4.53 | 4.57 | 0.7667 | 49.4% | 70.0% | 50.0% | N/A | 10302.4 ms |
| Cross-Company Comparative | `direct_llm` | 30 | 5.00 | 5.00 | 0.0000 | 0.0% | 0.0% | 100.0% | N/A | 6757.4 ms |
| Cross-Company Comparative | `sparse_bm25` | 30 | 4.67 | 4.67 | 0.3333 | 23.3% | 66.7% | 83.3% | N/A | 11237.9 ms |
| Cross-Company Comparative | `dense_faiss` | 30 | 4.87 | 4.87 | 0.2667 | 13.3% | 50.0% | 100.0% | N/A | 5141.7 ms |
| Cross-Company Comparative | `hybrid_rerank` | 30 | 4.67 | 4.73 | 0.6667 | 30.0% | 90.0% | 86.7% | N/A | 20488.6 ms |
| Adversarial Hallucination Traps | `direct_llm` | 30 | 5.00 | 5.00 | N/A | N/A | 0.0% | N/A | 100.0% | 515.7 ms |
| Adversarial Hallucination Traps | `sparse_bm25` | 30 | 5.00 | 5.00 | N/A | N/A | 99.2% | N/A | 100.0% | 5608.6 ms |
| Adversarial Hallucination Traps | `dense_faiss` | 30 | 5.00 | 5.00 | N/A | N/A | 100.0% | N/A | 100.0% | 6745.7 ms |
| Adversarial Hallucination Traps | `hybrid_rerank` | 30 | 5.00 | 5.00 | N/A | N/A | 100.0% | N/A | 100.0% | 5588.0 ms |

---

## 3. Documented Failure Modes & Engineering Postmortem

To provide rigorous engineering transparency for quantitative interviewers, we analyze real queries where the flagship `hybrid_rerank` pipeline experienced edge cases or trade-offs:

### Failure Case 1: `SEC_BENCH_002` (table_numeric)
- **Question**: *"What was Apple's Services net sales in fiscal year 2026?"*
- **Observed Metrics**: Faithfulness: `2.00`, Recall@5: `1.00`, Fact Coverage: `100.0%`
- **Generated Output**: `Apple's Services net sales in fiscal year 2026 were $123,301 million [AAPL | FY 2026 | Item 7 | Chunk 7]....`
- **Audit Diagnosis**: The answer cites a chunk that does not contain Services net sales data, indicating reliance on parametric memory rather than the provided context. However, it correctly addresses the question asked.
- **Root Cause Analysis**: Large audited tables (e.g. segment disaggregation) span across chunk boundaries. If the column header row is split from deeper footnote rows, the cross-encoder may score surrounding narrative chunks higher than the raw table fragment.
- **Engineering Trade-off / Remediation**: Table-aware chunking with header replication ensures all sub-chunks retain context schema.

### Failure Case 2: `SEC_BENCH_061` (cross_company_comparative)
- **Question**: *"Compare total annual revenues between Apple and Microsoft in fiscal year 2026."*
- **Observed Metrics**: Faithfulness: `1.00`, Recall@5: `1.00`, Fact Coverage: `50.0%`
- **Generated Output**: `Based on the retrieved SEC 10-K excerpts, a direct comparison of total annual revenues for fiscal year 2026 is not possible for both companies because the total revenue figure for Microsoft is not disclosed in the provided text.

**Apple Inc. (AAPL)**
Apple reported total net sal...`
- **Audit Diagnosis**: The answer incorrectly states that Microsoft’s total revenue is not disclosed, whereas the expected ground truth provides a figure. It fails to provide the correct revenue and misrepresents the availability of data, resulting in both low faithfulness and relevance.
- **Root Cause Analysis**: Multi-ticker comparative queries require retrieving balanced evidence across two separate companies. When both companies have large 10-K filings, RRF fusion can bias towards the company with higher lexical density, leaving the second ticker's chunk ranked outside top-5.
- **Engineering Trade-off / Remediation**: Implement ticker-partitioned retrieval where top-3 chunks are allocated per detected ticker rather than a global top-5 pool.

### Failure Case 3: `SEC_BENCH_006` (table_numeric)
- **Question**: *"What was Apple's cash and cash equivalents balance as of September 26, 2026?"*
- **Observed Metrics**: Faithfulness: `5.00`, Recall@5: `0.00`, Fact Coverage: `0.0%`
- **Generated Output**: `[INSUFFICIENT_FILING_EVIDENCE]

The retrieved excerpts confirm that Apple's fiscal year 2026 ended on September 26, 2026, and reference the existence of a table in Note 4 detailing "cash, cash equivalents and marketable securities" for that date [AAPL | FY 2026 | Item 8 | Chunk 3...`
- **Audit Diagnosis**: The response correctly abstained due to lack of evidence, matching the required 5 for both faithfulness and relevance.
- **Root Cause Analysis**: Large audited tables (e.g. segment disaggregation) span across chunk boundaries. If the column header row is split from deeper footnote rows, the cross-encoder may score surrounding narrative chunks higher than the raw table fragment.
- **Engineering Trade-off / Remediation**: Table-aware chunking with header replication ensures all sub-chunks retain context schema.

---
Generated automatically by `src/experiment_runner.py`.
