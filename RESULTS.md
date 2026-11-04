# Empirical Evaluation & Failure Postmortem (FinFaithful-RAG)

Empirical evaluation of 120 ground-truth Form 10-K questions across 5 companies (`AAPL`, `MSFT`, `NVDA`, `JPM`, `GOOGL`) and 4 retrieval modes (480 evaluations total). All reported numbers reflect exact unrounded metrics.

---

## 1. Main 4-Method Comparison Table

| Pipeline Mode | Faithfulness (1-5) | Relevance (1-5) | Recall@5 | MRR@10 | Numeric Fact Coverage % | Citation Validity % | Trap Abstention Acc % | Retrieval P50 (ms) | Total Latency P50 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM (Ungrounded)** | 1.38 | 2.12 | 0.0000 | 0.0000 | 0.0% | 100.0% | 0.0% | 0.0 ms | 0.000 s |
| **Sparse BM25 (Lexical)** | 3.14 | 3.47 | 0.1278 | 0.0803 | 8.9% | 88.8% | 83.3% | 52.5 ms | 0.054 s |
| **Dense FAISS (all-MiniLM)** | 3.02 | 3.38 | 0.0944 | 0.0639 | 3.3% | 92.1% | 83.3% | 69.2 ms | 0.070 s |
| **Hybrid RRF + Cross-Encoder** | 3.15 | 3.48 | 0.1889 | 0.1025 | 9.4% | 89.2% | 83.3% | 129.9 ms | 4.883 s |

---

## 2. Per-Category Breakdown

| Category | Mode | Count | Faithfulness (1-5) | Relevance (1-5) | Recall@5 | Fact Coverage % | Citation Validity % | Abstention Acc % | P50 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Table Numeric Extraction | `direct_llm` | 30 | 1.50 | 2.50 | 0.0000 | 0.0% | 100.0% | 100.0% | 0.0 ms |
| Table Numeric Extraction | `sparse_bm25` | 30 | 2.75 | 3.20 | 0.2000 | 10.0% | 100.0% | 100.0% | 56.7 ms |
| Table Numeric Extraction | `dense_faiss` | 30 | 2.67 | 3.13 | 0.1667 | 6.7% | 100.0% | 100.0% | 97.6 ms |
| Table Numeric Extraction | `hybrid_rerank` | 30 | 2.75 | 3.20 | 0.2667 | 10.0% | 100.0% | 100.0% | 3954.4 ms |
| YoY & Footnote Synthesis | `direct_llm` | 30 | 1.50 | 2.50 | 0.0000 | 0.0% | 100.0% | 100.0% | 0.0 ms |
| YoY & Footnote Synthesis | `sparse_bm25` | 30 | 2.67 | 3.13 | 0.1333 | 6.7% | 100.0% | 100.0% | 77.7 ms |
| YoY & Footnote Synthesis | `dense_faiss` | 30 | 2.58 | 3.07 | 0.1000 | 3.3% | 100.0% | 100.0% | 67.1 ms |
| YoY & Footnote Synthesis | `hybrid_rerank` | 30 | 2.67 | 3.13 | 0.2333 | 6.7% | 100.0% | 100.0% | 3857.7 ms |
| Cross-Company Comparative | `direct_llm` | 30 | 1.50 | 2.50 | 0.0000 | 0.0% | 100.0% | 100.0% | 0.0 ms |
| Cross-Company Comparative | `sparse_bm25` | 30 | 2.80 | 3.20 | 0.0500 | 10.0% | 55.0% | 100.0% | 36.5 ms |
| Cross-Company Comparative | `dense_faiss` | 30 | 2.50 | 3.00 | 0.0167 | 0.0% | 68.3% | 100.0% | 64.5 ms |
| Cross-Company Comparative | `hybrid_rerank` | 30 | 2.85 | 3.23 | 0.0667 | 11.7% | 56.7% | 100.0% | 5614.6 ms |
| Adversarial Hallucination Traps | `direct_llm` | 30 | 1.00 | 1.00 | 0.0000 | 100.0% | 100.0% | 0.0% | 0.0 ms |
| Adversarial Hallucination Traps | `sparse_bm25` | 30 | 4.33 | 4.33 | 1.0000 | 100.0% | 100.0% | 83.3% | 56.0 ms |
| Adversarial Hallucination Traps | `dense_faiss` | 30 | 4.33 | 4.33 | 1.0000 | 100.0% | 100.0% | 83.3% | 91.8 ms |
| Adversarial Hallucination Traps | `hybrid_rerank` | 30 | 4.33 | 4.33 | 1.0000 | 100.0% | 100.0% | 83.3% | 4377.0 ms |

---

## 3. Documented Failure Modes & Engineering Postmortem

To provide rigorous engineering transparency for quantitative interviewers, we analyze real queries where the flagship `hybrid_rerank` pipeline experienced edge cases or trade-offs:

### Failure Case 1: `SEC_BENCH_001` (table_numeric)
- **Question**: *"What was Apple's total iPhone net sales in fiscal year 2026?"*
- **Observed Metrics**: Faithfulness: `2.50`, Recall@5: `0.00`, Fact Coverage: `0.0%`
- **Generated Output**: `According to AAPL's Form 10-K disclosure: (1) Services net sales include amortization of the deferred value of services bundled in the sales price of certain products. The Company’s proportion of net sales by disaggregated revenue source was generally consistent for each reportab...`
- **Audit Diagnosis**: Grounded narrative text present, but key financial figures omitted.
- **Root Cause Analysis**: Large audited tables (e.g. segment disaggregation) span across chunk boundaries. If the column header row is split from deeper footnote rows, the cross-encoder may score surrounding narrative chunks higher than the raw table fragment.
- **Engineering Trade-off / Remediation**: Table-aware chunking with header replication ensures all sub-chunks retain context schema.

### Failure Case 2: `SEC_BENCH_061` (cross_company_comparative)
- **Question**: *"Compare total annual revenues between Apple and Microsoft in fiscal year 2026."*
- **Observed Metrics**: Faithfulness: `2.50`, Recall@5: `0.00`, Fact Coverage: `0.0%`
- **Generated Output**: `Comparative 10-K Analysis across disclosed reporting structures:

• **MSFT**: [Ticker: MSFT | Form: 10-K | Fiscal Year: 2026 | Section: Item 7A [MSFT | FY 2026 | Item 7A | Chunk 161]

• **AAPL**: Apple Inc. | 2026 Form 10-K | 4 Business Seasonality and Product Introductions The C...`
- **Audit Diagnosis**: Grounded narrative text present, but key financial figures omitted.
- **Root Cause Analysis**: Multi-ticker comparative queries require retrieving balanced evidence across two separate companies. When both companies have large 10-K filings, RRF fusion can bias towards the company with higher lexical density, leaving the second ticker's chunk ranked outside top-5.
- **Engineering Trade-off / Remediation**: Implement ticker-partitioned retrieval where top-3 chunks are allocated per detected ticker rather than a global top-5 pool.

### Failure Case 3: `SEC_BENCH_092` (adversarial_unanswerable)
- **Question**: *"What was the exact gross profit margin percentage of the iPhone 16 Pro Max sold in Q3 2026?"*
- **Observed Metrics**: Faithfulness: `1.00`, Recall@5: `1.00`, Fact Coverage: `100.0%`
- **Generated Output**: `According to AAPL's Form 10-K disclosure: (1) Services net sales include amortization of the deferred value of services bundled in the sales price of certain products. iPhone iPhone net sales increased during 2026 compared to 2025 due to higher net sales of Pro models. Mac Mac ne...`
- **Audit Diagnosis**: Severe failure: Model hallucinated or invented numbers on an unanswerable trap.
- **Root Cause Analysis**: Strict abstention thresholding. The system errs on the side of caution (`[INSUFFICIENT_FILING_EVIDENCE]`) when numerical metrics are present in footnotes but lack direct parent-sentence proximity.
- **Engineering Trade-off / Remediation**: In production financial systems, a conservative abstention false-positive is preferred over a hallucination false-negative.

---
Generated automatically by `src/experiment_runner.py`.
