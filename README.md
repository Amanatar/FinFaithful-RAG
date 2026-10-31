# FinFaithful-RAG: SEC 10-K Retrieval & Faithfulness Benchmark

<div align="center">

[![CI Verified](https://github.com/Amanatar/FinFaithful-RAG/actions/workflows/ci-verify.yml/badge.svg)](https://github.com/Amanatar/FinFaithful-RAG/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Ask an ungrounded LLM for a specific figure from an SEC 10-K filing -- YoY segment revenue growth, debt maturities, or footnote adjustments -- and it will confidently make numbers up when the filing isn't in context. On our baseline SEC 10-K benchmark, direct LLM prompting scored just **1.67 / 5.0** on faithfulness.

FinFaithful-RAG is an end-to-end retrieval and automated evaluation harness over SEC EDGAR 10-K filings (AAPL, MSFT, etc.) built to benchmark keyword, dense vector (FAISS), and hybrid retrieval against an LLM-as-a-Judge grading pipeline.

---

## Benchmark Results

| Method | Avg Faithfulness | Avg Relevance | Avg Latency |
| :--- | :---: | :---: | :---: |
| **Direct LLM (No Context)** | 1.67 / 5.0 | 2.33 / 5.0 | 4.7s |
| **TF-IDF Keyword RAG** | 3.67 / 5.0 | 3.67 / 5.0 | 11.7s |
| **FAISS Vector RAG** | **5.00 / 5.0** | 4.33 / 5.0 | **5.0s** |
| **Hybrid RAG (Dense + Sparse)** | **5.00 / 5.0** | **4.50 / 5.0** | 8.6s |

### Engineering Takeaways
- **Latency vs. Faithfulness Sweet Spot:** Dense FAISS vector retrieval achieves a perfect 5.00 / 5.0 faithfulness score with only +0.3s latency overhead over raw generation (5.0s vs 4.7s). It completely eliminates ungrounded hallucination without introducing query bottlenecks.
- **Lexical Precision in Finance:** Hybrid RAG edges out pure vector search on Relevance (4.50 vs 4.33) because TF-IDF sparse matching captures exact financial line-items, debt tranches, and specific statutory definitions that semantic embeddings occasionally blur, at the cost of an extra 3.6s retrieval pass.

---

## Architecture & Pipeline

```text
SEC EDGAR API (10-K Filings)
         │
         ▼
Structural Parser (Item 1, Item 7 MD&A, Notes) ──► 1,200-char chunks (150-char overlap)
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
Sentence-Transformers (all-MiniLM-L6-v2)    TF-IDF Sparse Index
         │                                          │
         ▼                                          ▼
   FAISS FlatL2 Dense Index                     BM25/TF-IDF Cosine
         │                                          │
         └───────────────────┬──────────────────────┘
                             ▼
                 Hybrid Retrieval Fusion
                             │
                             ▼
         NVIDIA NIM (meta/llama-3.1-70b-instruct)
                             │
                             ▼
            Context-Grounded Financial Answer
                             │
                             ▼
           LLM-as-a-Judge Evaluation Pipeline
           (Faithfulness & Relevance Scoring)
```

1. **Automated EDGAR Ingestion (`download_sec.py`, `pipeline_manager.py`):** Queries SEC submission endpoints by CIK, downloading official 10-K annual reports with declared user-agent compliance.
2. **Structural Chunking & Preprocessing (`build_index.py`, `parse_filings.py`):** Splits raw filings by statutory headings (`PART`, `Item 1`, `Item 7 MD&A`, `Notes to Consolidated Financial Statements`) into 1,200-character windows with 150-character overlap.
3. **Dense Indexing (`build_index.py`):** Generates 384-dimensional embeddings via `sentence-transformers/all-MiniLM-L6-v2` and persists them in Meta `FAISS` (`IndexFlatL2`) alongside chunk metadata.
4. **Retrieval Comparison Harness (`retrieval_baseline_runner.py`):** Evaluates four retrieval regimes:
   - *Direct LLM:* Zero context baseline to quantify native hallucination rates.
   - *TF-IDF Keyword RAG:* Sparse lexical retrieval scoring chunks via cosine similarity.
   - *FAISS Vector RAG:* Dense Euclidean distance similarity search over embedding space.
   - *Hybrid RAG:* Weighted reciprocal rank fusion combining dense semantic context with sparse lexical matches.
5. **Context-Grounded Inference (`rag_qa.py`):** Conditions `meta/llama-3.1-70b-instruct` via the NVIDIA NIM API on retrieved top-$k$ filing chunks.
6. **Synthetic Benchmark Builder (`benchmark_builder.py`):** Generates target ground-truth financial evaluation pairs directly from disclosure sections.
7. **LLM-as-a-Judge Evaluation (`evaluate.py`):** Automated judge model scoring generated responses against ground truth context for *Faithfulness* (1.0 - 5.0) and *Relevance* (1.0 - 5.0) in structured JSON.
8. **Interactive Explorer (`app.py`):** Streamlit web application with ticker lookup, dynamic on-demand filing ingestion, chunk inspection, and live latency metrics.

---

## Repository Structure

```text
FinFaithful-RAG/
├── src/
│   ├── app.py                      # Interactive Streamlit financial Q&A interface
│   ├── benchmark_builder.py        # Synthetic ground-truth Q&A generator
│   ├── build_index.py              # Sentence-Transformers chunking & FAISS indexing
│   ├── download_sec.py             # SEC EDGAR 10-K downloader with rate limiting
│   ├── evaluate.py                 # LLM-as-a-Judge evaluation harness
│   ├── experiment_runner.py        # End-to-end benchmark execution engine
│   ├── parse_filings.py            # SEC HTML-to-text parser
│   ├── pipeline_manager.py         # End-to-end ingestion and index coordinator
│   ├── rag_qa.py                   # Context-grounded inference with NVIDIA NIM
│   ├── retrieval_baseline_runner.py# Sparse vs dense vs hybrid retrieval benchmark
│   ├── summarize_results.py        # Statistical aggregation and score reporting
│   ├── ticker_lookup.py            # SEC CIK company registry lookup
│   └── validate_system.py          # Environment, key, and dependency health check
├── .env.example                    # Sample configuration
├── .gitignore                      # Strict ignore for data, indices & environments
├── LICENSE                         # MIT License
├── README.md                       # Architecture & benchmark documentation
└── requirements.txt                # Production dependencies
```

---

## Quickstart

### 1. Clone & Set Up Environment

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

Copy the example environment file and add your NVIDIA NIM API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
NVIDIA_API_KEY=nvapi-your_key_here
```

### 3. Run Benchmark Experiments

Run the end-to-end evaluation pipeline comparing Direct LLM, TF-IDF, FAISS, and Hybrid retrieval:

```bash
python src/experiment_runner.py
```

### 4. Launch Interactive Application

Launch the Streamlit interface to query filings interactively:

```bash
streamlit run src/app.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
