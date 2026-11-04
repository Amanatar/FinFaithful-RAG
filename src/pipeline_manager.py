import os
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

try:
    from ticker_lookup import get_cik, get_company_name
except ImportError:  # pragma: no cover
    from .ticker_lookup import get_cik, get_company_name

try:
    from sec_parser import chunk_section, parse_and_save_filing
except ImportError:
    from .sec_parser import chunk_section, parse_and_save_filing

try:
    from download_sec import download_filing_for_ticker
except ImportError:
    from .download_sec import download_filing_for_ticker

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "AmanAtarResearch ataraman9@gmail.com")}
DATA_DIR = Path("data/raw_10k")
LEGACY_RAW_DIR = Path("data/raw/sec")
TEXT_DIR = Path("data/processed/sec_text")
PROCESSED_10K_DIR = Path("data/processed_10k")
OUT_DIR = Path("outputs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_RAW_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_10K_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_filings(ticker: str, limit: int = 1, progress_callback=None) -> int:
    """Downloads 10-K filings for a given ticker with table preservation."""
    ticker = ticker.upper().strip()
    if progress_callback:
        progress_callback(f"📡 Downloading {ticker} 10-K filings from SEC EDGAR...")
    files = download_filing_for_ticker(
        ticker=ticker,
        limit=limit,
        raw_dir=DATA_DIR,
        processed_dir=PROCESSED_10K_DIR,
        auto_parse=True,
    )
    return len(files)


def parse_filings(ticker: str, progress_callback=None) -> int:
    """Parses downloaded HTML filings into structured sections with Markdown tables."""
    ticker = ticker.upper().strip()
    if progress_callback:
        progress_callback(f"📄 Parsing {ticker} HTML filings into structured sections...")

    parsed = 0
    raw_files = list(DATA_DIR.glob(f"{ticker}_*.html"))
    if not raw_files:
        raw_files = list(LEGACY_RAW_DIR.glob(f"{ticker}_*.html"))

    for f in raw_files:
        fdate = f.stem.split("_")[-1] if "_" in f.stem else "2026-11-03"
        parse_and_save_filing(
            raw_html_path=f,
            ticker=ticker,
            filing_date=fdate,
            output_dir=PROCESSED_10K_DIR,
            cik=get_cik(ticker),
        )
        parsed += 1

    return parsed


def chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]


def build_index(ticker: str, progress_callback=None) -> int:
    """Builds FAISS vector index for the ticker's filings."""
    ticker = ticker.upper().strip()
    if progress_callback:
        progress_callback("🧠 Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    import json
    chunks, meta = [], []
    for f in PROCESSED_10K_DIR.glob(f"{ticker}_*_parsed.json"):
        with open(f, "r", encoding="utf-8") as jf:
            parsed_data = json.load(jf)
        for sec_id, sec_data in parsed_data.get("sections", {}).items():
            sec_chunks = chunk_section(
                section_data=sec_data,
                ticker=ticker,
                fiscal_year=parsed_data.get("fiscal_year", 2026),
            )
            for sc in sec_chunks:
                chunks.append(sc["text"])
                meta.append({
                    "source": f"{ticker}_{parsed_data.get('filing_date', '')}.html",
                    "chunk_id": sc["chunk_id"],
                    "section_id": sc["item_section"],
                    "is_table": sc["is_table"],
                })

    if not chunks:
        # Fallback to legacy text
        for f in TEXT_DIR.glob(f"{ticker}_*.txt"):
            text = f.read_text(encoding="utf-8")
            for i, c in enumerate(chunk(text)):
                chunks.append(c)
                meta.append({"source": f.name, "chunk_id": i})

    if not chunks:
        raise ValueError(f"No text chunks found for {ticker}. Run download and parse first.")

    if progress_callback:
        progress_callback(f"🔢 Generating vectors for {len(chunks)} chunks...")

    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = OUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUT_DIR / f"{ticker}_meta.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)

    if progress_callback:
        progress_callback(f"✅ Index built! ({len(chunks)} chunks)")

    return len(chunks)


def run_full_pipeline(ticker: str, progress_callback=None):
    """Runs the complete pipeline: download → parse → index."""
    try:
        downloaded = download_filings(ticker, limit=1, progress_callback=progress_callback)
        parsed = parse_filings(ticker, progress_callback=progress_callback)
        chunks = build_index(ticker, progress_callback=progress_callback)

        return {
            "status": "success",
            "ticker": ticker,
            "filings_downloaded": downloaded,
            "filings_parsed": parsed,
            "chunks_indexed": chunks,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_pipeline(
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    ticker_filter: Optional[list] = None,
) -> dict:
    """
    Executes the 4-tier retrieval and citation generation pipeline.
    Modes:
    - 'direct_llm': Ungrounded Baseline (0 chunks)
    - 'sparse_bm25': BM25 lexical retrieval (top-5)
    - 'dense_faiss': FAISS dense semantic retrieval (top-5)
    - 'hybrid_rerank': Flagship RRF + Cross-Encoder reranking (top-5)
    """
    try:
        from retriever import retrieve
        from generator import generate
    except ImportError:
        from .retriever import retrieve
        from .generator import generate

    ret_res = retrieve(query=query, mode=mode, top_k=top_k, tickers=ticker_filter)
    gen_res = generate(query=query, retrieved_chunks=ret_res["chunks"], mode=mode)

    return {
        "query": query,
        "mode": mode,
        "detected_tickers": ret_res["detected_tickers"],
        "retrieved_chunks": ret_res["chunks"],
        "retrieval_latency_ms": ret_res["retrieval_ms"],
        "rerank_latency_ms": ret_res["rerank_ms"],
        "total_retrieval_ms": ret_res["total_latency_ms"],
        "generation_latency_ms": gen_res["generation_latency_ms"],
        "total_latency_ms": ret_res["total_latency_ms"] + gen_res["generation_latency_ms"],
        "answer": gen_res["answer"],
        "citation_audit": gen_res["citation_audit"],
        "engine": gen_res["engine"],
    }

