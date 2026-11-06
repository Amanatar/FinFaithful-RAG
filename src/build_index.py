import argparse
import gzip
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


try:
    from sec_parser import chunk_section, parse_and_save_filing
except ImportError:
    from .sec_parser import chunk_section, parse_and_save_filing

try:
    from ticker_lookup import get_cik
except ImportError:
    from .ticker_lookup import get_cik

PROCESSED_10K_DIR = REPO_ROOT / "data" / "processed_10k"
RAW_10K_DIR = REPO_ROOT / "data" / "raw_10k"
LEGACY_TEXT_DIR = REPO_ROOT / "data" / "processed" / "sec_text"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "sec_index"
OUTPUT_DIR = REPO_ROOT / "outputs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reparse_raw_filings(raw_dir: Path = RAW_10K_DIR, processed_dir: Path = PROCESSED_10K_DIR) -> int:
    """Parses all raw 10-K HTML files into structured JSON with updated table contextualization."""
    raw_files = sorted(raw_dir.glob("*.html"))
    if not raw_files:
        print(f"⚠️ No raw HTML files found in {raw_dir}")
        return 0

    print(f"🔄 Re-parsing {len(raw_files)} raw HTML filings with upgraded table contextualization...")
    count = 0
    for f in raw_files:
        ticker = f.stem.split("_")[0]
        filing_date = f.stem.split("_")[1] if "_" in f.stem else "2026-11-03"
        cik = get_cik(ticker)
        print(f"   Parsing {ticker} ({filing_date})...")
        parse_and_save_filing(
            raw_html_path=f,
            ticker=ticker,
            filing_date=filing_date,
            output_dir=processed_dir,
            cik=cik,
        )
        count += 1
    print(f"✅ Successfully re-parsed {count} filings into {processed_dir}")
    return count


def tokenize_for_bm25(text: str) -> List[str]:
    """Fast financial-aware tokenizer for BM25 indexing."""
    tokens = re.findall(r"\b[A-Za-z0-9\$\%]+(?:\.[A-Za-z0-9]+)?\b", text.lower())
    return tokens


def load_chunks_from_processed(processed_dir: Path) -> List[Dict[str, Any]]:
    """Loads all parsed filings and generates table-preserved chunks with provenance headers."""
    json_files = sorted(processed_dir.glob("*_parsed.json"))
    all_chunks: List[Dict[str, Any]] = []

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            ticker = data.get("ticker", f.stem.split("_")[0])
            fy = data.get("fiscal_year", 2026)
            form = data.get("form", "10-K")
            filing_date = data.get("filing_date", "")
            cik = data.get("cik", "")
            sections = data.get("sections", {})

            global_chunk_idx = 0
            for sec_id, sec_data in sections.items():
                sec_chunks = chunk_section(
                    section_data=sec_data,
                    ticker=ticker,
                    fiscal_year=fy,
                    form=form,
                    max_chunk_size=1400,
                    chunk_overlap=150,
                )
                for sc in sec_chunks:
                    global_chunk_idx += 1
                    sc["global_chunk_id"] = f"{ticker}_{fy}_{global_chunk_idx}"
                    sc["source_file"] = f.name
                    sc["filing_date"] = filing_date
                    sc["cik"] = cik
                    all_chunks.append(sc)
        except Exception as e:
            print(f"⚠️ Error loading parsed filing {f.name}: {e}")

    return all_chunks


def load_legacy_text_chunks(text_dir: Path) -> List[Dict[str, Any]]:
    """Fallback loader for legacy plain text files if processed_10k is absent."""
    chunks: List[Dict[str, Any]] = []
    chunk_size = 1200
    overlap = 150
    step = max(chunk_size - overlap, 1)

    for f in sorted(text_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        ticker = f.stem.split("_")[0]
        for i in range(0, len(text), step):
            ch = text[i : i + chunk_size]
            cid = i // step + 1
            prov = f"[Ticker: {ticker} | Form: 10-K | Fiscal Year: 2026 | Section: Full | Chunk: {cid}]"
            full_txt = f"{prov}\n\n{ch}"
            chunks.append({
                "global_chunk_id": f"{ticker}_2026_{cid}",
                "chunk_id": cid,
                "ticker": ticker,
                "fiscal_year": 2026,
                "form": "10-K",
                "item_section": "Item",
                "item_title": "10-K Text",
                "is_table": False,
                "table_count": 0,
                "char_count": len(full_txt),
                "token_count_est": len(full_txt) // 4,
                "provenance": prov,
                "text": full_txt,
                "source_file": f.name,
                "filing_date": "",
                "cik": "",
            })
    return chunks


def build(
    data_dir: Path = PROCESSED_10K_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    output_dir: Path = OUTPUT_DIR,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Builds and persists a compact, portable corpus & index bundle under artifacts_dir:
    1. faiss.index (normalized L2/IP dense index)
    2. bm25_corpus.pkl (BM25Okapi lexical index + tokenized corpus)
    3. chunks_metadata.parquet (structured columnar metadata)
    4. chunks_metadata.jsonl.gz (compressed portable stream)
    Also saves legacy outputs/faiss.index and outputs/meta.pkl for backwards compatibility.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Loading parsed SEC 10-K corpus chunks...")
    chunks_records = load_chunks_from_processed(data_dir)

    if not chunks_records and LEGACY_TEXT_DIR.exists():
        print("  Notice: No files in processed_10k; checking legacy processed text dir...")
        chunks_records = load_legacy_text_chunks(LEGACY_TEXT_DIR)

    if not chunks_records:
        raise RuntimeError(
            f"No chunks found in {data_dir} or {LEGACY_TEXT_DIR}. Run download_sec.py first to fetch filings."
        )

    total_chunks = len(chunks_records)
    print(f"📊 Total valid chunks assembled: {total_chunks:,}")

    # Extract chunk strings and metadata objects
    chunk_strings = [c["text"] for c in chunks_records]
    metadata_list = [
        {k: v for k, v in c.items() if k != "text"} for c in chunks_records
    ]

    # 1. Dense FAISS Embedding
    print(f"🧠 Loading embedding model '{embed_model_name}'...")
    embed_model = SentenceTransformer(embed_model_name)

    print(f"🔢 Generating dense embeddings for {total_chunks:,} chunks...")
    t0_embed = time.perf_counter()
    embeddings = embed_model.encode(
        chunk_strings,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    embed_time = time.perf_counter() - t0_embed
    print(f"  ↳ Dense embeddings generated in {embed_time:.2f}s (dim: {embeddings.shape[1]}).")

    # Build FAISS Index (IndexFlatIP with normalized vectors computes exact Cosine Similarity)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    faiss_path = artifacts_dir / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))

    # Also write legacy index for backwards compatibility
    legacy_faiss_path = output_dir / "faiss.index"
    faiss.write_index(faiss_index, str(legacy_faiss_path))

    # 2. Lexical BM25 Indexing
    print("📖 Fitting BM25 lexical index on corpus...")
    t0_bm25 = time.perf_counter()
    tokenized_corpus = [tokenize_for_bm25(text) for text in chunk_strings]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_time = time.perf_counter() - t0_bm25
    print(f"  ↳ BM25 index built in {bm25_time:.2f}s.")

    bm25_bundle = {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "chunk_ids": [c["global_chunk_id"] for c in chunks_records],
    }
    bm25_path = artifacts_dir / "bm25_corpus.pkl"
    with open(bm25_path, "wb") as f_bm25:
        pickle.dump(bm25_bundle, f_bm25, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Structured Metadata Persistence (Parquet + Compressed JSONL)
    print("💾 Persisting structured corpus metadata...")
    df_meta = pd.DataFrame(chunks_records)
    parquet_path = artifacts_dir / "chunks_metadata.parquet"
    df_meta.to_parquet(parquet_path, index=False, compression="snappy")

    jsonl_gz_path = artifacts_dir / "chunks_metadata.jsonl.gz"
    with gzip.open(jsonl_gz_path, "wt", encoding="utf-8") as gz:
        for rec in chunks_records:
            gz.write(json.dumps(rec) + "\n")

    # Legacy meta.pkl for backward compatibility
    legacy_meta_path = output_dir / "meta.pkl"
    legacy_meta_records = [
        {"source": c.get("source_file", ""), "chunk_id": c.get("chunk_id", 0), "section_title": c.get("item_title", "")}
        for c in chunks_records
    ]
    with open(legacy_meta_path, "wb") as f_meta:
        pickle.dump({"chunks": chunk_strings, "meta": legacy_meta_records}, f_meta, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate disk sizes
    faiss_size_mb = faiss_path.stat().st_size / (1024 * 1024)
    bm25_size_mb = bm25_path.stat().st_size / (1024 * 1024)
    parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    jsonl_gz_size_mb = jsonl_gz_path.stat().st_size / (1024 * 1024)
    total_bundle_size_mb = faiss_size_mb + bm25_size_mb + parquet_size_mb

    # Diagnostic breakdowns
    ticker_counts = df_meta["ticker"].value_counts().to_dict()
    table_counts = df_meta[df_meta["is_table"] == True]["ticker"].value_counts().to_dict()
    narrative_counts = df_meta[df_meta["is_table"] == False]["ticker"].value_counts().to_dict()

    diagnostic_summary = {
        "total_chunks": total_chunks,
        "ticker_chunk_counts": ticker_counts,
        "ticker_table_counts": table_counts,
        "ticker_narrative_counts": narrative_counts,
        "faiss_size_mb": round(faiss_size_mb, 2),
        "bm25_size_mb": round(bm25_size_mb, 2),
        "parquet_size_mb": round(parquet_size_mb, 2),
        "jsonl_gz_size_mb": round(jsonl_gz_size_mb, 2),
        "total_bundle_size_mb": round(total_bundle_size_mb, 2),
        "sample_table_chunk": next((c for c in chunks_records if c["is_table"]), None),
        "sample_narrative_chunk": next((c for c in chunks_records if not c["is_table"] and c.get("item_section") == "Item 8"), None),
    }

    print("\n" + "=" * 65)
    print("🎉 SEC 10-K CORPUS & MULTI-MODAL INDEX BUNDLE SUCCESSFULLY BUILT!")
    print("=" * 65)
    print(f"📦 Total Portable Index Bundle Size: {total_bundle_size_mb:.2f} MB (Budget Limit: < 35 MB)")
    print(f"  • FAISS Dense Index:       {faiss_size_mb:.2f} MB ({faiss_path})")
    print(f"  • BM25 Lexical Index:      {bm25_size_mb:.2f} MB ({bm25_path})")
    print(f"  • Metadata Parquet:        {parquet_size_mb:.2f} MB ({parquet_path})")
    print(f"  • Compressed JSONL:        {jsonl_gz_size_mb:.2f} MB ({jsonl_gz_path})")
    print("=" * 65)

    return diagnostic_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 index from processed SEC 10-K chunks.")
    parser.add_argument("--data-dir", default=str(PROCESSED_10K_DIR), help="Path to processed 10-K JSON directory")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Path to write index bundle")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Path to write legacy outputs")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--reparse", action="store_true", help="Re-parse raw HTML filings before building index")
    args = parser.parse_args()

    if args.reparse:
        reparse_raw_filings(RAW_10K_DIR, Path(args.data_dir))

    build(
        data_dir=Path(args.data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        output_dir=Path(args.output_dir),
        embed_model_name=args.model,
    )
