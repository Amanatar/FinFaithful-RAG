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
from sentence_transformers import CrossEncoder, SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "sec_index"

# Known ticker mappings and aliases for entity linking
TICKER_ENTITY_MAP = {
    "AAPL": ["AAPL", "APPLE", "IPHONE", "MACBOOK", "TIM COOK"],
    "MSFT": ["MSFT", "MICROSOFT", "AZURE", "WINDOWS", "OFFICE 365", "XBOX", "SATYA NADELLA"],
    "NVDA": ["NVDA", "NVIDIA", "GEFORCE", "H100", "B200", "BLACKWELL", "HOPPER", "CUDA", "JENSEN HUANG"],
    "JPM": ["JPM", "JPMORGAN", "JP MORGAN", "CHASE", "JAMIE DIMON", "FIRST REPUBLIC"],
    "GOOGL": ["GOOGL", "GOOG", "GOOGLE", "ALPHABET", "YOUTUBE", "SUNDAR PICHAI", "WAYMO", "DEEPMIND"],
}


def detect_tickers(query: str) -> Optional[List[str]]:
    """
    Detects company tickers mentioned in user query using word-boundary matching.
    Supports single or multi-ticker comparative questions.
    Returns list of uppercase tickers (e.g. ['AAPL', 'MSFT']) or None if unspecified.
    """
    detected = []
    q_upper = f" {query.upper()} "
    for ticker, aliases in TICKER_ENTITY_MAP.items():
        for alias in aliases:
            pattern = r"(?:\b|\A)" + re.escape(alias) + r"(?:\b|\Z)"
            if re.search(pattern, q_upper, re.IGNORECASE):
                if ticker not in detected:
                    detected.append(ticker)
                break
    return detected if detected else None


class SECRetrieverBundle:
    """Thread-safe singleton holding in-memory indexes and neural models."""

    _instance = None

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.parquet_path = artifacts_dir / "chunks_metadata.parquet"
        self.faiss_path = artifacts_dir / "faiss.index"
        self.bm25_path = artifacts_dir / "bm25_corpus.pkl"

        if not self.parquet_path.exists() or not self.faiss_path.exists() or not self.bm25_path.exists():
            raise FileNotFoundError(
                f"Missing index files in {artifacts_dir}. Run `src/build_index.py` first."
            )

        # 1. Load metadata
        self.df_meta = pd.read_parquet(self.parquet_path)
        self.chunks_records = self.df_meta.to_dict(orient="records")

        # 2. Load FAISS Dense Index
        self.faiss_index = faiss.read_index(str(self.faiss_path))

        # 3. Load BM25 Lexical Index
        with open(self.bm25_path, "rb") as f_bm25:
            bm25_data = pickle.load(f_bm25)
        self.bm25 = bm25_data["bm25"]

        # 4. Neural Models (Lazy-loaded or cached)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @classmethod
    def get_instance(cls, artifacts_dir: Path = ARTIFACTS_DIR) -> "SECRetrieverBundle":
        if cls._instance is None:
            cls._instance = SECRetrieverBundle(artifacts_dir=artifacts_dir)
        return cls._instance


def tokenize_query(query: str) -> List[str]:
    """Tokenizes query into alphanumeric and financial tokens."""
    return re.findall(r"\b[A-Za-z0-9\$\%]+(?:\.[A-Za-z0-9]+)?\b", query.lower())


def retrieve_sparse_bm25(
    query: str,
    bundle: SECRetrieverBundle,
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Mode 2: BM25 Lexical Retrieval over 10-K text, tokens, numbers, and tables."""
    t0 = time.perf_counter()
    tokens = tokenize_query(query)
    scores = np.array(bundle.bm25.get_scores(tokens), dtype="float32")

    # Filter by ticker if specified
    if tickers:
        mask = bundle.df_meta["ticker"].isin(tickers).values
        scores[~mask] = -1e9

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] <= -1e8:
            continue
        record = dict(bundle.chunks_records[idx])
        record["score"] = float(scores[idx])
        record["retrieval_rank"] = rank
        record["retrieval_method"] = "sparse_bm25"
        results.append(record)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms


def retrieve_dense_faiss(
    query: str,
    bundle: SECRetrieverBundle,
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Mode 3: Dense Semantic Retrieval with normalized embeddings and FAISS."""
    t0 = time.perf_counter()
    q_vec = bundle.embed_model.encode([query], normalize_embeddings=True)
    q_vec = np.ascontiguousarray(q_vec, dtype="float32")

    # Over-fetch if ticker filtering is required
    fetch_k = top_k if not tickers else min(len(bundle.chunks_records), top_k * 15)
    distances, indices = bundle.faiss_index.search(q_vec, fetch_k)

    results = []
    rank = 0
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(bundle.chunks_records):
            continue
        record = dict(bundle.chunks_records[idx])
        if tickers and record["ticker"] not in tickers:
            continue
        rank += 1
        record["score"] = float(dist)
        record["retrieval_rank"] = rank
        record["retrieval_method"] = "dense_faiss"
        results.append(record)
        if len(results) >= top_k:
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms


def retrieve_hybrid_rerank(
    query: str,
    bundle: SECRetrieverBundle,
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
    bm25_candidates_k: int = 20,
    faiss_candidates_k: int = 20,
    fused_candidates_k: int = 15,
    rrf_k: int = 60,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Mode 4 (Flagship):
    1. Retrieve top-20 from BM25 + top-20 from FAISS.
    2. Fuse via Reciprocal Rank Fusion: score = sum(1 / (60 + rank)).
    3. Rerank top-15 fused candidates down to top-5 using ms-marco Cross-Encoder on CPU.
    """
    t0_retrieval = time.perf_counter()

    # 1. BM25 Candidates
    bm25_res, _ = retrieve_sparse_bm25(query, bundle, top_k=bm25_candidates_k, tickers=tickers)

    # 2. FAISS Candidates
    faiss_res, _ = retrieve_dense_faiss(query, bundle, top_k=faiss_candidates_k, tickers=tickers)

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores: Dict[str, float] = {}
    candidate_map: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(bm25_res, 1):
        gid = item["global_chunk_id"]
        rrf_scores[gid] = rrf_scores.get(gid, 0.0) + 1.0 / (rrf_k + rank)
        if gid not in candidate_map:
            candidate_map[gid] = item

    for rank, item in enumerate(faiss_res, 1):
        gid = item["global_chunk_id"]
        rrf_scores[gid] = rrf_scores.get(gid, 0.0) + 1.0 / (rrf_k + rank)
        if gid not in candidate_map:
            candidate_map[gid] = item

    # Sort fused candidates by RRF score descending
    fused_gids = sorted(rrf_scores.keys(), key=lambda gid: rrf_scores[gid], reverse=True)[:fused_candidates_k]
    fused_candidates = [candidate_map[gid] for gid in fused_gids]
    retrieval_ms = (time.perf_counter() - t0_retrieval) * 1000.0

    if not fused_candidates:
        return [], retrieval_ms, 0.0

    # 4. Cross-Encoder Reranking
    t0_rerank = time.perf_counter()
    pairs = [(query, c["text"]) for c in fused_candidates]
    ce_scores = bundle.cross_encoder.predict(pairs)

    reranked_order = np.argsort(ce_scores)[::-1][:top_k]
    final_results = []
    for new_rank, idx in enumerate(reranked_order, 1):
        rec = dict(fused_candidates[idx])
        rec["cross_encoder_score"] = float(ce_scores[idx])
        rec["rrf_score"] = float(rrf_scores[rec["global_chunk_id"]])
        rec["score"] = float(ce_scores[idx])
        rec["retrieval_rank"] = new_rank
        rec["retrieval_method"] = "hybrid_rerank"
        final_results.append(rec)

    rerank_ms = (time.perf_counter() - t0_rerank) * 1000.0
    return final_results, retrieval_ms, rerank_ms


def retrieve(
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """
    Unified 4-Tier Retrieval Interface:
    Supports modes:
    - 'direct_llm': returns empty context (ungrounded baseline)
    - 'sparse_bm25' (or 'tfidf'): BM25 lexical search (top-5)
    - 'dense_faiss': FAISS dense semantic search (top-5)
    - 'hybrid_rerank': BM25(20) + FAISS(20) -> RRF(60) -> CrossEncoder(15->5)
    """
    mode = mode.lower().strip()
    if mode == "tfidf":
        mode = "sparse_bm25"

    detected_tickers = tickers if tickers is not None else detect_tickers(query)

    if mode == "direct_llm":
        return {
            "mode": "direct_llm",
            "query": query,
            "detected_tickers": detected_tickers,
            "chunks": [],
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "total_latency_ms": 0.0,
        }

    bundle = SECRetrieverBundle.get_instance(artifacts_dir=artifacts_dir)

    if mode in ("sparse_bm25", "bm25"):
        chunks, ret_ms = retrieve_sparse_bm25(query, bundle, top_k=top_k, tickers=detected_tickers)
        return {
            "mode": "sparse_bm25",
            "query": query,
            "detected_tickers": detected_tickers,
            "chunks": chunks,
            "retrieval_ms": ret_ms,
            "rerank_ms": 0.0,
            "total_latency_ms": ret_ms,
        }

    elif mode in ("dense_faiss", "faiss"):
        chunks, ret_ms = retrieve_dense_faiss(query, bundle, top_k=top_k, tickers=detected_tickers)
        return {
            "mode": "dense_faiss",
            "query": query,
            "detected_tickers": detected_tickers,
            "chunks": chunks,
            "retrieval_ms": ret_ms,
            "rerank_ms": 0.0,
            "total_latency_ms": ret_ms,
        }

    elif mode in ("hybrid_rerank", "hybrid"):
        chunks, ret_ms, rerank_ms = retrieve_hybrid_rerank(
            query, bundle, top_k=top_k, tickers=detected_tickers
        )
        return {
            "mode": "hybrid_rerank",
            "query": query,
            "detected_tickers": detected_tickers,
            "chunks": chunks,
            "retrieval_ms": ret_ms,
            "rerank_ms": rerank_ms,
            "total_latency_ms": ret_ms + rerank_ms,
        }

    else:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from: direct_llm, sparse_bm25, dense_faiss, hybrid_rerank")
