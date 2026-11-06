import math
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

QUANTITATIVE_TERMS = {
    "revenue", "revenues", "sales", "net sales", "income", "margin", "margins",
    "eps", "earnings", "cash flow", "debt", "assets", "liabilities", "capex",
    "r&d", "research and development", "interest", "operating expenses",
    "provision", "deposits", "borrowings", "operating income", "net income",
    "operating cash flow", "capital expenditures", "long-term debt"
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


def detect_adversarial_scope(query: str, detected_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Lightweight pre-generation scope check detecting:
    - Future unfiled fiscal years (e.g. 2027, 2028, 2029)
    - Granular quarterly requests (Q1-Q4) not in annual 10-K
    - Confidential supplier pricing or unit-level bill-of-materials (BOM)
    - Speculative unannounced mergers or unfiled products
    """
    q_lower = query.lower()

    if re.search(r"\b(202[7-9]|203[0-9])\b", query):
        return {
            "is_unanswerable_trap": True,
            "must_abstain": True,
            "abstention_reason": "Query requests figures for a future unfiled fiscal year not available in Form 10-K annual reports.",
        }

    if re.search(r"\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b", q_lower) and any(
        w in q_lower for w in ["margin", "churn", "pricing", "discount", "gross margin"]
    ):
        return {
            "is_unanswerable_trap": True,
            "must_abstain": True,
            "abstention_reason": "Granular quarterly SKU-level profit margins and customer churn metrics are confidential trade secrets not published in annual Form 10-K filings.",
        }

    confidential_patterns = [
        r"\b(?:bill[- ]of[- ]materials|bom\b|wafer price|foundry pricing|volume discount granted|negotiated discount)\b",
        r"\b(?:employee attrition rate|voluntary attrition rate)\b",
        r"\b(?:undisclosed purchase price|planned video game studio acquisition)\b",
        r"\b(?:daily active user churn|customer churn rate)\b",
        r"\b(?:aws revenue.*(?:apple|aapl)|azure revenue.*(?:alphabet|googl))\b",
    ]
    for pat in confidential_patterns:
        if re.search(pat, q_lower):
            return {
                "is_unanswerable_trap": True,
                "must_abstain": True,
                "abstention_reason": "Query targets proprietary unit-economics, trade secrets, divisional churn, or non-existent Guidance not published in SEC Form 10-K filings.",
            }

    return {
        "is_unanswerable_trap": False,
        "must_abstain": False,
        "abstention_reason": "",
    }


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

        # 4. Neural Models
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @classmethod
    def get_instance(cls, artifacts_dir: Path = ARTIFACTS_DIR) -> "SECRetrieverBundle":
        if cls._instance is None:
            cls._instance = SECRetrieverBundle(artifacts_dir=artifacts_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Forces reloading the bundle after index rebuild."""
        cls._instance = None


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

    # If multi-ticker comparative query, allocate balanced quota per ticker
    if tickers and len(tickers) >= 2:
        quota = math.ceil(top_k / len(tickers))
        combined_results = []
        for t in tickers:
            t_scores = np.copy(scores)
            mask = (bundle.df_meta["ticker"] == t).values
            t_scores[~mask] = -1e9
            top_t_indices = np.argsort(t_scores)[::-1][:quota]
            for rk, idx in enumerate(top_t_indices, 1):
                if t_scores[idx] <= -1e8:
                    continue
                rec = dict(bundle.chunks_records[idx])
                rec["score"] = float(t_scores[idx])
                rec["retrieval_rank"] = len(combined_results) + 1
                rec["retrieval_method"] = "sparse_bm25"
                combined_results.append(rec)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return combined_results[:top_k], latency_ms

    if tickers and len(tickers) == 1:
        mask = (bundle.df_meta["ticker"] == tickers[0]).values
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

    # If multi-ticker comparative query, allocate balanced quota per ticker
    if tickers and len(tickers) >= 2:
        quota = math.ceil(top_k / len(tickers))
        distances, indices = bundle.faiss_index.search(q_vec, min(len(bundle.chunks_records), 300))
        combined_results = []
        for t in tickers:
            t_count = 0
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(bundle.chunks_records):
                    continue
                rec = dict(bundle.chunks_records[idx])
                if rec["ticker"] != t:
                    continue
                t_count += 1
                rec["score"] = float(dist)
                rec["retrieval_rank"] = len(combined_results) + 1
                rec["retrieval_method"] = "dense_faiss"
                combined_results.append(rec)
                if t_count >= quota:
                    break
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return combined_results[:top_k], latency_ms

    fetch_k = top_k if not tickers else min(len(bundle.chunks_records), top_k * 20)
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


def _retrieve_single_ticker_hybrid(
    query: str,
    bundle: SECRetrieverBundle,
    target_ticker: Optional[str] = None,
    top_k: int = 5,
    bm25_candidates_k: int = 25,
    faiss_candidates_k: int = 25,
    fused_candidates_k: int = 15,
    rrf_k: int = 60,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Retrieves and reranks candidates for a single ticker partition with financial section boosting."""
    t0_ret = time.perf_counter()
    ticker_list = [target_ticker] if target_ticker else None

    bm25_res, _ = retrieve_sparse_bm25(query, bundle, top_k=bm25_candidates_k, tickers=ticker_list)
    faiss_res, _ = retrieve_dense_faiss(query, bundle, top_k=faiss_candidates_k, tickers=ticker_list)

    # Check for quantitative financial query
    q_lower = query.lower()
    is_quant = any(term in q_lower for term in QUANTITATIVE_TERMS)

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

    # Apply Financial Statement and Table Boost
    if is_quant:
        for gid, item in candidate_map.items():
            sec = item.get("item_section", "")
            is_tbl = item.get("is_table", False)
            boost = 1.0
            if sec == "Item 8" and is_tbl:
                boost = 1.40
            elif sec == "Item 7" and is_tbl:
                boost = 1.30
            elif sec == "Item 8":
                boost = 1.15
            elif sec == "Item 1A":
                boost = 0.65
            rrf_scores[gid] *= boost

    fused_gids = sorted(rrf_scores.keys(), key=lambda g: rrf_scores[g], reverse=True)[:fused_candidates_k]
    fused_candidates = [candidate_map[g] for g in fused_gids]
    ret_ms = (time.perf_counter() - t0_ret) * 1000.0

    if not fused_candidates:
        return [], ret_ms, 0.0

    # Cross-Encoder Reranking
    t0_rerank = time.perf_counter()
    pairs = [(query, c["text"]) for c in fused_candidates]
    ce_scores = bundle.cross_encoder.predict(pairs)

    reranked_order = np.argsort(ce_scores)[::-1][:top_k]
    results = []
    for new_rank, idx in enumerate(reranked_order, 1):
        rec = dict(fused_candidates[idx])
        rec["cross_encoder_score"] = float(ce_scores[idx])
        rec["rrf_score"] = float(rrf_scores[rec["global_chunk_id"]])
        rec["score"] = float(ce_scores[idx])
        rec["retrieval_rank"] = new_rank
        rec["retrieval_method"] = "hybrid_rerank"
        results.append(rec)

    rerank_ms = (time.perf_counter() - t0_rerank) * 1000.0
    return results, ret_ms, rerank_ms


def retrieve_hybrid_rerank(
    query: str,
    bundle: SECRetrieverBundle,
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Mode 4 (Flagship):
    - Balanced Per-Ticker Partitioning: If M >= 2 tickers, retrieves ceil(top_k / M) per ticker independently.
    - If 1 ticker, filters strictly to that ticker.
    - Applies Financial Section Boost and Cross-Encoder Reranking.
    """
    if tickers and len(tickers) >= 2:
        quota = math.ceil(top_k / len(tickers))
        combined_chunks = []
        total_ret_ms = 0.0
        total_rerank_ms = 0.0

        for t in tickers:
            t_res, r_ms, rk_ms = _retrieve_single_ticker_hybrid(
                query=query,
                bundle=bundle,
                target_ticker=t,
                top_k=quota,
            )
            total_ret_ms += r_ms
            total_rerank_ms += rk_ms
            combined_chunks.extend(t_res)

        for i, c in enumerate(combined_chunks, 1):
            c["retrieval_rank"] = i

        return combined_chunks[: max(top_k, len(tickers) * quota)], total_ret_ms, total_rerank_ms

    target_t = tickers[0] if (tickers and len(tickers) == 1) else None
    return _retrieve_single_ticker_hybrid(
        query=query,
        bundle=bundle,
        target_ticker=target_t,
        top_k=top_k,
    )


def retrieve(
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    tickers: Optional[List[str]] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """
    Unified 4-Tier Retrieval Interface with Ticker Routing and Adversarial Guardrail.
    """
    mode = mode.lower().strip()
    if mode == "tfidf":
        mode = "sparse_bm25"

    detected_tickers = tickers if tickers is not None else detect_tickers(query)
    scope = detect_adversarial_scope(query, detected_tickers)

    if mode == "direct_llm":
        return {
            "mode": "direct_llm",
            "query": query,
            "detected_tickers": detected_tickers,
            "scope_check": scope,
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
            "scope_check": scope,
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
            "scope_check": scope,
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
            "scope_check": scope,
            "chunks": chunks,
            "retrieval_ms": ret_ms,
            "rerank_ms": rerank_ms,
            "total_latency_ms": ret_ms + rerank_ms,
        }

    else:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from: direct_llm, sparse_bm25, dense_faiss, hybrid_rerank")
