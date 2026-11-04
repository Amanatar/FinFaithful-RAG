import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline_manager import query_pipeline

TEST_QUERIES = [
    {
        "id": "Q1_EXACT_TABLE_LOOKUP",
        "description": "Exact Table Lookup on NVDA (Data Center Revenue FY 2026)",
        "query": "What was NVIDIA's Data Center revenue in FY 2026?",
    },
    {
        "id": "Q2_CROSS_COMPANY_COMPARISON",
        "description": "Cross-Company Comparison: AAPL vs MSFT (Cloud & Services Revenue)",
        "query": "Compare Apple and Microsoft cloud and services revenue in fiscal 2026.",
    },
    {
        "id": "Q3_UNANSWERABLE_TRAP",
        "description": "Unanswerable Trap: AAPL FY2028 Revenue (Out of Scope / Non-Existent Period)",
        "query": "What is Apple's projected revenue for fiscal year 2028?",
    },
]

MODES = ["direct_llm", "sparse_bm25", "dense_faiss", "hybrid_rerank"]


def run_smoke_tests():
    print("=" * 80)
    print("🚀 FINFAITHFUL-RAG: 4-TIER RETRIEVAL & CITATION ENGINE SMOKE TESTS")
    print("=" * 80)

    all_results = []

    for q_meta in TEST_QUERIES:
        qid = q_meta["id"]
        qdesc = q_meta["description"]
        query = q_meta["query"]

        print(f"\n" + "-" * 80)
        print(f"📌 [{qid}] {qdesc}")
        print(f"Query: \"{query}\"")
        print("-" * 80)

        for mode in MODES:
            t0 = time.perf_counter()
            res = query_pipeline(query, mode=mode, top_k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            cids = [c["global_chunk_id"] for c in res["retrieved_chunks"]]
            table_cids = [c["global_chunk_id"] for c in res["retrieved_chunks"] if c.get("is_table")]
            audit = res["citation_audit"]

            print(f"\n  ▶ MODE: {mode.upper()}")
            print(f"    • Detected Tickers : {res['detected_tickers']}")
            print(f"    • Retrieved Chunks : {len(cids)} {cids}")
            print(f"    • Table Chunks     : {len(table_cids)} {table_cids}")
            print(f"    • Retrieval Latency: {res['retrieval_latency_ms']:.2f} ms")
            if res["rerank_latency_ms"] > 0:
                print(f"    • Rerank Latency   : {res['rerank_latency_ms']:.2f} ms")
            print(f"    • Total Latency    : {res['total_latency_ms']:.2f} ms")
            print(f"    • Citation Audit   : is_valid={audit['is_valid']}, valid={len(audit['valid_citations'])}, phantom={len(audit['phantom_citations'])}, abstention={audit['has_abstention']}")
            print(f"    • Answer Generated :")
            # Indent answer lines
            for line in res["answer"].strip().splitlines():
                print(f"        {line}")

            all_results.append({
                "query_id": qid,
                "description": qdesc,
                "query": query,
                "mode": mode,
                "detected_tickers": res["detected_tickers"],
                "retrieved_chunk_ids": cids,
                "table_chunk_ids": table_cids,
                "retrieval_latency_ms": round(res["retrieval_latency_ms"], 2),
                "rerank_latency_ms": round(res["rerank_latency_ms"], 2),
                "total_latency_ms": round(res["total_latency_ms"], 2),
                "answer": res["answer"],
                "citation_audit": audit,
                "engine": res["engine"],
            })

    # Save structured smoke test results to outputs
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ All 12 test runs completed successfully! Results written to {out_path}")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    run_smoke_tests()
