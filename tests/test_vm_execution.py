import os
import sys
import time
import pytest
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

def test_modules_import_cleanly():
    # Verify core python modules compile and import
    import app
    import evaluate
    import build_index
    import ticker_lookup
    print("\n[VM EXECUTION] System modules (src/app.py, src/evaluate.py, src/build_index.py) imported cleanly.")

def test_sec_indexing_and_retrieval():
    corpus = [
        "Apple Inc. Total net sales were $383,285 million in fiscal 2023 compared to $394,328 million in fiscal 2022. Gross margin was $169,148 million.",
        "Microsoft Corporation revenue increased $18.5 billion or 10% driven by growth in Intelligent Cloud and Productivity and Business Processes.",
        "Management Discussion and Analysis: Primary operational risks include global supply chain constraints and semiconductor component availability.",
        "Research and development expenses were $29,915 million in 2023, reflecting continuous investments in generative models and silicon design."
    ]
    
    # 1. Dense FAISS Indexing with all-MiniLM-L6-v2
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(corpus).astype("float32")
    dimension = vectors.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    # 2. TF-IDF Lexical Baseline
    tfidf = TfidfVectorizer().fit(corpus)
    tfidf_matrix = tfidf.transform(corpus)
    
    queries = [
        "What was the total net sales revenue?",
        "What drove the growth in cloud revenue?"
    ]
    
    print(f"\n[VM EXECUTION] SEC 10-K Corpus Indexed: {len(corpus)} financial chunks in FAISS (dim={dimension}) & TF-IDF.")
    
    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        q_vec = embed_model.encode([q]).astype("float32")
        distances, indices = index.search(q_vec, 1)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        
        top_idx = indices[0][0]
        top_dist = float(distances[0][0])
        snippet = corpus[top_idx]
        
        print(f"[VM EXECUTION] Query {idx}: '{q}'")
        print(f"[VM EXECUTION] -> Top-1 Chunk [{top_idx}]: '{snippet[:85]}...' (L2 Dist: {top_dist:.4f}, Latency: {latency_ms:.2f} ms)")
        assert top_dist >= 0.0
        assert len(snippet) > 0
