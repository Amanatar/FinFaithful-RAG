import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    from pipeline_manager import run_full_pipeline
except ImportError:  # pragma: no cover - fallback for package-style imports
    from .pipeline_manager import run_full_pipeline


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "finfaithful_bench.json"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_benchmark(path: Path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_company_texts(ticker: str):
    text_dir = REPO_ROOT / "data" / "processed" / "sec_text"
    texts = []
    for f in sorted(text_dir.glob(f"{ticker}_*.txt")):
        texts.append((f.name, f.read_text(encoding="utf-8")))
    return texts


def section_chunk(text: str, size: int = 1200, overlap: int = 150):
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]


def split_sections(text: str):
    import re

    parts = re.split(r"\n(?=Note\s+\d|Item\s+\d|PART|Management's Discussion and Analysis|Business)", text)
    return [p.strip() for p in parts if p.strip()]


def build_docs_for_ticker(ticker: str):
    docs = []
    for source_name, text in load_company_texts(ticker):
        for section_id, section in enumerate(split_sections(text)):
            for chunk_id, chunk in enumerate(section_chunk(section)):
                docs.append(
                    {
                        "source": source_name,
                        "section_id": section_id,
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "section_title": section.splitlines()[0][:120] if section.splitlines() else "",
                    }
                )
    return docs


def ensure_pipeline(ticker: str):
    index_path = OUTPUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUTPUT_DIR / f"{ticker}_meta.pkl"
    if index_path.exists() and meta_path.exists():
        return
    run_full_pipeline(ticker, progress_callback=lambda msg: print(f"[{ticker}] {msg}"))


def load_faiss_data(ticker: str):
    index_path = OUTPUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUTPUT_DIR / f"{ticker}_meta.pkl"
    if not index_path.exists() or not meta_path.exists():
        return None, None
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    return index, meta_data


def retrieve_faiss(ticker: str, query: str, k: int = 4):
    index, meta_data = load_faiss_data(ticker)
    if index is None:
        return []

    q_vec = EMBED_MODEL.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, k * 2)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(meta_data["chunks"]):
            continue
        results.append(
            {
                "text": meta_data["chunks"][idx],
                "src": meta_data["meta"][idx]["source"],
                "score": float(distances[0][i]),
            }
        )

    seen = set()
    filtered = []
    for item in results:
        key = (item["src"], item["text"][:250])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
        if len(filtered) >= k:
            break
    return filtered


def retrieve_tfidf(ticker: str, query: str, k: int = 4):
    docs = build_docs_for_ticker(ticker)
    if not docs:
        return []

    corpus = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(matrix, q_vec).flatten()
    top_indices = scores.argsort()[::-1][:k]

    return [
        {
            "text": docs[idx]["text"],
            "src": docs[idx]["source"],
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def retrieve_hybrid(ticker: str, query: str, k: int = 4):
    docs = build_docs_for_ticker(ticker)
    if not docs:
        return []

    corpus = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    q_tfidf = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(tfidf_matrix, q_tfidf).flatten()

    dense_query = EMBED_MODEL.encode([query]).astype("float32")
    dense_index = faiss.IndexFlatL2(EMBED_MODEL.get_sentence_embedding_dimension())
    dense_index.add(EMBED_MODEL.encode(corpus).astype("float32"))
    dense_distances, dense_indices = dense_index.search(dense_query, len(corpus))
    dense_scores = np.zeros(len(corpus), dtype="float32")
    for rank, idx in enumerate(dense_indices[0]):
        dense_scores[idx] = 1.0 / (1.0 + float(dense_distances[0][rank]))

    combined = 0.5 * tfidf_scores + 0.5 * dense_scores
    top_indices = combined.argsort()[::-1][:k]
    return [
        {
            "text": docs[idx]["text"],
            "src": docs[idx]["source"],
            "score": float(combined[idx]),
        }
        for idx in top_indices
    ]


def ask_with_context(ticker: str, question: str, context):
    context_str = "\n\n".join([f"Source: {c['src']}\n{c['text']}" for c in context])
    prompt = f"""You are a strict financial auditor AI analyzing SEC filings for {ticker}.
Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say exactly: "Information not found in the provided filings."
3. Do not invent numbers or hallucinate.
4. Cite the source file names in your answer.

Context:
{context_str}

Question: {question}

Answer:"""

    completion = CLIENT.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a highly accurate financial AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=0.7,
        max_tokens=1024,
        stream=False,
    )
    return completion.choices[0].message.content


def ask_direct(question: str):
    completion = CLIENT.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a highly accurate financial AI assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        top_p=0.7,
        max_tokens=1024,
        stream=False,
    )
    return completion.choices[0].message.content


def judge_answer(question, answer, context):
    context_str = "\n".join([c["text"] for c in context])
    judge_prompt = f"""You are an expert AI evaluator. Grade the AI's answer based on the context.

Context:
{context_str}

Question: {question}
AI Answer: {answer}

Score the answer from 1 to 5 on:
1. Faithfulness (Does it only use the context?)
2. Relevance (Does it answer the question?)

Respond ONLY with a JSON object like this:
{{"faithfulness_score": X, "relevance_score": Y, "reason": "brief reason"}}
"""
    try:
        completion = CLIENT.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
    except Exception:
        return '{"faithfulness_score": 0, "relevance_score": 0, "reason": "Judge failed"}'


def parse_args():
    parser = argparse.ArgumentParser(description="Run FinFaithful baseline experiments.")
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARK_PATH),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["direct", "tfidf", "faiss", "hybrid"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of benchmark questions to evaluate.",
    )
    parser.add_argument(
        "--ensure-pipeline",
        action="store_true",
        help="Run the download/parse/index pipeline if a company index is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = load_benchmark(Path(args.benchmark))
    if args.limit > 0:
        benchmark = benchmark[: args.limit]

    retrieval_methods = {
        "direct": None,
        "tfidf": retrieve_tfidf,
        "faiss": retrieve_faiss,
        "hybrid": retrieve_hybrid,
    }

    results = []
    for item in benchmark:
        ticker = item["ticker"]
        if args.ensure_pipeline:
            ensure_pipeline(ticker)

        for method in args.methods:
            if method not in retrieval_methods:
                raise ValueError(f"Unknown method: {method}")

            print(f"\n🔎 {item['id']} | {method}: {item['question']}")
            start_time = time.time()

            if method == "direct":
                answer = ask_direct(item["question"])
                context = []
                retrieved_sources = ""
            else:
                context = retrieval_methods[method](ticker, item["question"])
                answer = ask_with_context(ticker, item["question"], context)
                retrieved_sources = ", ".join([c["src"] for c in context])

            latency = round(time.time() - start_time, 2)
            judge_scores = judge_answer(item["question"], answer, context)

            results.append(
                {
                    **item,
                    "method": method,
                    "ai_answer": answer,
                    "retrieved_sources": retrieved_sources,
                    "latency_seconds": latency,
                    "judge_scores": judge_scores,
                }
            )
            print(f"✅ Completed in {latency}s")

    output_path = OUTPUT_DIR / "experiment_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved experiment results to {output_path}")


if __name__ == "__main__":
    main()