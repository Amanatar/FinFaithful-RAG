import argparse
import csv
import json
import os
import time
from pathlib import Path

import faiss
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    from pipeline_manager import run_full_pipeline
    from ticker_lookup import get_company_name
except ImportError:  # pragma: no cover - fallback for package-style imports
    from .pipeline_manager import run_full_pipeline
    from .ticker_lookup import get_company_name


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"
OUTPUT_DIR = REPO_ROOT / "outputs"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DEFAULT_COMPANIES = [
    ("AAPL", "Technology"),
    ("MSFT", "Technology"),
    ("NVDA", "Semiconductors"),
    ("JPM", "Finance"),
    ("PFE", "Healthcare"),
    ("XOM", "Energy"),
    ("WMT", "Consumer"),
    ("TSLA", "Automotive"),
    ("AMZN", "E-commerce"),
    ("GOOGL", "Cloud/AI"),
]

QUESTION_TEMPLATES = {
    "extraction": [
        ("What is the company's fiscal year end?", "Item 8 Financial Statements"),
        ("Which reportable segments does the company disclose?", "Note 13 Segment Information"),
        ("What are the company's main product categories?", "Business / Products and Services"),
        ("Where is the company headquartered?", "Business / Properties / General Company Description"),
    ],
    "comparison": [
        ("Did revenue increase or decrease compared with the prior year?", "Item 7 MD&A"),
        ("Did net income increase or decrease compared with the prior year?", "Item 7 MD&A"),
        ("Did research and development expense increase or decrease compared with the prior year?", "Item 7 MD&A"),
        ("Did operating cash flow improve or worsen compared with the prior year?", "Item 7 MD&A"),
    ],
    "risk_summary": [
        ("What are the main competition risks disclosed?", "Item 1A Risk Factors"),
        ("What supply chain risks are disclosed?", "Item 1A Risk Factors"),
        ("What macroeconomic or foreign exchange risks are disclosed?", "Item 1A Risk Factors"),
        ("What cybersecurity or privacy risks are disclosed?", "Item 1A Risk Factors"),
    ],
    "calculation": [
        ("What is the year-over-year percentage change in revenue?", "Item 7 MD&A"),
        ("What is the year-over-year percentage change in research and development expense?", "Item 7 MD&A"),
        ("What is the year-over-year percentage change in operating income?", "Item 7 MD&A"),
        ("What is the year-over-year percentage change in operating cash flow?", "Item 7 MD&A"),
    ],
    "unanswerable": [
        ("What is the company's planned acquisition target for next year?", "Not disclosed"),
        ("Which product will the company launch next quarter?", "Not disclosed"),
        ("Who is the company's next CEO candidate?", "Not disclosed"),
        ("What is the exact undisclosed customer contract amount for next year?", "Not disclosed"),
    ],
}


def load_index(ticker):
    index_path = OUTPUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUTPUT_DIR / f"{ticker}_meta.pkl"

    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    return index, meta_data


def retrieve(ticker, query, k=4):
    index, meta_data = load_index(ticker)
    if index is None:
        return []

    q_vec = EMBED_MODEL.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, k)

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
    return results


def ask_rag(ticker, query):
    ctx = retrieve(ticker, query)
    if not ctx:
        return f"⚠️ No index found for {ticker}. Please run the pipeline first.", []

    context_str = "\n\n".join([f"Source: {c['src']}\n{c['text']}" for c in ctx])
    prompt = f"""You are a strict financial auditor AI analyzing SEC filings for {ticker}.
Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say exactly: "Information not found in the provided filings."
3. Do not invent numbers or hallucinate.
4. Cite the source file names in your answer.

Context:
{context_str}

Question: {query}

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
    return completion.choices[0].message.content, ctx


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


def build_benchmark(companies, questions_per_company):
    benchmark = []
    question_id = 1

    categories = list(QUESTION_TEMPLATES.items())
    per_category = max(1, questions_per_company // len(categories))

    for ticker, sector in companies:
        company_name = get_company_name(ticker)
        for category, templates in categories:
            for template_index in range(per_category):
                question_template, expected_section = templates[template_index % len(templates)]
                benchmark.append(
                    {
                        "id": f"{ticker}_{question_id:03d}",
                        "ticker": ticker,
                        "company_name": company_name,
                        "sector": sector,
                        "question_type": category,
                        "question": question_template,
                        "answerable": category != "unanswerable",
                        "expected_source_section": expected_section,
                    }
                )
                question_id += 1

    return benchmark


def ensure_company_ready(ticker):
    index, meta = load_index(ticker)
    if index is not None and meta is not None:
        return
    run_full_pipeline(ticker, progress_callback=lambda msg: print(f"[{ticker}] {msg}"))


def run_evaluation(benchmark, ensure_pipeline=True):
    results = []

    for item in benchmark:
        ticker = item["ticker"]
        if ensure_pipeline:
            ensure_company_ready(ticker)

        print(f"\n🔎 {item['id']}: {item['question']}")
        start_time = time.time()
        answer, context = ask_rag(ticker, item["question"])
        latency = round(time.time() - start_time, 2)
        judge_result = judge_answer(item["question"], answer, context)

        results.append(
            {
                **item,
                "ai_answer": answer,
                "retrieved_sources": ", ".join([c["src"] for c in context]),
                "latency_seconds": latency,
                "judge_scores": judge_result,
            }
        )
        print(f"✅ Completed in {latency}s")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Build and optionally evaluate FinFaithful-Bench.")
    parser.add_argument(
        "--questions-per-company",
        type=int,
        default=20,
        help="Number of questions to generate per company.",
    )
    parser.add_argument(
        "--companies",
        nargs="*",
        default=[ticker for ticker, _ in DEFAULT_COMPANIES],
        help="Tickers to include in the benchmark.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the current RAG system and judge results.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Do not download/parse/index filings before evaluating.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    companies = []
    for ticker in args.companies:
        sector = next((sector for company_ticker, sector in DEFAULT_COMPANIES if company_ticker == ticker), "General")
        companies.append((ticker, sector))

    benchmark = build_benchmark(companies, args.questions_per_company)
    benchmark_path = BENCHMARK_DIR / "finfaithful_bench.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Saved benchmark dataset to {benchmark_path}")

    if not args.evaluate:
        return

    results = run_evaluation(benchmark, ensure_pipeline=not args.skip_pipeline)
    results_path = OUTPUT_DIR / "benchmark_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()