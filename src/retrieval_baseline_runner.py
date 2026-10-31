import argparse
import csv
from pathlib import Path

from experiment_runner import (
    ask_with_context,
    ensure_pipeline,
    judge_answer,
    load_benchmark,
    retrieve_faiss,
    retrieve_hybrid,
    retrieve_tfidf,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "finfaithful_bench.json"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TF-IDF vs FAISS vs hybrid benchmark comparisons.")
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARK_PATH),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["tfidf", "faiss", "hybrid"],
        help="Retrieval methods to evaluate.",
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

            context = retrieval_methods[method](ticker, item["question"])
            answer = ask_with_context(ticker, item["question"], context)
            retrieved_sources = ", ".join([c["src"] for c in context])
            judge_scores = judge_answer(item["question"], answer, context)

            results.append(
                {
                    **item,
                    "method": method,
                    "ai_answer": answer,
                    "retrieved_sources": retrieved_sources,
                    "judge_scores": judge_scores,
                }
            )
            print("✅ Completed")

    output_path = OUTPUT_DIR / "retrieval_baseline_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved retrieval baseline results to {output_path}")


if __name__ == "__main__":
    main()