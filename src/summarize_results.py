import argparse
import ast
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"


def parse_score(raw):
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception:
        try:
            value = json.loads(raw)
        except Exception:
            return None
    if isinstance(value, dict):
        return value
    return None


def summarize_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    summary = []
    for row in rows:
        scores = parse_score(row.get("judge_scores", ""))
        method = row.get("method", "") or "direct"
        summary.append(
            {
                "id": row.get("id", ""),
                "ticker": row.get("ticker", ""),
                "method": method,
                "question_type": row.get("question_type", ""),
                "faithfulness": scores.get("faithfulness_score") if scores else None,
                "relevance": scores.get("relevance_score") if scores else None,
                "latency_seconds": float(row.get("latency_seconds", 0) or 0),
            }
        )

    return summary


def build_summary(rows):
    by_method = {}
    for row in rows:
        method = row["method"] or "direct"
        by_method.setdefault(method, []).append(row)

    summary_rows = []
    for method, entries in sorted(by_method.items()):
        if not entries:
            continue
        avg_faith = sum(r["faithfulness"] or 0 for r in entries) / len(entries)
        avg_rel = sum(r["relevance"] or 0 for r in entries) / len(entries)
        avg_latency = sum(r["latency_seconds"] for r in entries) / len(entries)
        summary_rows.append(
            {
                "method": method,
                "n_questions": len(entries),
                "avg_faithfulness": round(avg_faith, 3),
                "avg_relevance": round(avg_rel, 3),
                "avg_latency_seconds": round(avg_latency, 3),
            }
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation CSV results into a compact table.")
    parser.add_argument("--input", default=str(OUTPUT_DIR / "experiment_results.csv"), help="CSV file to summarize.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "results_summary.csv"), help="Where to write the summary CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = summarize_csv(input_path)
    summary_rows = build_summary(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n_questions", "avg_faithfulness", "avg_relevance", "avg_latency_seconds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
