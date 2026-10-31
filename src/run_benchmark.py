import os
import csv
import time
import faiss
import pickle
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_index(ticker):
    index = faiss.read_index(f"outputs/{ticker}_faiss.index")
    with open(f"outputs/{ticker}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(index, meta, query, k=3):
    q_vec = embed_model.encode([query]).astype('float32')
    _, indices = index.search(q_vec, k)
    return [meta["chunks"][i] for i in indices[0]]


def ask_llm(context_str, question):
    prompt = f"""You are a strict financial auditor AI. 
Rules: Answer ONLY using the context. If not found, say EXACTLY: "Information not found in the provided filings."
Context: {context_str}
Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content


def judge_answer(question, answer, is_trap):
    if is_trap:
        passed = ("not found" in answer.lower() or "does not" in answer.lower() or "no information" in answer.lower()) and "$" not in answer
        return {"faithfulness": 5 if passed else 1, "relevance": 5 if passed else 1, "refusal_correct": passed}

    judge_prompt = f"""Grade this AI answer from 1 to 5 on Faithfulness (only uses context) and Relevance (answers the question).
Question: {question}
Answer: {answer}
Respond ONLY with JSON: {{"faithfulness": X, "relevance": Y}}"""

    try:
        import json
        res = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        scores = json.loads(res.choices[0].message.content)
        return {"faithfulness": scores.get("faithfulness", 0), "relevance": scores.get("relevance", 0), "refusal_correct": "N/A"}
    except Exception:
        return {"faithfulness": 0, "relevance": 0, "refusal_correct": "N/A"}


if __name__ == "__main__":
    ticker = "MSFT"
    index, meta = load_index(ticker)

    benchmark = [
        {"q": "What are the main macroeconomic risks?", "trap": False},
        {"q": "How do foreign exchange rates affect the company?", "trap": False},
        {"q": "When does the fiscal year end?", "trap": False},
        {"q": "What are the main product categories?", "trap": False},
        {"q": "What was the revenue from selling physical gold bars?", "trap": True},
        {"q": "What is the exact square footage of the CEO's private office?", "trap": True}
    ]

    print(f"🚀 RUNNING BENCHMARK ON {ticker}...\n")
    results = []

    for i, item in enumerate(benchmark):
        print(f"[{i+1}/6] Evaluating: {item['q']}")
        start = time.time()

        ctx = retrieve(index, meta, item["q"])
        context_str = "\n".join(ctx)

        answer = ask_llm(context_str, item["q"])
        latency = round(time.time() - start, 2)

        scores = judge_answer(item["q"], answer, item["trap"])

        results.append({
            "question": item["q"],
            "is_trap": item["trap"],
            "ai_answer": answer.replace("\n", " "),
            "latency_sec": latency,
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "refusal_correct": scores["refusal_correct"]
        })
        print(f"   ↳ Latency: {latency}s | Faithfulness: {scores['faithfulness']}/5 | Refusal: {scores['refusal_correct']}")

    out_path = "outputs/research_benchmark_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ BENCHMARK COMPLETE! Data saved to {out_path}")
    print("Use this CSV to write the 'Experiments' and 'Results' sections of your research paper.")
