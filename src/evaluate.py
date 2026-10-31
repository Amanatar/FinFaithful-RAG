import csv
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag_qa import ask

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY") or "nvapi-placeholder-mock"
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

benchmark = [
    {"id": 1, "question": "What are the main macroeconomic risk factors mentioned?"},
    {"id": 2, "question": "How do foreign exchange rates affect the company?"},
    {"id": 3, "question": "What are the specific credit risks disclosed?"},
    {"id": 4, "question": "When does the company's fiscal year end?"},
    {"id": 5, "question": "What are the company's main product categories?"},
]


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
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
    except Exception:
        return '{"faithfulness_score": 0, "relevance_score": 0, "reason": "Judge failed"}'


print("🚀 Starting Advanced RAG Evaluation (with LLM-as-a-Judge)...")
print("-" * 50)

results = []
for item in benchmark:
    print(f"\n🔎 Q{item['id']}: {item['question']}")
    start_time = time.time()

    answer, context = ask(item["question"])
    latency = round(time.time() - start_time, 2)

    print("⚖️ Judging answer with Llama-3...")
    judge_result = judge_answer(item["question"], answer, context)

    results.append(
        {
            "id": item["id"],
            "question": item["question"],
            "ai_answer": answer,
            "latency_seconds": latency,
            "judge_scores": judge_result,
        }
    )
    print(f"✅ Completed in {latency}s")

with open("outputs/advanced_rag_evaluation.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 50)
print("🎉 Advanced Evaluation Complete! Saved to outputs/advanced_rag_evaluation.csv")
