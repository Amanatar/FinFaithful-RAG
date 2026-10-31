import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()


def get_client():
    api_key = os.getenv("NVIDIA_API_KEY") or "nvapi-placeholder-mock"
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def load_index():
    repo_root = Path(__file__).resolve().parents[1]
    index_path = repo_root / "outputs" / "faiss.index"
    meta_path = repo_root / "outputs" / "meta.pkl"
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    return index, meta_data


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 4):
    index, meta_data = load_index()
    q_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, k * 2)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk_text = meta_data["chunks"][idx]
        if not chunk_text.strip():
            continue
        results.append(
            {
                "text": chunk_text,
                "src": meta_data["meta"][idx]["source"],
                "score": float(distances[0][i]),
            }
        )

    # Deduplicate by source and text similarity; keep the best 4.
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


def ask(query: str):
    ctx = retrieve(query)
    context_str = "\n\n".join([f"Source: {c['src']}\n{c['text']}" for c in ctx])

    prompt = f"""You are a strict financial auditor AI.
Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say exactly: "Information not found in the provided filings."
3. Do not invent numbers or hallucinate.
4. Cite the source file names in your answer.

Context:
{context_str}

Question: {query}

Answer:"""

    if os.getenv("NVIDIA_API_KEY"):
        try:
            client = get_client()
            completion = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly accurate financial AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=0.7,
                max_tokens=1024,
                stream=False,
            )
            return completion.choices[0].message.content, ctx
        except Exception as e:
            return f"⚠️ NVIDIA NIM API Error: {str(e)}", ctx

    return "⚠️ Missing NVIDIA_API_KEY in .env file.\n\nRaw Context:\n" + context_str[:500], ctx


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main risk factors?"
    print(ask(q)[0])
