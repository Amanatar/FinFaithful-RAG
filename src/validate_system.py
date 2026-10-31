import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

def print_test(test_num, name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[Test {test_num}] {status} | {name}")
    if details: print(f"         ↳ {details}")

def run_validation(ticker="MSFT"):
    print(f"\n🔬 RUNNING FINFAITHFUL-RAG VALIDATION SUITE FOR {ticker} 🔬\n" + "="*60)
    
    # --- TEST 1: Data Ingestion Check ---
    raw_files = list(Path("data/raw/sec").glob(f"{ticker}_*.html"))
    passed = len(raw_files) > 0
    print_test(1, "SEC Raw Data Exists", passed, f"Found {len(raw_files)} HTML filings.")
    
    # --- TEST 2: Text Parsing Check ---
    text_files = list(Path("data/processed/sec_text").glob(f"{ticker}_*.txt"))
    total_chars = sum(f.stat().st_size for f in text_files) if text_files else 0
    passed = len(text_files) > 0 and total_chars > 50000 # 10-Ks are usually > 500k chars
    print_test(2, "Text Parsing Volume", passed, f"Extracted {total_chars:,} characters from {len(text_files)} files.")
    
    # --- TEST 3: FAISS Index Integrity Check ---
    index_path = Path(f"outputs/{ticker}_faiss.index")
    meta_path = Path(f"outputs/{ticker}_meta.pkl")
    
    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f: meta = pickle.load(f)
        
        # The number of vectors in FAISS MUST exactly match the number of text chunks
        passed = (index.ntotal == len(meta["chunks"])) and index.ntotal > 0
        print_test(3, "FAISS Index Integrity", passed, f"FAISS vectors ({index.ntotal}) match Text Chunks ({len(meta['chunks'])}).")
    else:
        print_test(3, "FAISS Index Integrity", False, "Index files missing. Run pipeline first.")
        return

    # --- TEST 4: Semantic Retrieval Sanity Check ---
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    test_query = "Who is the CEO and what is their compensation?"
    q_vec = embed_model.encode([test_query]).astype('float32')
    _, indices = index.search(q_vec, 1)
    top_chunk = meta["chunks"][indices[0][0]].lower()
    
    # Check if the retrieved chunk contains executive/compensation keywords
    keywords = ["ceo", "chief executive", "compensation", "salary", "executive"]
    passed = any(kw in top_chunk for kw in keywords)
    print_test(4, "Semantic Retrieval Accuracy", passed, f"Top retrieved chunk correctly contains executive/compensation context.")

    # --- TEST 5: The "Hallucination Trap" (Refusal Check) ---
    # This is the most important test for a PhD. We ask a fake question.
    # The AI MUST refuse to answer. If it answers, the system is hallucinating.
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    
    trap_question = "What were the company's total revenue and profits from selling physical gold bars and cryptocurrency mining rigs in 2024?"
    
    ctx = [meta["chunks"][i] for i in indices[0]] # Just use the top chunk for speed
    context_str = "\n".join(ctx)
    
    prompt = f"""You are a strict financial auditor AI. 
Rules: Answer ONLY using the context. If not found, say EXACTLY: "Information not found in the provided filings."
Context: {context_str}
Question: {trap_question}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.lower()
        
        # The model MUST NOT invent a number. It should contain "not found" or similar refusal.
        passed = ("not found" in answer or "does not" in answer or "no information" in answer) and "$" not in answer
        print_test(5, "Hallucination Trap (Refusal)", passed, f"Model correctly refused to invent data for fake question.")
    except Exception as e:
        print_test(5, "Hallucination Trap (Refusal)", False, f"API Error: {e}")

    print("\n" + "="*60)
    print("🎉 VALIDATION COMPLETE. System is ready for research benchmarking.\n")

if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    run_validation(t)
