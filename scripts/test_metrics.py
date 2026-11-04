import re
from typing import List, Dict, Any

def normalize_number(num_str: str) -> str:
    """Strips currency symbols, commas, percent signs, and spaces."""
    s = re.sub(r"[\$,\s\%]", "", str(num_str)).lower()
    return s

def calculate_numeric_fact_coverage(key_numbers: List[str], answer: str) -> float:
    """Calculates what percentage of key_numbers are present in the answer text."""
    if not key_numbers:
        return 1.0  # N/A or adversarial trap where 0 numbers are expected
    
    answer_norm = normalize_number(answer)
    found_count = 0
    for num in key_numbers:
        # Check raw substring first
        if num in answer:
            found_count += 1
            continue
        # Check normalized form
        norm_num = normalize_number(num)
        if norm_num and norm_num in answer_norm:
            found_count += 1
            continue
        # Check float equivalency (e.g. 214.4 billion vs 214.4)
        num_clean = re.sub(r"[^\d\.]", "", num)
        if num_clean and num_clean in answer:
            found_count += 1

    return found_count / len(key_numbers)

def calculate_recall_at_k(gold_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int = 5) -> float:
    if not gold_chunk_ids:
        return 1.0  # For Category D, no gold chunks exist
    top_k_retrieved = set(retrieved_chunk_ids[:k])
    hits = sum(1 for gid in gold_chunk_ids if gid in top_k_retrieved)
    return hits / len(gold_chunk_ids)

def calculate_mrr_at_k(gold_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int = 10) -> float:
    if not gold_chunk_ids:
        return 1.0  # For Category D
    gold_set = set(gold_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids[:k], 1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0

# Quick test
gold = ["AAPL_2026_102", "MSFT_2026_160"]
ret = ["AAPL_2026_102", "MSFT_2026_160", "AAPL_2026_103"]
print("Recall@5:", calculate_recall_at_k(gold, ret, 5))
print("MRR@10:", calculate_mrr_at_k(gold, ret, 10))
ans = "Apple reported $478,834 million while Microsoft reported $331,839 million [AAPL | Chunk 7]."
print("Fact coverage:", calculate_numeric_fact_coverage(["478,834", "331,839"], ans))
