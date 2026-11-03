"""
FinFaithful-RAG: Benchmark Builder & Ground-Truth Evaluation Suite (Phase 3)

Constructs, audits, and validates the 120-question SEC 10-K ground-truth benchmark
covering AAPL, MSFT, NVDA, JPM, and GOOGL across 4 categories (30 questions each):
  1. table_numeric (exact line-item extraction from tables)
  2. yoy_footnote_reasoning (multi-step YoY synthesis & footnote reasoning)
  3. cross_company_comparative (multi-ticker cross-company comparisons)
  4. adversarial_unanswerable (unanswerable hallucination traps with must_abstain=True)

Maintains full backwards compatibility with legacy evaluation routines.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for safe Windows UTF-8 execution
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "eval"
BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"
OUTPUT_DIR = REPO_ROOT / "outputs"
SEC_INDEX_DIR = REPO_ROOT / "artifacts" / "sec_index"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY") or "nvapi-placeholder-mock",
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

# =============================================================================
# 120-QUESTION GROUND-TRUTH SUITE DEFINITION
# =============================================================================

BENCHMARK_120_DATA: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # CATEGORY A: table_numeric (30 Qs: 6 AAPL, 6 MSFT, 6 NVDA, 6 JPM, 6 GOOGL)
    # -------------------------------------------------------------------------
    # AAPL (6)
    {
        "question_id": "SEC_BENCH_001",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was Apple's total iPhone net sales in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Apple reported total iPhone net sales of $254,098 million (or approximately $254.1 billion).",
        "key_numbers": ["254,098"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_002",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was Apple's Services net sales in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Apple reported Services net sales of $123,301 million.",
        "key_numbers": ["123,301"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_003",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was Apple's total net sales across all products and services in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Apple's consolidated total net sales were $478,834 million.",
        "key_numbers": ["478,834"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_004",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "What were Apple's Research and Development (R&D) expenses in fiscal year 2026?",
        "ground_truth_answer": "Apple recorded Research and Development (R&D) expenses of $45,765 million in fiscal year 2026.",
        "key_numbers": ["45,765"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_125"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_005",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "What was Apple's total gross margin in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Apple generated a total gross margin of $233,602 million.",
        "key_numbers": ["233,602"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_125"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_006",
        "category": "table_numeric",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "What was Apple's cash and cash equivalents balance as of September 26, 2026?",
        "ground_truth_answer": "As of September 26, 2026, Apple held $36,305 million in cash and cash equivalents.",
        "key_numbers": ["36,305"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_130"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # MSFT (6)
    {
        "question_id": "SEC_BENCH_007",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was Microsoft's Intelligent Cloud segment revenue in fiscal year 2026?",
        "ground_truth_answer": "Microsoft reported Intelligent Cloud segment revenue of $137,791 million in fiscal year 2026.",
        "key_numbers": ["137,791"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_159"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_008",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was Microsoft's Productivity and Business Processes segment revenue in fiscal year 2026?",
        "ground_truth_answer": "Microsoft reported Productivity and Business Processes segment revenue of $139,996 million in fiscal year 2026.",
        "key_numbers": ["139,996"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_159"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_009",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was Microsoft's Server products and cloud services revenue in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Microsoft reported Server products and cloud services revenue of $129,425 million.",
        "key_numbers": ["129,425"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_164"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_010",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was total Microsoft Cloud revenue in fiscal year 2026 as disclosed in the segment commentary?",
        "ground_truth_answer": "Microsoft Cloud revenue was $214.4 billion in fiscal year 2026.",
        "key_numbers": ["214.4 billion"],
        "unit": "$ billions",
        "gold_chunk_ids": ["MSFT_2026_165"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_011",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was Microsoft's total assets balance as of June 30, 2026?",
        "ground_truth_answer": "As of June 30, 2026, Microsoft reported total assets of $758,376 million.",
        "key_numbers": ["758,376"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_15"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_012",
        "category": "table_numeric",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was Microsoft's LinkedIn revenue in fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, Microsoft reported LinkedIn revenue of $19,817 million.",
        "key_numbers": ["19,817"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_164"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # NVDA (6)
    {
        "question_id": "SEC_BENCH_013",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was NVIDIA's total revenue for fiscal year 2026?",
        "ground_truth_answer": "NVIDIA's total revenue for fiscal year 2026 was $215,938 million.",
        "key_numbers": ["215,938"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_014",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was NVIDIA's Compute & Networking segment revenue for fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, NVIDIA reported Compute & Networking segment revenue of $193,479 million.",
        "key_numbers": ["193,479"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_015",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was NVIDIA's Graphics segment revenue for fiscal year 2026?",
        "ground_truth_answer": "In fiscal year 2026, NVIDIA reported Graphics segment revenue of $22,459 million.",
        "key_numbers": ["22,459"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_016",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was NVIDIA's GAAP operating income for fiscal year 2026?",
        "ground_truth_answer": "NVIDIA reported GAAP operating income of $130,387 million for fiscal year 2026.",
        "key_numbers": ["130,387"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_017",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "What were NVIDIA's Research and Development (R&D) expenses in fiscal year 2026?",
        "ground_truth_answer": "NVIDIA recorded Research and Development (R&D) expenses of $18,497 million in fiscal year 2026.",
        "key_numbers": ["18,497"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_197"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_018",
        "category": "table_numeric",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "What was NVIDIA's cash and cash equivalents balance as of January 25, 2026?",
        "ground_truth_answer": "As of January 25, 2026, NVIDIA held $10,605 million in cash and cash equivalents.",
        "key_numbers": ["10,605"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_202"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # JPM (6)
    {
        "question_id": "SEC_BENCH_019",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was JPMorgan Chase's total net revenue in 2025?",
        "ground_truth_answer": "In 2025, JPMorgan Chase reported total net revenue of $182,447 million.",
        "key_numbers": ["182,447"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_020",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was JPMorgan Chase's net interest income in 2025?",
        "ground_truth_answer": "JPMorgan Chase reported net interest income of $95,443 million in 2025.",
        "key_numbers": ["95,443"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_021",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was JPMorgan Chase's total noninterest expense in 2025?",
        "ground_truth_answer": "JPMorgan Chase recorded total noninterest expense of $95,640 million in 2025.",
        "key_numbers": ["95,640"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_022",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was JPMorgan Chase's provision for credit losses in 2025?",
        "ground_truth_answer": "In 2025, JPMorgan Chase reported a provision for credit losses of $14,212 million.",
        "key_numbers": ["14,212"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_023",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What were JPMorgan Chase's total deposits as of December 31, 2025?",
        "ground_truth_answer": "As of December 31, 2025, JPMorgan Chase held total deposits of $2,559,320 million (approx. $2.56 trillion).",
        "key_numbers": ["2,559,320"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_175"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_024",
        "category": "table_numeric",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was JPMorgan Chase's consolidated net income in 2025?",
        "ground_truth_answer": "In 2025, JPMorgan Chase achieved net income of $57,048 million.",
        "key_numbers": ["57,048"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # GOOGL (6)
    {
        "question_id": "SEC_BENCH_025",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What were Alphabet's total revenues in fiscal year 2025?",
        "ground_truth_answer": "Alphabet reported total consolidated revenues of $402,836 million in fiscal year 2025.",
        "key_numbers": ["402,836"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_026",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was Google Cloud segment revenue in fiscal year 2025?",
        "ground_truth_answer": "In fiscal year 2025, Google Cloud segment revenue reached $58,705 million.",
        "key_numbers": ["58,705"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_027",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was Google Search & other advertising revenue in fiscal year 2025?",
        "ground_truth_answer": "In fiscal year 2025, Google Search & other revenue was $224,532 million.",
        "key_numbers": ["224,532"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_028",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was YouTube advertising revenue in fiscal year 2025?",
        "ground_truth_answer": "YouTube advertising revenue in fiscal year 2025 was $40,367 million.",
        "key_numbers": ["40,367"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_029",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was Alphabet's income from operations in fiscal year 2025?",
        "ground_truth_answer": "Alphabet generated total income from operations of $129,039 million in fiscal year 2025.",
        "key_numbers": ["129,039"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_132"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_030",
        "category": "table_numeric",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 8",
        "question": "What were Alphabet's Research and Development (R&D) expenses in fiscal year 2025?",
        "ground_truth_answer": "Alphabet recorded Research and Development expenses of $61,087 million in fiscal year 2025.",
        "key_numbers": ["61,087"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_183"],
        "must_abstain": False,
        "abstention_reason": None,
    },

    # -------------------------------------------------------------------------
    # CATEGORY B: yoy_footnote_reasoning (30 Qs: 6 AAPL, 6 MSFT, 6 NVDA, 6 JPM, 6 GOOGL)
    # -------------------------------------------------------------------------
    # AAPL (6)
    {
        "question_id": "SEC_BENCH_031",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the year-over-year percentage growth in Apple's iPhone net sales from fiscal 2025 to 2026, and what was the reported operational driver?",
        "ground_truth_answer": "Apple's iPhone net sales increased by 21% from $209,586 million in 2025 to $254,098 million in 2026, primarily driven by higher net sales of Pro models.",
        "key_numbers": ["254,098", "209,586", "21%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_032",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "How much did Apple's Services net sales increase year-over-year in fiscal 2026 in percentage terms and what were the reported revenue levels in 2025 and 2026?",
        "ground_truth_answer": "Apple Services net sales increased 13% from $109,158 million in 2025 to $123,301 million in 2026.",
        "key_numbers": ["123,301", "109,158", "13%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_033",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the year-over-year growth percentage for Apple's Mac segment from fiscal 2025 to 2026, and what product line drove this change?",
        "ground_truth_answer": "Mac net sales increased 13% from $33,708 million in 2025 to $37,929 million in 2026, driven by higher net sales of laptops.",
        "key_numbers": ["37,929", "33,708", "13%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_034",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "By how much did Apple's Research and Development expenses increase in dollar terms from fiscal 2025 to 2026?",
        "ground_truth_answer": "Apple's R&D expenses increased by $11,215 million, from $34,550 million in 2025 to $45,765 million in 2026.",
        "key_numbers": ["45,765", "34,550"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_125"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_035",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "How did Apple's cash generated by operating activities change between fiscal 2025 and 2026?",
        "ground_truth_answer": "Cash generated by operating activities increased by $33,615 million, rising from $111,482 million in 2025 to $145,097 million in 2026.",
        "key_numbers": ["145,097", "111,482"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_137"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_036",
        "category": "yoy_footnote_reasoning",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the consolidated total net sales growth percentage for Apple between fiscal 2025 and fiscal 2026?",
        "ground_truth_answer": "Apple's total net sales increased 15% from $416,161 million in 2025 to $478,834 million in 2026.",
        "key_numbers": ["478,834", "416,161", "15%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # MSFT (6)
    {
        "question_id": "SEC_BENCH_037",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "By what dollar amount did Microsoft's Intelligent Cloud segment revenue increase from fiscal 2025 to 2026?",
        "ground_truth_answer": "Microsoft Intelligent Cloud revenue increased by $31,526 million, growing from $106,265 million in 2025 to $137,791 million in 2026.",
        "key_numbers": ["137,791", "106,265"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_159"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_038",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was the dollar change in Microsoft's Productivity and Business Processes segment revenue between fiscal 2025 and 2026?",
        "ground_truth_answer": "Productivity and Business Processes segment revenue increased by $19,186 million, rising from $120,810 million in 2025 to $139,996 million in 2026.",
        "key_numbers": ["139,996", "120,810"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_159"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_039",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "How much did Microsoft's Server products and cloud services revenue grow in dollar terms from fiscal 2025 to 2026?",
        "ground_truth_answer": "Server products and cloud services revenue increased by $30,990 million, from $98,435 million in 2025 to $129,425 million in 2026.",
        "key_numbers": ["129,425", "98,435"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_164"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_040",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "By how many billions of dollars did Microsoft Cloud revenue increase between fiscal 2025 and 2026 according to the segment footnote?",
        "ground_truth_answer": "Microsoft Cloud revenue increased by $45.5 billion, rising from $168.9 billion in 2025 to $214.4 billion in 2026.",
        "key_numbers": ["214.4 billion", "168.9 billion"],
        "unit": "$ billions",
        "gold_chunk_ids": ["MSFT_2026_165"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_041",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was the change in Microsoft's total consolidated operating income from fiscal 2025 to 2026?",
        "ground_truth_answer": "Microsoft's operating income increased by $26,709 million, advancing from $128,528 million in 2025 to $155,237 million in 2026.",
        "key_numbers": ["155,237", "128,528"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_042",
        "category": "yoy_footnote_reasoning",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A",
        "question": "What was the year-over-year dollar increase in Microsoft's total revenue from fiscal 2025 to 2026?",
        "ground_truth_answer": "Microsoft's total revenue increased by $50,115 million, expanding from $281,724 million in 2025 to $331,839 million in 2026.",
        "key_numbers": ["331,839", "281,724"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # NVDA (6)
    {
        "question_id": "SEC_BENCH_043",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the dollar change and percentage growth of NVIDIA's Compute & Networking segment from fiscal 2025 to 2026?",
        "ground_truth_answer": "Compute & Networking revenue increased by $77,286 million or 67%, expanding from $116,193 million in 2025 to $193,479 million in 2026.",
        "key_numbers": ["193,479", "116,193", "77,286", "67%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["NVDA_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_044",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the dollar change and percentage growth of NVIDIA's Graphics segment from fiscal 2025 to 2026?",
        "ground_truth_answer": "Graphics revenue increased by $8,155 million or 57%, rising from $14,304 million in 2025 to $22,459 million in 2026.",
        "key_numbers": ["22,459", "14,304", "8,155", "57%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["NVDA_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_045",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the total revenue growth percentage for NVIDIA in fiscal 2026 compared to 2025, and what were the reported revenue figures?",
        "ground_truth_answer": "NVIDIA total revenue increased 65%, rising from $130,497 million in 2025 to $215,938 million in 2026.",
        "key_numbers": ["215,938", "130,497", "65%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_046",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "How did NVIDIA's GAAP gross margin change between fiscal year 2025 and 2026 in basis points or percentage terms?",
        "ground_truth_answer": "NVIDIA's gross margin declined by 3.9 percentage points (-3.9 pts), moving from 75.0% in 2025 to 71.1% in 2026.",
        "key_numbers": ["71.1%", "75.0%", "-3.9 pts"],
        "unit": "percentage / points",
        "gold_chunk_ids": ["NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_047",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "What was the growth in NVIDIA's GAAP net income between fiscal 2025 and 2026?",
        "ground_truth_answer": "NVIDIA net income grew 65%, increasing from $72,880 million in 2025 to $120,067 million in 2026.",
        "key_numbers": ["120,067", "72,880", "65%"],
        "unit": "$ millions / percentage",
        "gold_chunk_ids": ["NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_048",
        "category": "yoy_footnote_reasoning",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "By how much did NVIDIA's Research and Development expenses increase in dollar terms from fiscal 2025 to 2026?",
        "ground_truth_answer": "NVIDIA's R&D expenses increased by $5,583 million, from $12,914 million in 2025 to $18,497 million in 2026.",
        "key_numbers": ["18,497", "12,914"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_197"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # JPM (6)
    {
        "question_id": "SEC_BENCH_049",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was the dollar change in JPMorgan Chase's total net revenue between 2024 and 2025?",
        "ground_truth_answer": "JPMorgan Chase's total net revenue increased by $4,891 million, from $177,556 million in 2024 to $182,447 million in 2025.",
        "key_numbers": ["182,447", "177,556"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_050",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By how much did JPMorgan Chase's net interest income change between 2024 and 2025?",
        "ground_truth_answer": "Net interest income increased by $2,860 million, from $92,583 million in 2024 to $95,443 million in 2025.",
        "key_numbers": ["95,443", "92,583"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_051",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "How did JPMorgan Chase's total noninterest expense change from 2024 to 2025?",
        "ground_truth_answer": "Total noninterest expense increased by $3,843 million, from $91,797 million in 2024 to $95,640 million in 2025.",
        "key_numbers": ["95,640", "91,797"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_052",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By how much did JPMorgan Chase increase its provision for credit losses between 2024 and 2025?",
        "ground_truth_answer": "The provision for credit losses rose by $3,534 million, from $10,678 million in 2024 to $14,212 million in 2025.",
        "key_numbers": ["14,212", "10,678"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_053",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By how much did JPMorgan Chase's total assets expand between year-end 2024 and year-end 2025?",
        "ground_truth_answer": "Total assets increased by $422,086 million, expanding from $4,002,814 million in 2024 to $4,424,900 million in 2025.",
        "key_numbers": ["4,424,900", "4,002,814"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_175"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_054",
        "category": "yoy_footnote_reasoning",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By what dollar amount did total deposits at JPMorgan Chase grow between 2024 and 2025?",
        "ground_truth_answer": "Total deposits increased by $153,288 million, from $2,406,032 million in 2024 to $2,559,320 million in 2025.",
        "key_numbers": ["2,559,320", "2,406,032"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_175"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # GOOGL (6)
    {
        "question_id": "SEC_BENCH_055",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By how much did Alphabet's consolidated total revenues increase in dollar terms between fiscal 2024 and 2025?",
        "ground_truth_answer": "Alphabet's revenues grew by $52,818 million, from $350,018 million in 2024 to $402,836 million in 2025.",
        "key_numbers": ["402,836", "350,018"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_056",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By what dollar amount did Google Cloud segment revenue increase from fiscal 2024 to 2025?",
        "ground_truth_answer": "Google Cloud revenue expanded by $15,476 million, from $43,229 million in 2024 to $58,705 million in 2025.",
        "key_numbers": ["58,705", "43,229"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_057",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "How did Google Cloud operating income change between fiscal 2024 and fiscal 2025?",
        "ground_truth_answer": "Google Cloud operating income more than doubled, increasing by $7,798 million from $6,112 million in 2024 to $13,910 million in 2025.",
        "key_numbers": ["13,910", "6,112"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_132"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_058",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "By how much did Google Search & other advertising revenues grow in dollar terms from 2024 to 2025?",
        "ground_truth_answer": "Google Search & other advertising revenues increased by $26,448 million, rising from $198,084 million in 2024 to $224,532 million in 2025.",
        "key_numbers": ["224,532", "198,084"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_059",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "What was the dollar increase in YouTube advertising revenue from fiscal 2024 to 2025?",
        "ground_truth_answer": "YouTube advertising revenue increased by $4,220 million, from $36,147 million in 2024 to $40,367 million in 2025.",
        "key_numbers": ["40,367", "36,147"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_060",
        "category": "yoy_footnote_reasoning",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "Item 8",
        "question": "By what dollar amount did Alphabet's Research and Development expenses increase between fiscal 2024 and 2025?",
        "ground_truth_answer": "Alphabet's R&D expenses increased by $11,761 million, from $49,326 million in 2024 to $61,087 million in 2025.",
        "key_numbers": ["61,087", "49,326"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_183"],
        "must_abstain": False,
        "abstention_reason": None,
    },

    # -------------------------------------------------------------------------
    # CATEGORY C: cross_company_comparative (30 Qs: 6 per primary ticker)
    # -------------------------------------------------------------------------
    # AAPL Primary (6)
    {
        "question_id": "SEC_BENCH_061",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "MSFT"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7 & Item 7A",
        "question": "Compare total annual revenues between Apple and Microsoft in fiscal year 2026.",
        "ground_truth_answer": "Apple reported total net sales of $478,834 million in fiscal 2026, compared to Microsoft's total revenue of $331,839 million, making Apple's revenue higher by $146,995 million.",
        "key_numbers": ["478,834", "331,839"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102", "MSFT_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_062",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "GOOGL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare total annual revenues between Apple (FY2026) and Alphabet (FY2025).",
        "ground_truth_answer": "Apple generated $478,834 million in net sales in fiscal 2026, while Alphabet reported $402,836 million in total revenues in fiscal 2025.",
        "key_numbers": ["478,834", "402,836"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_063",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "NVDA"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7",
        "question": "Compare total annual revenues between Apple and NVIDIA in fiscal year 2026.",
        "ground_truth_answer": "In fiscal 2026, Apple generated $478,834 million in total net sales, compared to NVIDIA's total revenue of $215,938 million.",
        "key_numbers": ["478,834", "215,938"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102", "NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_064",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "MSFT"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 7 & Item 7A",
        "question": "Compare Apple's Services revenue with Microsoft's Server products and cloud services revenue in fiscal year 2026.",
        "ground_truth_answer": "Apple recorded Services net sales of $123,301 million, while Microsoft recorded Server products and cloud services revenue of $129,425 million in fiscal 2026.",
        "key_numbers": ["123,301", "129,425"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102", "MSFT_2026_164"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_065",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "GOOGL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare Apple's iPhone net sales in FY2026 with Alphabet's Google Search & other advertising revenues in FY2025.",
        "ground_truth_answer": "Apple's iPhone net sales were $254,098 million in fiscal 2026, while Alphabet's Google Search & other revenue was $224,532 million in fiscal 2025.",
        "key_numbers": ["254,098", "224,532"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_102", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_066",
        "category": "cross_company_comparative",
        "tickers": ["AAPL", "NVDA"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "Item 8 & Item 7",
        "question": "Compare consolidated annual Net Income between Apple and NVIDIA in fiscal year 2026.",
        "ground_truth_answer": "In fiscal 2026, Apple achieved consolidated Net Income of $132,175 million, compared to NVIDIA's Net Income of $120,067 million.",
        "key_numbers": ["132,175", "120,067"],
        "unit": "$ millions",
        "gold_chunk_ids": ["AAPL_2026_126", "NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # MSFT Primary (6)
    {
        "question_id": "SEC_BENCH_067",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "GOOGL"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7A & Item 7",
        "question": "Compare total cloud revenues between Microsoft Cloud (FY2026) and Google Cloud (FY2025).",
        "ground_truth_answer": "Microsoft Cloud revenue was $214.4 billion in fiscal 2026, whereas Google Cloud segment revenue was $58,705 million ($58.7 billion) in fiscal 2025.",
        "key_numbers": ["214.4 billion", "58,705"],
        "unit": "$ billions / $ millions",
        "gold_chunk_ids": ["MSFT_2026_165", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_068",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "GOOGL"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7A & Item 7",
        "question": "Compare GAAP operating income between Microsoft (FY2026) and Alphabet (FY2025).",
        "ground_truth_answer": "Microsoft reported GAAP operating income of $155,237 million in fiscal 2026, compared to Alphabet's operating income of $129,039 million in fiscal 2025.",
        "key_numbers": ["155,237", "129,039"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_160", "GOOGL_2025_132"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_069",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "NVDA"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A & Item 7",
        "question": "Compare GAAP operating income between Microsoft and NVIDIA in fiscal year 2026.",
        "ground_truth_answer": "In fiscal 2026, Microsoft recorded operating income of $155,237 million, while NVIDIA reported operating income of $130,387 million.",
        "key_numbers": ["155,237", "130,387"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_160", "NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_070",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "GOOGL"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7A & Item 8",
        "question": "Compare total assets between Microsoft (June 30, 2026) and Alphabet (December 31, 2025).",
        "ground_truth_answer": "Microsoft reported total assets of $758,376 million as of June 30, 2026, compared to Alphabet's total assets of $595,281 million as of December 31, 2025.",
        "key_numbers": ["758,376", "595,281"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_15", "GOOGL_2025_179"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_071",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "GOOGL"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7A & Item 7",
        "question": "Compare full-year revenue between Microsoft (FY2026) and Alphabet (FY2025).",
        "ground_truth_answer": "Microsoft's fiscal 2026 total revenue was $331,839 million, while Alphabet's fiscal 2025 total revenues were $402,836 million.",
        "key_numbers": ["331,839", "402,836"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_160", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_072",
        "category": "cross_company_comparative",
        "tickers": ["MSFT", "NVDA"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "Item 7A & Item 7",
        "question": "Compare Microsoft's Server products and cloud services revenue with NVIDIA's Compute & Networking revenue in fiscal year 2026.",
        "ground_truth_answer": "Microsoft generated $129,425 million in Server products and cloud services, compared to NVIDIA's Compute & Networking revenue of $193,479 million in fiscal 2026.",
        "key_numbers": ["129,425", "193,479"],
        "unit": "$ millions",
        "gold_chunk_ids": ["MSFT_2026_164", "NVDA_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # NVDA Primary (6)
    {
        "question_id": "SEC_BENCH_073",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "GOOGL"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare total annual revenues between NVIDIA (FY2026) and Alphabet (FY2025).",
        "ground_truth_answer": "In fiscal 2026, NVIDIA reported total revenue of $215,938 million, while Alphabet reported $402,836 million in total revenues for fiscal 2025.",
        "key_numbers": ["215,938", "402,836"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_148", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_074",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "AAPL"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 8",
        "question": "Compare Research and Development expenses between NVIDIA and Apple in fiscal year 2026.",
        "ground_truth_answer": "NVIDIA recorded R&D expenses of $18,497 million, while Apple spent $45,765 million on R&D in fiscal year 2026.",
        "key_numbers": ["18,497", "45,765"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_197", "AAPL_2026_125"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_075",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "GOOGL"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare operating income between NVIDIA (FY2026) and Alphabet (FY2025).",
        "ground_truth_answer": "NVIDIA recorded operating income of $130,387 million in fiscal 2026, slightly exceeding Alphabet's income from operations of $129,039 million in fiscal 2025.",
        "key_numbers": ["130,387", "129,039"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_148", "GOOGL_2025_132"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_076",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "GOOGL"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare NVIDIA's Compute & Networking segment revenue (FY2026) with Google Cloud revenue (FY2025).",
        "ground_truth_answer": "NVIDIA's Compute & Networking revenue reached $193,479 million in fiscal 2026, compared to Google Cloud revenue of $58,705 million in fiscal 2025.",
        "key_numbers": ["193,479", "58,705"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_160", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_077",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "JPM"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare annual Net Income between NVIDIA (FY2026) and JPMorgan Chase (2025).",
        "ground_truth_answer": "In fiscal 2026, NVIDIA achieved net income of $120,067 million, compared to JPMorgan Chase's net income of $57,048 million in 2025.",
        "key_numbers": ["120,067", "57,048"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_148", "JPM_2026_228"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_078",
        "category": "cross_company_comparative",
        "tickers": ["NVDA", "MSFT"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "Item 8 & Item 7A",
        "question": "Compare total assets between NVIDIA (as of January 25, 2026) and Microsoft (as of June 30, 2026).",
        "ground_truth_answer": "NVIDIA reported total assets of $206,803 million as of January 25, 2026, compared to Microsoft's total assets of $758,376 million as of June 30, 2026.",
        "key_numbers": ["206,803", "758,376"],
        "unit": "$ millions",
        "gold_chunk_ids": ["NVDA_2026_202", "MSFT_2026_15"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # JPM Primary (6)
    {
        "question_id": "SEC_BENCH_079",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "AAPL"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7 & Item 8",
        "question": "Compare total assets between JPMorgan Chase (as of December 31, 2025) and Apple (as of September 26, 2026).",
        "ground_truth_answer": "JPMorgan Chase held total assets of $4,424,900 million ($4.42 trillion), compared to Apple's total assets of $404,548 million.",
        "key_numbers": ["4,424,900", "404,548"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_175", "AAPL_2026_130"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_080",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "MSFT"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7 & Item 7A",
        "question": "Compare total assets between JPMorgan Chase (December 31, 2025) and Microsoft (June 30, 2026).",
        "ground_truth_answer": "JPMorgan Chase reported total assets of $4,424,900 million ($4.42 trillion), compared to Microsoft's total assets of $758,376 million.",
        "key_numbers": ["4,424,900", "758,376"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_175", "MSFT_2026_15"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_081",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "NVDA"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare total revenue between JPMorgan Chase (2025) and NVIDIA (FY2026).",
        "ground_truth_answer": "JPMorgan Chase recorded total net revenue of $182,447 million in 2025, compared to NVIDIA's total revenue of $215,938 million in fiscal 2026.",
        "key_numbers": ["182,447", "215,938"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228", "NVDA_2026_148"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_082",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "MSFT"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7 & Item 7A",
        "question": "Compare total revenue between JPMorgan Chase (2025) and Microsoft (FY2026).",
        "ground_truth_answer": "JPMorgan Chase reported total net revenue of $182,447 million in 2025, while Microsoft reported total revenue of $331,839 million in fiscal 2026.",
        "key_numbers": ["182,447", "331,839"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228", "MSFT_2026_160"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_083",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "GOOGL"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "question": "Compare total annual revenue between JPMorgan Chase and Alphabet for fiscal year 2025.",
        "ground_truth_answer": "In 2025, JPMorgan Chase reported total net revenue of $182,447 million, compared to Alphabet's total consolidated revenues of $402,836 million.",
        "key_numbers": ["182,447", "402,836"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228", "GOOGL_2025_118"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_084",
        "category": "cross_company_comparative",
        "tickers": ["JPM", "AAPL"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7",
        "question": "Compare total net revenue between JPMorgan Chase (2025) and Apple (FY2026).",
        "ground_truth_answer": "JPMorgan Chase generated total net revenue of $182,447 million in 2025, compared to Apple's total net sales of $478,834 million in fiscal 2026.",
        "key_numbers": ["182,447", "478,834"],
        "unit": "$ millions",
        "gold_chunk_ids": ["JPM_2026_228", "AAPL_2026_102"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    # GOOGL Primary (6)
    {
        "question_id": "SEC_BENCH_085",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "AAPL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 8",
        "question": "Compare Research and Development expenses between Alphabet (FY2025) and Apple (FY2026).",
        "ground_truth_answer": "Alphabet spent $61,087 million on Research and Development in fiscal 2025, compared to Apple's R&D expenditure of $45,765 million in fiscal 2026.",
        "key_numbers": ["61,087", "45,765"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_183", "AAPL_2026_125"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_086",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "NVDA"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 8",
        "question": "Compare Research and Development expenses between Alphabet (FY2025) and NVIDIA (FY2026).",
        "ground_truth_answer": "Alphabet recorded R&D expenses of $61,087 million in fiscal 2025, compared to NVIDIA's R&D expenses of $18,497 million in fiscal 2026.",
        "key_numbers": ["61,087", "18,497"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_183", "NVDA_2026_197"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_087",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "AAPL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 8",
        "question": "Compare cash and cash equivalents between Alphabet (as of December 31, 2025) and Apple (as of September 26, 2026).",
        "ground_truth_answer": "Alphabet held $30,708 million in cash and cash equivalents as of December 31, 2025, compared to Apple's $36,305 million as of September 26, 2026.",
        "key_numbers": ["30,708", "36,305"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_179", "AAPL_2026_130"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_088",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "NVDA"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 8",
        "question": "Compare cash and cash equivalents between Alphabet (December 31, 2025) and NVIDIA (January 25, 2026).",
        "ground_truth_answer": "Alphabet held $30,708 million in cash and cash equivalents, compared to NVIDIA's cash and cash equivalents of $10,605 million.",
        "key_numbers": ["30,708", "10,605"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_179", "NVDA_2026_202"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_089",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "AAPL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 8",
        "question": "Compare total assets between Alphabet (as of December 31, 2025) and Apple (as of September 26, 2026).",
        "ground_truth_answer": "Alphabet reported total assets of $595,281 million as of December 31, 2025, compared to Apple's total assets of $404,548 million as of September 26, 2026.",
        "key_numbers": ["595,281", "404,548"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_179", "AAPL_2026_130"],
        "must_abstain": False,
        "abstention_reason": None,
    },
    {
        "question_id": "SEC_BENCH_090",
        "category": "cross_company_comparative",
        "tickers": ["GOOGL", "MSFT"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025-2026",
        "item_section": "Item 7 & Item 7A",
        "question": "Compare Google Cloud segment revenue (FY2025) with Microsoft's Server products and cloud services revenue (FY2026).",
        "ground_truth_answer": "Google Cloud segment revenue was $58,705 million in fiscal 2025, compared to Microsoft's Server products and cloud services revenue of $129,425 million in fiscal 2026.",
        "key_numbers": ["58,705", "129,425"],
        "unit": "$ millions",
        "gold_chunk_ids": ["GOOGL_2025_118", "MSFT_2026_164"],
        "must_abstain": False,
        "abstention_reason": None,
    },

    # -------------------------------------------------------------------------
    # CATEGORY D: adversarial_unanswerable (30 Qs: 6 AAPL, 6 MSFT, 6 NVDA, 6 JPM, 6 GOOGL)
    # -------------------------------------------------------------------------
    # AAPL Traps (6)
    {
        "question_id": "SEC_BENCH_091",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2028",
        "item_section": "N/A",
        "question": "What is Apple's projected revenue guidance for fiscal year 2028?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The retrieved Form 10-K filings for AAPL only disclose audited financial results through fiscal year 2026. The filings do not contain guidance, forecasts, or projected revenue numbers for fiscal year 2028.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Form 10-K reports only disclose audited historical figures through fiscal 2026 and omit forward-looking quantitative revenue guidance for fiscal 2028.",
    },
    {
        "question_id": "SEC_BENCH_092",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was the exact gross profit margin percentage of the iPhone 16 Pro Max sold in Q3 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K does not disclose product-level or SKU-level gross profit margins for individual hardware models such as the iPhone 16 Pro Max.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Apple reports consolidated product gross margin but keeps individual hardware model unit margins strictly proprietary.",
    },
    {
        "question_id": "SEC_BENCH_093",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was Apple's daily paid subscriber churn rate for Apple TV+ in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K does not disclose subscriber churn rates, daily active metrics, or individual service retention figures for Apple TV+.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Apple does not publish individual streaming service subscriber counts or daily churn statistics in annual SEC filings.",
    },
    {
        "question_id": "SEC_BENCH_094",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was Apple's total annual payment to Amazon Web Services (AWS) for cloud infrastructure hosting in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K does not disclose vendor-specific payments, third-party hosting contracts, or expenditures paid to Amazon Web Services.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Specific vendor contract payments and third-party cloud infrastructure hosting expenses are not disaggregated in the Form 10-K.",
    },
    {
        "question_id": "SEC_BENCH_095",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "What is the scheduled commercial retail launch date and MSRP for Apple's unannounced augmented reality smart glasses?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Form 10-K filings disclose business operations and risk factors, not future unannounced product release dates or retail pricing.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Unannounced products and speculative release schedules are strictly outside the scope of annual Form 10-K disclosures.",
    },
    {
        "question_id": "SEC_BENCH_096",
        "category": "adversarial_unanswerable",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "fiscal_year": "2027",
        "item_section": "N/A",
        "question": "Who has the Board of Directors designated to succeed Tim Cook as Apple's Chief Executive Officer in 2027?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K lists current executive officers and governance procedures, but does not disclose confidential future CEO succession timelines or designated candidates.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Confidential executive succession decisions and future appointments are not disclosed in Form 10-K filings.",
    },
    # MSFT Traps (6)
    {
        "question_id": "SEC_BENCH_097",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was Microsoft's confidential internal gross margin percentage for the Azure OpenAI Service in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K does not disclose individual service-level or workload-level gross margin percentages for Azure OpenAI Service.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Microsoft reports consolidated segment margins but does not disclose service-specific unit economics or gross margins for Azure OpenAI Service.",
    },
    {
        "question_id": "SEC_BENCH_098",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2029",
        "item_section": "N/A",
        "question": "What is Microsoft's long-term projected revenue target for fiscal year 2029?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K only covers audited historical financial results through fiscal 2026 and provides no quantitative revenue projections for fiscal year 2029.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Multi-year forward-looking financial targets are not provided in Microsoft's Form 10-K.",
    },
    {
        "question_id": "SEC_BENCH_099",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was the exact daily active user (DAU) count for Bing Search in December 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Microsoft reports Search advertising revenues in its Form 10-K, but does not disclose exact daily active user (DAU) figures for Bing Search.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Daily active user counts for Bing are not disclosed in Microsoft's annual Form 10-K filing.",
    },
    {
        "question_id": "SEC_BENCH_100",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "What is the undisclosed purchase price Microsoft offered for its next planned video game studio acquisition?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K only discloses closed, legally effective, or publicly announced material acquisitions, and does not disclose speculative or pending bids.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Undisclosed future acquisitions and private negotiation bids are strictly excluded from annual SEC filings.",
    },
    {
        "question_id": "SEC_BENCH_101",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What is the exact silicon manufacturing wafer price Microsoft pays TSMC for custom Maia 100 AI accelerator chips?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Commercial supplier contracts, individual silicon wafer costs, and foundry fabrication pricing with TSMC are trade secrets not disclosed in the Form 10-K.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Individual chip fabrication agreements and foundry wafer unit costs are confidential supplier trade secrets omitted from 10-K reports.",
    },
    {
        "question_id": "SEC_BENCH_102",
        "category": "adversarial_unanswerable",
        "tickers": ["MSFT"],
        "primary_ticker": "MSFT",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was Microsoft's voluntary employee attrition rate in the AI and Cloud engineering division in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Microsoft provides global employee headcounts and human capital management disclosures in its Form 10-K, but does not disclose divisional employee attrition rates.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Specific divisional voluntary attrition rates are not required or disclosed in Form 10-K filings.",
    },
    # NVDA Traps (6)
    {
        "question_id": "SEC_BENCH_103",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2028",
        "item_section": "N/A",
        "question": "What is NVIDIA's projected full-year revenue guidance for fiscal year 2028?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] NVIDIA's Form 10-K contains audited financial statements through fiscal 2026, without forward-looking revenue guidance for fiscal year 2028.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Form 10-K filings do not publish multi-year future financial projections or fiscal 2028 guidance.",
    },
    {
        "question_id": "SEC_BENCH_104",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What is the exact bill-of-materials (BOM) manufacturing cost to build a single NVIDIA GB200 NVL72 rack?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] The Form 10-K reports consolidated cost of revenue, but does not disclose unit-level bill-of-materials (BOM) costs or rack manufacturing costs for the GB200 NVL72.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Granular product-level BOM costs and server rack assembly economics are proprietary manufacturing secrets.",
    },
    {
        "question_id": "SEC_BENCH_105",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was the exact dollar volume discount NVIDIA granted Microsoft for bulk purchases of H100 GPUs in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Bilateral pricing terms, customer-specific rebate programs, and volume discount concessions for specific corporate buyers are not disclosed in NVIDIA's Form 10-K.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Confidential customer pricing agreements and negotiated discounts are not published in SEC reports.",
    },
    {
        "question_id": "SEC_BENCH_106",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "What is the planned consumer retail release date and launch price for NVIDIA's next-generation Rubin GPU architecture?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Form 10-K filings disclose existing product lines and technological risks, but omit future unannounced product release dates or retail pricing.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Commercial launch dates and pricing for unreleased microarchitectures like Rubin are outside annual 10-K reporting.",
    },
    {
        "question_id": "SEC_BENCH_107",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What was NVIDIA's revenue and market share in smartphone mobile applications processors in fiscal 2026?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] NVIDIA does not sell a dedicated line of smartphone mobile applications processors, and the Form 10-K does not provide market share figures for smartphone processors.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "NVIDIA is not an active participant in smartphone application processors and reports no such segment.",
    },
    {
        "question_id": "SEC_BENCH_108",
        "category": "adversarial_unanswerable",
        "tickers": ["NVDA"],
        "primary_ticker": "NVDA",
        "fiscal_year": "2026",
        "item_section": "N/A",
        "question": "What is the internal wafer defect and yield percentage for NVIDIA Blackwell B200 dies on TSMC's 4NP process node?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Silicon foundry yield rates, die defect densities, and internal manufacturing yield percentages are confidential foundry trade secrets not published in Form 10-K filings.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Foundry die yields and semiconductor defect densities are strictly confidential operational data omitted from public filings.",
    },
    # JPM Traps (6)
    {
        "question_id": "SEC_BENCH_109",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What was the exact dollar cash withdrawal volume from JPMorgan Chase ATMs on December 24, 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] JPMorgan Chase's Form 10-K discloses aggregate deposit balances and fee revenues, but does not disclose single-day cash withdrawal volumes for specific dates.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Single-day operational cash transaction volumes are not reported in annual Form 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_110",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2028",
        "item_section": "N/A",
        "question": "What is JPMorgan Chase's confidential internal capital loss forecast under a hypothetical 60% equity market crash in 2028?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Confidential internal stress test models, proprietary loss simulations, and hypothetical projections beyond public regulatory stress testing summaries are not published in the Form 10-K.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Proprietary internal stress scenario projections for 2028 are not published in the annual Form 10-K.",
    },
    {
        "question_id": "SEC_BENCH_111",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "What is JPMorgan Chase's target acquisition valuation for its next private banking acquisition in Switzerland?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Unannounced bank acquisition negotiations, hypothetical buyout targets, and private valuation bids are strictly excluded from annual SEC filings.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Speculative or unconsummated future acquisition bids are strictly omitted from Form 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_112",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What was the average FICO credit score for new Chase Freedom credit card accounts opened in December 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] JPMorgan Chase discloses broad consumer loan credit quality and delinquency brackets, but does not disclose single-month average FICO scores for individual credit card brands.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Product-level single-month average FICO scores are proprietary underwriting metrics not published in 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_113",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "On what exact date will Jamie Dimon officially retire as Chairman and Chief Executive Officer of JPMorgan Chase?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Form 10-K filings disclose current executive leadership and general governance succession policies, but do not state a specific future retirement date for the CEO.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Precise future CEO retirement dates are not scheduled or disclosed in annual Form 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_114",
        "category": "adversarial_unanswerable",
        "tickers": ["JPM"],
        "primary_ticker": "JPM",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What is the exact dollar amount of physical Bitcoin held directly on JPMorgan Chase's balance sheet in proprietary inventory as of year-end 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] JPMorgan Chase does not hold physical Bitcoin in proprietary inventory, and the Form 10-K consolidated balance sheet discloses no proprietary cryptocurrency holdings.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "JPMorgan Chase holds no physical Bitcoin inventory on its balance sheet, and no such line item exists in the 10-K.",
    },
    # GOOGL Traps (6)
    {
        "question_id": "SEC_BENCH_115",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What was the exact total number of search queries handled by Google Search on November 15, 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Alphabet reports advertising revenues and aggregate growth drivers in its Form 10-K, but does not disclose exact daily search query volumes for specific dates.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Single-day search query volumes are proprietary operational metrics not published in Form 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_116",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2028",
        "item_section": "N/A",
        "question": "What is Alphabet's projected revenue guidance for fiscal year 2028?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Alphabet's Form 10-K only covers audited historical results through fiscal 2025 and does not provide multi-year forward revenue guidance for fiscal year 2028.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Alphabet does not provide forward-looking annual revenue guidance for fiscal 2028 in its Form 10-K.",
    },
    {
        "question_id": "SEC_BENCH_117",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What is the unit manufacturing cost paid to TSMC to produce a Google Tensor G5 custom silicon processor?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Commercial supplier contract pricing, custom ASIC fabrication terms, and unit manufacturing costs for Tensor processors are trade secrets not disclosed in the Form 10-K.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Custom processor manufacturing wafer prices and unit assembly costs are proprietary confidential supplier data omitted from SEC filings.",
    },
    {
        "question_id": "SEC_BENCH_118",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What was the exact dollar revenue generated exclusively from the Gemini Advanced AI subscription tier in 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Alphabet reports subscription revenues within the aggregated 'Google subscriptions, platforms, and devices' category and does not break out stand-alone revenue for Gemini Advanced.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Revenues for specific consumer AI subscription tiers are grouped into broader segment lines and not individually disclosed.",
    },
    {
        "question_id": "SEC_BENCH_119",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "Future",
        "item_section": "N/A",
        "question": "What is the official commercial launch date for Waymo's autonomous ride-hailing robotaxi service in London, UK?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Form 10-K filings describe Other Bets segment operations and related business risks, but do not disclose future city-by-city international commercial launch dates for Waymo.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Future commercial operational rollouts in specific foreign metropolitan areas are not published in Form 10-K filings.",
    },
    {
        "question_id": "SEC_BENCH_120",
        "category": "adversarial_unanswerable",
        "tickers": ["GOOGL"],
        "primary_ticker": "GOOGL",
        "fiscal_year": "2025",
        "item_section": "N/A",
        "question": "What was the exact average click-through rate (CTR) percentage for YouTube Shorts ads in the fourth quarter of 2025?",
        "ground_truth_answer": "[INSUFFICIENT_FILING_EVIDENCE] Alphabet reports total YouTube advertising revenue, but does not disclose format-specific or quarterly click-through rates (CTR) for YouTube Shorts in its Form 10-K.",
        "key_numbers": [],
        "unit": "N/A",
        "gold_chunk_ids": [],
        "must_abstain": True,
        "abstention_reason": "Ad-format-specific engagement metrics and click-through rates are proprietary operational data omitted from annual 10-K filings.",
    },
]


def validate_sec_10k_benchmark_120(
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
    index_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs a strict ground-truth verification pass on the 120-question benchmark.
    Checks schema completeness, existence of all gold_chunk_ids in the index,
    and presence of key_numbers as substrings in the text of gold_chunk_ids.
    """
    if benchmark_data is None:
        benchmark_data = BENCHMARK_120_DATA
    if index_path is None:
        index_path = SEC_INDEX_DIR / "chunks_metadata.parquet"

    if not index_path.exists():
        raise FileNotFoundError(f"SEC index file not found at {index_path}")

    df_meta = pd.read_parquet(index_path)
    chunk_map = {r["global_chunk_id"]: r["text"] for _, r in df_meta.iterrows()}

    required_fields = [
        "question_id", "category", "tickers", "primary_ticker", "fiscal_year",
        "item_section", "question", "ground_truth_answer", "key_numbers",
        "unit", "gold_chunk_ids", "must_abstain", "abstention_reason",
    ]

    passed = 0
    failed = 0
    errors = []
    category_counts = {}
    primary_ticker_counts = {}

    for item in benchmark_data:
        qid = item.get("question_id", "UNKNOWN")
        cat = item.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        pt = item.get("primary_ticker", "UNKNOWN")
        primary_ticker_counts[pt] = primary_ticker_counts.get(pt, 0) + 1

        # Check required fields
        missing = [f for f in required_fields if f not in item]
        if missing:
            failed += 1
            errors.append(f"[{qid}] Missing fields: {missing}")
            continue

        # Check gold chunks
        missing_chunks = [cid for cid in item["gold_chunk_ids"] if cid not in chunk_map]
        if missing_chunks:
            failed += 1
            errors.append(f"[{qid}] Gold chunks not in index: {missing_chunks}")
            continue

        # Check key numbers in gold chunks
        if not item["must_abstain"]:
            combined_text = "\n".join(chunk_map[cid] for cid in item["gold_chunk_ids"])
            missing_nums = [n for n in item["key_numbers"] if n not in combined_text]
            if missing_nums:
                failed += 1
                errors.append(f"[{qid}] Key numbers {missing_nums} not in gold chunk text")
                continue
        else:
            if not item["abstention_reason"]:
                failed += 1
                errors.append(f"[{qid}] Adversarial question missing abstention_reason")
                continue
            if len(item["gold_chunk_ids"]) != 0:
                failed += 1
                errors.append(f"[{qid}] Adversarial question should have empty gold_chunk_ids")
                continue

        passed += 1

    stats = {
        "total": len(benchmark_data),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "category_counts": category_counts,
        "primary_ticker_counts": primary_ticker_counts,
        "index_chunks_count": len(chunk_map),
    }
    return (failed == 0), stats


def build_sec_10k_benchmark_120(
    output_path: Optional[Path] = None,
    validate: bool = True,
) -> Path:
    """
    Builds, validates, and serializes the 120-question benchmark to JSONL.
    """
    if output_path is None:
        output_path = DATA_DIR / "sec_10k_benchmark_120.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        is_valid, stats = validate_sec_10k_benchmark_120(BENCHMARK_120_DATA)
        if not is_valid:
            error_msg = "\n".join(stats["errors"])
            raise ValueError(f"Benchmark validation failed with {stats['failed']} errors:\n{error_msg}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in BENCHMARK_120_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path


# =============================================================================
# LEGACY PIPELINE & BENCHMARK COMPATIBILITY FUNCTIONS
# =============================================================================

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


def load_index(ticker: str):
    import faiss
    import pickle
    index_path = OUTPUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUTPUT_DIR / f"{ticker}_meta.pkl"

    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    return index, meta_data


def retrieve(ticker: str, query: str, k: int = 4):
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


def ask_rag(ticker: str, query: str):
    ctx = retrieve(ticker, query)
    if not ctx:
        return f"[INSUFFICIENT_FILING_EVIDENCE] No index found for {ticker}.", []

    context_str = "\n\n".join([f"Source: {c['src']}\n{c['text']}" for c in ctx])
    prompt = f"""You are a strict financial auditor AI analyzing SEC filings for {ticker}.
Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say exactly: "[INSUFFICIENT_FILING_EVIDENCE] Information not found in the provided filings."
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


def judge_answer(question: str, answer: str, context: List[Dict[str, Any]]):
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


def build_benchmark(companies: List[Tuple[str, str]], questions_per_company: int):
    from ticker_lookup import get_company_name
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


def ensure_company_ready(ticker: str):
    from pipeline_manager import run_full_pipeline
    index, meta = load_index(ticker)
    if index is not None and meta is not None:
        return
    run_full_pipeline(ticker, progress_callback=lambda msg: print(f"[{ticker}] {msg}"))


def run_evaluation(benchmark: List[Dict[str, Any]], ensure_pipeline: bool = True):
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


# =============================================================================
# CLI PARSER AND ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and validate FinFaithful-RAG SEC 10-K Ground-Truth Benchmark."
    )
    parser.add_argument(
        "--build-120",
        action="store_true",
        default=True,
        help="Generate and validate data/eval/sec_10k_benchmark_120.jsonl (Default: True).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run strict ground-truth verification pass on existing benchmark file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DATA_DIR / "sec_10k_benchmark_120.jsonl"),
        help="Target output path for benchmark JSONL.",
    )
    # Legacy arguments preserved
    parser.add_argument(
        "--questions-per-company",
        type=int,
        default=20,
        help="Number of questions to generate per company (legacy mode).",
    )
    parser.add_argument(
        "--companies",
        nargs="*",
        default=[ticker for ticker, _ in DEFAULT_COMPANIES],
        help="Tickers to include in the benchmark (legacy mode).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAG system and judge results (legacy mode).",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Do not download/parse/index filings before evaluating (legacy mode).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Priority 1: Flagship 120-Question Benchmark Generation & Validation
    if args.build_120 or args.validate_only:
        out_path = Path(args.output_path)
        print("=" * 70)
        print("FinFaithful-RAG: 120-Question Ground-Truth Suite (Phase 3)")
        print("=" * 70)

        is_valid, stats = validate_sec_10k_benchmark_120(BENCHMARK_120_DATA)
        print(f"• Total Benchmark Items   : {stats['total']}")
        print(f"• Verified Valid Records  : {stats['passed']}")
        print(f"• Validation Failures     : {stats['failed']}")
        print(f"• SEC Corpus Chunks       : {stats['index_chunks_count']}")
        print(f"• Category Breakdown      : {stats['category_counts']}")
        print(f"• Primary Tickers         : {stats['primary_ticker_counts']}")

        if not is_valid:
            print("\n❌ VALIDATION ERRORS DETECTED:")
            for err in stats["errors"]:
                print(f"  - {err}")
            sys.exit(1)

        print("\n✅ STRICT GROUND-TRUTH VERIFICATION: PASSED (100% precision)")

        if not args.validate_only:
            written_path = build_sec_10k_benchmark_120(out_path, validate=False)
            file_size_kb = written_path.stat().st_size / 1024.0
            print(f"💾 Committed Benchmark to: {written_path} ({file_size_kb:.1f} KB)")

        # Print 1 sample from each category
        print("\n" + "=" * 70)
        print("SAMPLE BENCHMARK ENTRIES (1 PER CATEGORY)")
        print("=" * 70)

        samples_by_cat = {}
        for item in BENCHMARK_120_DATA:
            cat = item["category"]
            if cat not in samples_by_cat:
                samples_by_cat[cat] = item

        for cat, sample in samples_by_cat.items():
            print(f"\n[{cat.upper()}] (ID: {sample['question_id']} | Tickers: {sample['tickers']})")
            print(f"  Q : {sample['question']}")
            print(f"  A : {sample['ground_truth_answer']}")
            print(f"  Key Numbers: {sample['key_numbers']} ({sample['unit']})")
            print(f"  Gold Chunks: {sample['gold_chunk_ids']}")
            print(f"  Must Abstain: {sample['must_abstain']} (Reason: {sample['abstention_reason']})")

        print("=" * 70)

    # Legacy evaluate flow
    if args.evaluate:
        companies = []
        for ticker in args.companies:
            sector = next((s for t, s in DEFAULT_COMPANIES if t == ticker), "General")
            companies.append((ticker, sector))

        benchmark = build_benchmark(companies, args.questions_per_company)
        results = run_evaluation(benchmark, ensure_pipeline=not args.skip_pipeline)
        results_path = OUTPUT_DIR / "benchmark_results.csv"
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved legacy evaluation results to {results_path}")


if __name__ == "__main__":
    main()
