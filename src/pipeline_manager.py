import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

try:
    from ticker_lookup import get_cik, get_company_name
except ImportError:  # pragma: no cover - fallback for package-style imports
    from .ticker_lookup import get_cik, get_company_name

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "AmanAtarResearch ataraman9@gmail.com")}
DATA_DIR = Path("data/raw/sec")
TEXT_DIR = Path("data/processed/sec_text")
OUT_DIR = Path("outputs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_filings(ticker: str, limit: int = 3, progress_callback=None):
    """Downloads 10-K filings for a given ticker."""
    cik = get_cik(ticker)
    company_name = get_company_name(ticker)

    if progress_callback:
        progress_callback(f"📡 Downloading {company_name} ({ticker}) filings from SEC EDGAR...")

    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    recent = res.json()["filings"]["recent"]

    downloaded = 0
    for i, form in enumerate(recent["form"]):
        if form == "10-K" and downloaded < limit:
            acc = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]
            f_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"

            if progress_callback:
                progress_callback(f"  ↳ Downloading 10-K from {filing_date}...")

            html = requests.get(f_url, headers=HEADERS, timeout=60).text
            out_file = DATA_DIR / f"{ticker}_{filing_date}.html"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(html)

            downloaded += 1
            time.sleep(0.5)

    return downloaded


def parse_filings(ticker: str, progress_callback=None):
    """Parses downloaded HTML filings into text."""
    if progress_callback:
        progress_callback("📄 Parsing HTML filings into clean text...")

    parsed = 0
    for f in DATA_DIR.glob(f"{ticker}_*.html"):
        with open(f, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file.read(), "lxml")
        text = soup.get_text(separator="\n", strip=True)

        out_path = TEXT_DIR / f.with_suffix(".txt").name
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(text)
        parsed += 1

    return parsed


def chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]


def build_index(ticker: str, progress_callback=None):
    """Builds FAISS vector index for the ticker's filings."""
    if progress_callback:
        progress_callback("🧠 Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks, meta = [], []
    for f in TEXT_DIR.glob(f"{ticker}_*.txt"):
        text = f.read_text(encoding="utf-8")
        for i, c in enumerate(chunk(text)):
            chunks.append(c)
            meta.append({"source": f.name, "chunk_id": i})

    if not chunks:
        raise ValueError(f"No text chunks found for {ticker}. Run download and parse first.")

    if progress_callback:
        progress_callback(f"🔢 Generating vectors for {len(chunks)} chunks...")

    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path = OUT_DIR / f"{ticker}_faiss.index"
    meta_path = OUT_DIR / f"{ticker}_meta.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)

    if progress_callback:
        progress_callback(f"✅ Index built! ({len(chunks)} chunks)")

    return len(chunks)


def run_full_pipeline(ticker: str, progress_callback=None):
    """Runs the complete pipeline: download → parse → index."""
    try:
        downloaded = download_filings(ticker, limit=3, progress_callback=progress_callback)
        parsed = parse_filings(ticker, progress_callback=progress_callback)
        chunks = build_index(ticker, progress_callback=progress_callback)

        return {
            "status": "success",
            "ticker": ticker,
            "filings_downloaded": downloaded,
            "filings_parsed": parsed,
            "chunks_indexed": chunks,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}