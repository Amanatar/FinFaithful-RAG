import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


try:
    from ticker_lookup import get_cik, get_company_name
except ImportError:
    from .ticker_lookup import get_cik, get_company_name

try:
    from sec_parser import parse_and_save_filing
except ImportError:
    from .sec_parser import parse_and_save_filing

RAW_10K_DIR = REPO_ROOT / "data" / "raw_10k"
PROCESSED_10K_DIR = REPO_ROOT / "data" / "processed_10k"
LEGACY_RAW_DIR = REPO_ROOT / "data" / "raw" / "sec"
LEGACY_TEXT_DIR = REPO_ROOT / "data" / "processed" / "sec_text"

RAW_10K_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_10K_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_RAW_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_TEXT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "GOOGL"]
DEFAULT_USER_AGENT = "AmanAtarResearch ataraman9@gmail.com"
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT)}

# Strict rate limit: maximum 5 req/sec -> minimum 0.25s delay
MIN_REQUEST_INTERVAL = 0.25
_LAST_REQUEST_TIME = 0.0


def _rate_limited_request(url: str, timeout: int = 45, max_retries: int = 3) -> requests.Response:
    """Enforces <= 5 req/sec rate limit and handles transient retries."""
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    for attempt in range(max_retries):
        try:
            _LAST_REQUEST_TIME = time.time()
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", 2 * (attempt + 1)))
                print(f"⚠️ Rate limited (429). Sleeping {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def download_filing_for_ticker(
    ticker: str,
    cik: Optional[str] = None,
    limit: int = 1,
    raw_dir: Path = RAW_10K_DIR,
    processed_dir: Path = PROCESSED_10K_DIR,
    auto_parse: bool = True,
) -> List[Path]:
    """
    Downloads latest Form 10-K filings for a single ticker via official SEC EDGAR submissions endpoint.
    Saves raw HTML into raw_dir and automatically parses structured sections into processed_dir.
    """
    ticker = ticker.upper().strip()
    if not cik:
        cik = get_cik(ticker)

    company_title = get_company_name(ticker)
    submissions_url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"

    print(f"\n📡 Fetching SEC EDGAR submissions for {ticker} ({company_title}, CIK: {cik})...")
    res = _rate_limited_request(submissions_url)
    sub_data = res.json()

    recent = sub_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", filing_dates)

    downloaded_files: List[Path] = []
    count = 0

    for i, form in enumerate(forms):
        if form == "10-K" and count < limit:
            acc_raw = accessions[i]
            acc_clean = acc_raw.replace("-", "")
            doc = primary_docs[i]
            filing_date = filing_dates[i]
            report_date = report_dates[i] if i < len(report_dates) else filing_date
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"

            out_html_path = raw_dir / f"{ticker}_{filing_date}.html"
            legacy_html_path = LEGACY_RAW_DIR / f"{ticker}_{filing_date}.html"

            if out_html_path.exists() and out_html_path.stat().st_size > 10000:
                print(f"  ↳ Using cached raw 10-K ({filing_date}) -> {out_html_path.name} ({out_html_path.stat().st_size:,} bytes)")
            else:
                print(f"  ↳ Downloading 10-K ({filing_date}, Period: {report_date})...")
                f_res = _rate_limited_request(filing_url, timeout=90)
                html_text = f_res.text
                out_html_path.write_text(html_text, encoding="utf-8")
                # Also maintain backwards compatibility in legacy dir
                try:
                    legacy_html_path.write_text(html_text, encoding="utf-8")
                except Exception:
                    pass
                print(f"  ✅ Saved raw HTML to {out_html_path} ({len(html_text):,} chars)")

            downloaded_files.append(out_html_path)

            if auto_parse:
                print(f"  ⚙️ Parsing high-signal sections and tables for {ticker} ({filing_date})...")
                parsed = parse_and_save_filing(
                    raw_html_path=out_html_path,
                    ticker=ticker,
                    filing_date=filing_date,
                    output_dir=processed_dir,
                    cik=str(cik),
                )
                print(f"  ✅ Extracted {len(parsed['sections'])} sections with {parsed['total_tables_preserved']} preserved tables.")

                # Also populate legacy text dir with combined text for backwards compatibility
                legacy_txt_path = LEGACY_TEXT_DIR / f"{ticker}_{filing_date}.txt"
                try:
                    combined_text = "\n\n".join(
                        [f"=== {s_id}: {s_data.get('section_title', s_id)} ===\n{s_data['text']}" for s_id, s_data in parsed["sections"].items()]
                    )
                    legacy_txt_path.write_text(combined_text, encoding="utf-8")
                except Exception:
                    pass

            count += 1

    return downloaded_files


def download(
    ticker: str = "AAPL",
    cik: Optional[str] = None,
    lim: int = 1,
    tickers: Optional[List[str]] = None,
    raw_dir: Path = RAW_10K_DIR,
    processed_dir: Path = PROCESSED_10K_DIR,
    auto_parse: bool = True,
) -> List[Path]:
    """
    Backwards-compatible download entry point.
    Supports single ticker download or multi-ticker ingestion.
    """
    all_downloaded: List[Path] = []
    target_tickers = tickers if tickers else ([ticker] if ticker else DEFAULT_TICKERS)

    print(f"🚀 Starting SEC EDGAR Form 10-K Ingestion for {len(target_tickers)} companies: {', '.join(target_tickers)}")
    for t in target_tickers:
        files = download_filing_for_ticker(
            ticker=t,
            cik=cik if len(target_tickers) == 1 else None,
            limit=lim,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            auto_parse=auto_parse,
        )
        all_downloaded.extend(files)

    print(f"\n🎉 Form 10-K Ingestion complete! Total filings retrieved: {len(all_downloaded)}")
    return all_downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official SEC EDGAR 10-K filings with table preservation.")
    parser.add_argument("--ticker", default=None, help="Single stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker symbols (e.g. AAPL,MSFT,NVDA,JPM,GOOGL)")
    parser.add_argument("--cik", default=None, help="SEC CIK number for single ticker")
    parser.add_argument("--limit", type=int, default=1, help="Max filings to fetch per ticker (default: 1)")
    parser.add_argument("--all", action="store_true", help="Download all 5 default major companies (AAPL, MSFT, NVDA, JPM, GOOGL)")
    parser.add_argument("--no-parse", action="store_true", help="Skip section extraction and table parsing")
    args = parser.parse_args()

    if args.all:
        selected_tickers = DEFAULT_TICKERS
    elif args.tickers:
        selected_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        selected_tickers = [args.ticker.strip().upper()]
    else:
        # Default to 5 major public companies covering distinct financial reporting structures
        selected_tickers = DEFAULT_TICKERS

    download(
        tickers=selected_tickers,
        cik=args.cik if len(selected_tickers) == 1 else None,
        lim=args.limit,
        auto_parse=not args.no_parse,
    )
