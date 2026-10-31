import argparse
import os
import time
from pathlib import Path

import requests

DATA_DIR = Path("data/raw/sec")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "AmanAtarResearch ataraman9@gmail.com")}


def download(ticker: str = "AAPL", cik: str = "320193", lim: int = 3) -> None:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    res = response.json()

    recent = res["filings"]["recent"]
    count = 0

    for i, form in enumerate(recent["form"]):
        if form == "10-K" and count < lim:
            acc = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"

            print(f"Downloading {ticker} 10-K from {filing_date}...")
            filing_response = requests.get(filing_url, headers=HEADERS, timeout=30)
            filing_response.raise_for_status()
            html = filing_response.text

            out_path = DATA_DIR / f"{ticker}_{filing_date}.html"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)

            count += 1
            time.sleep(1)

    print(f"✅ Downloaded {count} filings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC EDGAR 10-K filings.")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--cik", default="320193", help="SEC CIK number (default: 320193 for Apple)")
    parser.add_argument("--limit", type=int, default=3, help="Max filings to fetch (default: 3)")
    args = parser.parse_args()
    download(args.ticker, args.cik, args.limit)
