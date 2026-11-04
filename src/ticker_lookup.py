import os
import requests

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "AmanAtarResearch ataraman9@gmail.com")}

KNOWN_TICKERS = {
    "AAPL": {"cik": "320193", "title": "Apple Inc."},
    "MSFT": {"cik": "789019", "title": "Microsoft Corp"},
    "NVDA": {"cik": "1045810", "title": "NVIDIA Corp"},
    "JPM": {"cik": "19617", "title": "JPMorgan Chase & Co"},
    "GOOGL": {"cik": "1652044", "title": "Alphabet Inc."},
    "GOOG": {"cik": "1652044", "title": "Alphabet Inc."},
    "AMZN": {"cik": "1018724", "title": "Amazon.com Inc."},
    "TSLA": {"cik": "1318605", "title": "Tesla, Inc."},
    "META": {"cik": "1326801", "title": "Meta Platforms, Inc."},
    "PFE": {"cik": "78003", "title": "Pfizer Inc."},
    "XOM": {"cik": "34088", "title": "Exxon Mobil Corp"},
    "WMT": {"cik": "104169", "title": "Walmart Inc."},
}


def get_ticker_to_cik_map():
    """Downloads SEC's official ticker-to-CIK mapping with fallback."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        ticker_map = {}
        for entry in data.values():
            ticker_map[entry["ticker"].upper()] = str(entry["cik_str"])
        return ticker_map
    except Exception:
        # Fallback to known tickers
        return {k: v["cik"] for k, v in KNOWN_TICKERS.items()}


def get_cik(ticker: str) -> str:
    """Gets CIK for a given ticker symbol."""
    ticker_clean = ticker.upper().strip()
    if ticker_clean in KNOWN_TICKERS:
        return KNOWN_TICKERS[ticker_clean]["cik"]
    ticker_map = get_ticker_to_cik_map()
    if ticker_clean in ticker_map:
        return ticker_map[ticker_clean]
    raise ValueError(f"Ticker '{ticker}' not found in SEC database.")


def get_company_name(ticker: str) -> str:
    """Gets company name for a given ticker."""
    ticker_clean = ticker.upper().strip()
    if ticker_clean in KNOWN_TICKERS:
        return KNOWN_TICKERS[ticker_clean]["title"]
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker_clean:
                return entry["title"]
    except Exception:
        pass
    return ticker
