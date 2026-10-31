import requests

HEADERS = {"User-Agent": "Aman Atar ataraman9@gmail.com"}


def get_ticker_to_cik_map():
    """Downloads SEC's official ticker-to-CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    ticker_map = {}
    for entry in data.values():
        ticker_map[entry["ticker"].upper()] = str(entry["cik_str"])

    return ticker_map


def get_cik(ticker: str):
    """Gets CIK for a given ticker symbol."""
    ticker_map = get_ticker_to_cik_map()
    ticker = ticker.upper().strip()

    if ticker in ticker_map:
        return ticker_map[ticker]
    raise ValueError(f"Ticker '{ticker}' not found in SEC database.")


def get_company_name(ticker: str):
    """Gets company name for a given ticker."""
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return entry["title"]
    return ticker