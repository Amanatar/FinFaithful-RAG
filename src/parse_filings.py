from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw/sec")
TEXT_DIR = Path("data/processed/sec_text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def parse() -> None:
    for f in RAW_DIR.glob("*.html"):
        print(f"Parsing {f.name}...")
        with open(f, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file.read(), "lxml")

        out_path = TEXT_DIR / f.with_suffix(".txt").name
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(soup.get_text(separator="\n", strip=True))

    print("✅ Parsed all filings.")


if __name__ == "__main__":
    parse()
