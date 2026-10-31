from pathlib import Path
import pickle
import re
import faiss
from sentence_transformers import SentenceTransformer

TEXT_DIR = Path("data/processed/sec_text")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]


def split_sections(text: str) -> list[str]:
    parts = re.split(r"\n(?=Note\s+\d|Item\s+\d|PART|Management's Discussion and Analysis|Business)", text)
    return [p.strip() for p in parts if p.strip()]


def build() -> None:
    print("Loading embedding model (this may take a minute)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks: list[str] = []
    meta: list[dict] = []

    for f in TEXT_DIR.glob("*.txt"):
        print(f"Embedding {f.name}...")
        text = f.read_text(encoding="utf-8")
        sections = split_sections(text)
        for i, section in enumerate(sections):
            section_chunks = chunk(section)
            for j, c in enumerate(section_chunks):
                chunks.append(c)
                meta.append({"source": f.name, "section_id": i, "chunk_id": j, "section_title": section.splitlines()[0][:120] if section.splitlines() else ""})

    if not chunks:
        raise RuntimeError("No parsed text files found in data/processed/sec_text. Run parse_filings.py first.")

    print(f"Generating vectors for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    with open(OUT_DIR / "meta.pkl", "wb") as out:
        pickle.dump({"chunks": chunks, "meta": meta}, out)

    print(f"✅ Dense Vector Index Built! ({len(chunks)} chunks)")


if __name__ == "__main__":
    build()
