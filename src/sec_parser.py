import json
import os
import re
import sys
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple
import warnings

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹"}

HIGH_SIGNAL_SECTIONS = [
    ("Item 1", "Business", r"(?:\n|\A)\s*(?:PART\s+I\s+)?Item\s+1[.\s–—\-]+Business\b"),
    ("Item 1A", "Risk Factors", r"(?:\n|\A)\s*Item\s+1A[.\s–—\-]+Risk\s+Factors\b"),
    ("Item 1B", "Unresolved Staff Comments", r"(?:\n|\A)\s*Item\s+1B[.\s–—\-]+Unresolved\b"),
    ("Item 2", "Properties", r"(?:\n|\A)\s*Item\s+2[.\s–—\-]+Properties\b"),
    ("Item 3", "Legal Proceedings", r"(?:\n|\A)\s*Item\s+3[.\s–—\-]+Legal\s+Proceedings\b"),
    ("Item 7", "MD&A", r"(?:\n|\A)\s*(?:PART\s+II\s+)?Item\s+7[.\s–—\-]+Management[\'’]?s\s+Discussion\b"),
    ("Item 7A", "Quantitative Market Risk", r"(?:\n|\A)\s*Item\s+7A[.\s–—\-]+Quantitative\b"),
    ("Item 8", "Financial Statements & Footnotes", r"(?:\n|\A)\s*Item\s+8[.\s–—\-]+Financial\s+Statements\b"),
    ("Item 9", "Changes & Disagreements", r"(?:\n|\A)\s*Item\s+9[.\s–—\-]+Changes\b"),
]

TARGET_FOCUS_SECTIONS = {"Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"}


def clean_table_cells(raw_cells: List[str]) -> List[str]:
    """Cleans table cells, merging currencies, parentheses, and percentages."""
    cells = [re.sub(r"\s+", " ", c).strip() for c in raw_cells]
    merged = []
    i = 0
    while i < len(cells):
        c = cells[i]
        if not c:
            i += 1
            continue
        # Merge currency symbol with following numeric token
        if c in CURRENCY_SYMBOLS and i + 1 < len(cells) and cells[i + 1]:
            merged.append(c + cells[i + 1])
            i += 2
        # Merge percentage with preceding token
        elif c == "%" and merged:
            merged[-1] = merged[-1] + "%"
            i += 1
        # Merge negative parentheses e.g. ["(", "4", ")"]
        elif c in ("(", "[") and i + 2 < len(cells) and cells[i + 2] in (")", "]"):
            merged.append(f"({cells[i + 1]})")
            i += 3
        # Merge solitary closing bracket or percent
        elif c.startswith("%") and merged:
            merged[-1] = merged[-1] + c
            i += 1
        else:
            merged.append(c)
            i += 1
    return merged


def html_table_to_markdown(table_tag) -> Optional[str]:
    """Converts a BeautifulSoup <table> into a clean pipe-delimited Markdown table."""
    rows: List[List[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["td", "th"])]
        cleaned = clean_table_cells(cells)
        if cleaned:
            rows.append(cleaned)

    if not rows or len(rows) < 2:
        return None

    # Check that table contains substantive data, not pure navigation or single cell
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return None

    # If header row has fewer columns than max_cols, pad or prepend a label
    header = rows[0]
    if len(header) < max_cols:
        missing = max_cols - len(header)
        header = ["Line Item / Description"] * missing + header
        rows[0] = header

    # Standardize column width across all rows
    standardized = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        # Clean pipes inside cells to avoid breaking markdown syntax
        clean_row = [cell.replace("|", "/") for cell in r]
        standardized.append(clean_row)

    header_line = "| " + " | ".join(standardized[0]) + " |"
    sep_line = "| " + " | ".join([":---"] * max_cols) + " |"
    data_lines = ["| " + " | ".join(r) + " |" for r in standardized[1:]]

    return "\n" + "\n".join([header_line, sep_line] + data_lines) + "\n"


def split_table_chunks(table_md: str, max_rows: int = 16) -> List[str]:
    """Splits a large Markdown table across chunks while preserving headers on every chunk."""
    lines = [line.strip() for line in table_md.strip().splitlines() if line.strip()]
    if len(lines) <= 2:
        return [table_md]

    header_line = lines[0]
    sep_line = lines[1] if lines[1].startswith("|") and "---" in lines[1] else "| " + " | ".join([":---"] * (header_line.count("|") - 1)) + " |"
    data_rows = lines[2:] if (len(lines) > 2 and "---" in lines[1]) else lines[1:]

    if len(data_rows) <= max_rows:
        return [table_md]

    chunks = []
    for i in range(0, len(data_rows), max_rows):
        sub_rows = data_rows[i : i + max_rows]
        chunk_str = "\n".join([header_line, sep_line] + sub_rows)
        chunks.append(chunk_str)
    return chunks


def chunk_narrative_text(text: str, max_chunk_size: int = 1400, overlap: int = 150) -> List[str]:
    """Splits narrative financial text into coherent chunks respecting sentence and line boundaries."""
    lines = [l.strip() for l in re.split(r"\n+", text) if l.strip()]
    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0

    for line in lines:
        if curr_len + len(line) > max_chunk_size and curr:
            chunks.append(" ".join(curr))
            # Keep overlap lines
            overlap_lines: List[str] = []
            overlap_len = 0
            for prev_l in reversed(curr):
                if overlap_len + len(prev_l) <= overlap:
                    overlap_lines.insert(0, prev_l)
                    overlap_len += len(prev_l)
                else:
                    break
            curr = overlap_lines + [line]
            curr_len = sum(len(x) for x in curr) + len(curr)
        else:
            # If a single line itself is longer than max_chunk_size, slice on sentences
            while len(line) > max_chunk_size:
                split_pt = line[:max_chunk_size].rfind(". ")
                if split_pt == -1 or split_pt < max_chunk_size // 2:
                    split_pt = max_chunk_size
                else:
                    split_pt += 1
                curr.append(line[:split_pt].strip())
                chunks.append(" ".join(curr))
                curr = []
                curr_len = 0
                line = line[split_pt:].strip()
            if line:
                curr.append(line)
                curr_len += len(line) + 1

    if curr:
        chunks.append(" ".join(curr))
    return chunks


def extract_fiscal_year(html_text: str, default_year: Optional[int] = None) -> int:
    """Extracts fiscal year from 10-K text or returns default."""
    m = re.search(r"FOR\s+THE\s+FISCAL\s+YEAR\s+ENDED\s+([A-Za-z]+\s+\d{1,2},\s+(\d{4}))", html_text[:15000], re.IGNORECASE)
    if m:
        return int(m.group(2))
    m2 = re.search(r"fiscal\s+year\s+ended\s+.*?(\d{4})", html_text[:30000], re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return default_year or 2026


def parse_10k_html(
    html_content: str,
    ticker: str,
    filing_date: str = "2026-11-03",
    cik: str = "",
    focus_sections_only: bool = True,
) -> Dict[str, Any]:
    """
    Parses a Form 10-K HTML filing:
    1. Removes script/style/xbrl boilerplate.
    2. Converts HTML <table> elements into clean Markdown tables.
    3. Segments the document into standard Item sections (Item 1, 1A, 7, 7A, 8, etc.).
    4. Returns structured section texts and table counts.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style", "noscript", "ix:header"]):
        tag.decompose()

    # Convert tables to markdown placeholders
    tables = soup.find_all("table")
    preserved_table_count = 0
    for t in tables:
        md = html_table_to_markdown(t)
        if md:
            preserved_table_count += 1
            t.replace_with(soup.new_string(f"\n\n[FINANCIAL_TABLE_START]\n{md}\n[FINANCIAL_TABLE_END]\n\n"))
        else:
            t.replace_with(soup.new_string(" " + t.get_text(separator=" ", strip=True) + " "))

    full_text = soup.get_text(separator="\n")
    full_text = full_text.replace("\xa0", " ").replace("\u200b", "").replace("\u2009", " ")
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)


    fy = extract_fiscal_year(full_text[:35000])

    # Detect section boundaries past the Table of Contents (pos > 6000)
    section_matches = []
    for sec_id, sec_title, pat in HIGH_SIGNAL_SECTIONS:
        all_matches = list(re.finditer(pat, full_text, re.IGNORECASE))
        if all_matches:
            body_matches = [m for m in all_matches if m.start() > 6000]
            chosen = body_matches[0] if body_matches else all_matches[-1]
            section_matches.append((chosen.start(), sec_id, sec_title, chosen.group(0).strip()))

    section_matches.sort(key=lambda x: x[0])

    sections: Dict[str, Dict[str, Any]] = {}
    if len(section_matches) >= 2:
        for idx, (start_pos, sec_id, sec_title, header_txt) in enumerate(section_matches):
            end_pos = section_matches[idx + 1][0] if idx + 1 < len(section_matches) else len(full_text)
            sec_body = full_text[start_pos:end_pos].strip()

            if focus_sections_only and sec_id not in TARGET_FOCUS_SECTIONS:
                continue

            t_count = len(re.findall(r"\[FINANCIAL_TABLE_START\]", sec_body))
            sections[sec_id] = {
                "section_id": sec_id,
                "section_title": sec_title,
                "header_text": header_txt,
                "text": sec_body,
                "table_count": t_count,
                "char_length": len(sec_body),
            }

    # Fallback if formal regex matches were insufficient
    if len(sections) < 3:
        parts = re.split(r"\n(?=Item\s+\d|Management\'?s\s+Discussion|Financial\s+Statements)", full_text, flags=re.IGNORECASE)
        for i, p in enumerate(parts):
            p = p.strip()
            if len(p) < 300:
                continue
            first_line = p.splitlines()[0][:80]
            sec_id = f"Section_{i}"
            for sid, stitle, _ in HIGH_SIGNAL_SECTIONS:
                if sid.lower() in first_line.lower():
                    sec_id = sid
                    break
            if focus_sections_only and sec_id not in TARGET_FOCUS_SECTIONS and not sec_id.startswith("Section_"):
                continue
            t_count = len(re.findall(r"\[FINANCIAL_TABLE_START\]", p))
            sections[sec_id] = {
                "section_id": sec_id,
                "section_title": first_line,
                "header_text": first_line,
                "text": p,
                "table_count": t_count,
                "char_length": len(p),
            }

    # Enhanced Resolution for Cross-Referenced Financial Statements (e.g. JPM, NVDA)
    # When Item 7 or Item 8 only contain cross-reference text ("appears on pages X-Y"),
    # locate and attach the substantive financial statements and MD&A sections.
    if "Item 7" in sections and sections["Item 7"]["char_length"] < 2500:
        mda_match = re.search(
            r"(?:\n|\A)\s*Management[\'’]?s\s+Discussion\s+and\s+Analysis\b",
            full_text[30000:],
            re.IGNORECASE,
        )
        if mda_match:
            mda_start = 30000 + mda_match.start()
            # Stop before Financial Statements
            mda_end_m = re.search(
                r"(?:\n|\A)\s*(?:Consolidated\s+Financial\s+Statements|Report\s+of\s+Independent)\b",
                full_text[mda_start + 5000:],
                re.IGNORECASE,
            )
            mda_end = (mda_start + 5000 + mda_end_m.start()) if mda_end_m else min(len(full_text), mda_start + 500000)
            substantive_mda = full_text[mda_start:mda_end].strip()
            if len(substantive_mda) > 5000:
                combined_mda = sections["Item 7"]["text"] + "\n\n" + substantive_mda
                t_count = len(re.findall(r"\[FINANCIAL_TABLE_START\]", combined_mda))
                sections["Item 7"]["text"] = combined_mda
                sections["Item 7"]["char_length"] = len(combined_mda)
                sections["Item 7"]["table_count"] = t_count

    if "Item 8" in sections and sections["Item 8"]["char_length"] < 2500:
        fs_match = re.search(
            r"(?:\n|\A)\s*(?:Report\s+of\s+Independent|Audited\s+Financial\s+Statements|Consolidated\s+Statements\s+of\s+(?:Income|Operations|Earnings)\b|Notes\s+to\s+(?:the\s+)?Consolidated\s+Financial\s+Statements\b|INDEX\s+TO\s+CONSOLIDATED\s+FINANCIAL\s+STATEMENTS)",
            full_text[30000:],
            re.IGNORECASE,
        )
        if fs_match:
            fs_start = 30000 + fs_match.start()
            fs_end_m = re.search(
                r"(?:\n|\A)\s*(?:SIGNATURES\b|Item\s+16\b|EXHIBIT\s+INDEX\b)",
                full_text[fs_start + 10000:],
                re.IGNORECASE,
            )
            fs_end = (fs_start + 10000 + fs_end_m.start()) if fs_end_m else len(full_text)
            substantive_fs = full_text[fs_start:fs_end].strip()
            if len(substantive_fs) > 5000:
                combined_fs = sections["Item 8"]["text"] + "\n\n" + substantive_fs
                t_count = len(re.findall(r"\[FINANCIAL_TABLE_START\]", combined_fs))
                sections["Item 8"]["text"] = combined_fs
                sections["Item 8"]["char_length"] = len(combined_fs)
                sections["Item 8"]["table_count"] = t_count



    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "form": "10-K",
        "fiscal_year": fy,
        "filing_date": filing_date,
        "total_tables_preserved": preserved_table_count,
        "total_sections": len(sections),
        "sections": sections,
    }


def chunk_section(
    section_data: Dict[str, Any],
    ticker: str,
    fiscal_year: int,
    form: str = "10-K",
    max_chunk_size: int = 1400,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Chunks a 10-K section, preserving Markdown tables with attached headers
    and prepending the required provenance header to every chunk.
    Provenance format:
    [Ticker: {ticker} | Form: 10-K | Fiscal Year: {fy} | Section: {item_section} | Chunk: {chunk_id}]
    """
    sec_id = section_data["section_id"]
    sec_title = section_data.get("section_title", sec_id)
    raw_text = section_data["text"]

    chunks_out: List[Dict[str, Any]] = []
    blocks = re.split(r"(\[FINANCIAL_TABLE_START\].*?\[FINANCIAL_TABLE_END\])", raw_text, flags=re.DOTALL)
    chunk_counter = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if block.startswith("[FINANCIAL_TABLE_START]") and block.endswith("[FINANCIAL_TABLE_END]"):
            table_md = block.replace("[FINANCIAL_TABLE_START]", "").replace("[FINANCIAL_TABLE_END]", "").strip()
            table_subchunks = split_table_chunks(table_md, max_rows=16)
            for sub_table in table_subchunks:
                chunk_counter += 1
                prov_header = f"[Ticker: {ticker} | Form: {form} | Fiscal Year: {fiscal_year} | Section: {sec_id} | Chunk: {chunk_counter}]"
                full_chunk_text = f"{prov_header}\n\n{sub_table}"
                chunks_out.append({
                    "chunk_id": chunk_counter,
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "form": form,
                    "item_section": sec_id,
                    "item_title": sec_title,
                    "is_table": True,
                    "table_count": 1,
                    "char_count": len(full_chunk_text),
                    "token_count_est": max(1, len(full_chunk_text) // 4),
                    "provenance": prov_header,
                    "text": full_chunk_text,
                })
        else:
            clean_narrative = re.sub(r"\[/?FINANCIAL_TABLE_(?:START|END)\]", "", block).strip()
            if not clean_narrative:
                continue

            sub_chunks = chunk_narrative_text(clean_narrative, max_chunk_size=max_chunk_size, overlap=chunk_overlap)
            for ch in sub_chunks:
                chunk_counter += 1
                prov_header = f"[Ticker: {ticker} | Form: {form} | Fiscal Year: {fiscal_year} | Section: {sec_id} | Chunk: {chunk_counter}]"
                full_chunk_text = f"{prov_header}\n\n{ch}"
                chunks_out.append({
                    "chunk_id": chunk_counter,
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "form": form,
                    "item_section": sec_id,
                    "item_title": sec_title,
                    "is_table": False,
                    "table_count": 0,
                    "char_count": len(full_chunk_text),
                    "token_count_est": max(1, len(full_chunk_text) // 4),
                    "provenance": prov_header,
                    "text": full_chunk_text,
                })

    return chunks_out


def parse_and_save_filing(
    raw_html_path: Path,
    ticker: str,
    filing_date: str,
    output_dir: Path,
    cik: str = "",
) -> Dict[str, Any]:
    """Parses a raw HTML file and writes structured sections JSON into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_content = raw_html_path.read_text(encoding="utf-8")
    parsed_data = parse_10k_html(html_content, ticker=ticker, filing_date=filing_date, cik=cik)

    out_file = output_dir / f"{ticker}_{filing_date}_parsed.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2)

    return parsed_data
