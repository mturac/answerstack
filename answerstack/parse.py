"""Stage 4: content understanding. Clean + chunk a fetched doc."""
import re

NAV = re.compile(r"^(\[!|nav|menu|advertisement|cookie|subscribe|sign in|share this)", re.I)


def parse_doc(doc: dict, chunk_chars: int = 700, max_chunks: int = 2) -> list:
    lines = [l for l in doc["text"].splitlines() if l.strip() and not NAV.match(l.strip())]
    clean = " ".join(lines)
    clean = re.sub(r"\s+", " ", clean).strip()
    chunks = []
    for i in range(0, min(len(clean), chunk_chars * max_chunks), chunk_chars):
        chunks.append({
            "id": doc.get("id", ""), "title": doc["title"], "url": doc.get("url", ""),
            "text": clean[i:i + chunk_chars],
        })
    return chunks
