"""Stage 4+5 (integrated): full-coverage passages -> budgeted evidence -> citation audit.

Backwards compatible: blocks render in the legacy "[n] TITLE\nURL\nTEXT"
format so providers.MockLLM and eval harness keep working.
"""
import hashlib
import re
import unicodedata
from collections import Counter

DEFAULT_MAX_CHARS = 1200
DEFAULT_BUDGET = 12000
DEFAULT_PER_SOURCE = 3
DEFAULT_DEDUP_JACCARD = 0.9


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.strip() for line in t.split("\n")]
    out = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    return "\n".join(out).strip()


def extract(doc: dict) -> str:
    if "html" not in doc:
        return normalize(doc.get("text", ""))
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(doc["html"], "html.parser")
    for node in soup.select("script, style, nav, footer, header, aside, noscript, template, [hidden]"):
        node.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return normalize(main.get_text(separator="\n"))


def _split_paragraph(par: str, max_chars: int) -> list:
    if len(par) <= max_chars:
        return [par]
    parts = re.split(r"(?<=[.!?])\s+", par)
    buf = ""
    out = []
    for p in parts:
        if buf and len(buf) + 1 + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = (buf + " " + p).strip()
    if buf:
        out.append(buf)
    hard = []
    for piece in out:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            hard.append(piece[:cut])
            piece = piece[cut:].lstrip()
        if piece:
            hard.append(piece)
    return hard


def iter_passages(doc: dict, max_chars: int = DEFAULT_MAX_CHARS) -> list:
    """Every run of normalized text, with exact char offsets. No leading-window drop."""
    text = extract(doc)
    if not text:
        return []
    paras = []
    for par in text.split("\n\n"):
        paras.extend(_split_paragraph(par, max_chars))
    passages = []
    offset = 0
    for piece in paras:
        piece = piece.strip()
        if not piece:
            continue
        start = text.find(piece, offset)
        if start == -1:
            start = offset
        end = start + len(piece)
        passages.append({
            "id": f"{doc.get('id', 'doc')}:{start}-{end}",
            "source_id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "text": piece,
            "start": start,
            "end": end,
            "hash": digest(piece),
            "version": digest(text)[:16],
        })
        offset = end
    return passages


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assemble_evidence(query: str, passages: list,
                      budget_chars: int = DEFAULT_BUDGET,
                      per_source: int = DEFAULT_PER_SOURCE,
                      keep_jaccard_below: float = DEFAULT_DEDUP_JACCARD):
    """Pick budgeted, deduped evidence. Returns (blocks, citations, audit)."""
    q_terms = set(re.findall(r"[^\W_]+", query.casefold()))
    scored = []
    for p in passages:
        p_terms = set(re.findall(r"[^\W_]+", p["text"].casefold()))
        hits = q_terms & p_terms
        score = len(hits) * 10 + len(hits) / max(len(p_terms), 1)
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])

    chosen, seen_hashes = [], set()
    used = {"chars": 0, "by_source": Counter()}
    decisions = []
    keep_toks = []
    for score, p in scored:
        reason = None
        if score <= 0:
            reason = "below_threshold"
        elif p["hash"] in seen_hashes:
            reason = "duplicate"
        elif used["by_source"][p["source_id"]] >= per_source:
            reason = "source_cap"
        elif used["chars"] + len(p["text"]) > budget_chars:
            reason = "budget"
        elif any(_jaccard(keep, set(re.findall(r"[^\W_]+", p["text"].casefold()))) >= keep_jaccard_below for keep in keep_toks):
            reason = "near_duplicate"
        if reason:
            decisions.append({"passage": p["id"], "action": "skip", "reason": reason})
            continue
        chosen.append(p)
        seen_hashes.add(p["hash"])
        keep_toks.append(set(re.findall(r"[^\W_]+", p["text"].casefold())))
        used["chars"] += len(p["text"])
        used["by_source"][p["source_id"]] += 1
        decisions.append({"passage": p["id"], "action": "keep", "score": round(score, 2)})

    blocks = []
    cite = {}
    for i, p in enumerate(chosen, 1):
        blocks.append(f"[{i}] {p['title']}\n{p['url']}\n{p['text']}")
        cite[i] = p
    return blocks, cite, {"decisions": decisions, "chars_used": used["chars"], "budget": budget_chars}


def citation_audit(answer: str, citations: dict) -> dict:
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    all_ids = set(citations)
    quoted = set(cited)
    unknown = sorted(n for n in quoted if n not in all_ids)
    uncited_sentences = []
    for sent in re.split(r"(?<=[.!?])\s+", answer):
        clean = re.sub(r"\[\d+\]", "", sent).strip()
        if clean and not re.search(r"\[\d+\]", sent):
            if len(clean.split()) > 2:
                uncited_sentences.append(clean.strip())
    return {
        "unknown": unknown,
        "used": sorted(quoted),
        "uncited_sentences": uncited_sentences[:5],
        "ok": not unknown,
    }
