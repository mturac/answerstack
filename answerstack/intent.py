"""Stage 1: intent parsing + query rewrite."""
import re

TREND = re.compile(r"\b(latest|news|today|breaking|current|son dakika|bu (hafta|ay|yil))\b", re.I)
COMPARATIVE = re.compile(r"\b(vs\.?|versus|difference between|compare|farki|karsilastir)\b", re.I)
PROCEDURAL = re.compile(r"\b(how (to|do)|nasil|step|setup|install|kurulum)\b", re.I)


def classify_intent(query: str) -> dict:
    if TREND.search(query):
        kind = "trending"
    elif COMPARATIVE.search(query):
        kind = "comparative"
    elif PROCEDURAL.search(query):
        kind = "procedural"
    else:
        kind = "factual"
    return {"kind": kind, "freshness": kind in ("trending", "factual")}


def rewrite_for_intent(query: str, intent: dict) -> str:
    """Intent-conditioned query expansion, pre-retrieval."""
    kind = intent["kind"]
    if kind == "trending":
        return query + " 2026"
    if kind == "comparative":
        return query + " key differences"
    if kind == "procedural":
        return query + " step by step"
    return query
