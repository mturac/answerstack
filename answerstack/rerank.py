"""Stage 3: cascade rerank + quality gate + requery fail-safe."""
from .retrieval import toks

GATE = 0.55


def _cross_encoder_lite(query_terms: set, doc: dict) -> float:
    text = toks(doc["text"])
    title = toks(doc["title"])
    if not query_terms:
        return 0.0
    cov = sum(1 for t in query_terms if t in set(text)) / len(query_terms)
    pos = 1.0 if any(t in title for t in query_terms) else 0.6
    return 0.7 * cov + 0.3 * pos


def rerank_cascade(query: str, candidates: list):
    """Stage A: lexical shortlist. Stage B: cross-encoder-lite scoring."""
    q = set(toks(query))
    stage_a = sorted(
        candidates,
        key=lambda d: len(q & (set(toks(d["title"])) | set(toks(d["text"])))),
        reverse=True,
    )[:5]
    scored = sorted(
        ((_cross_encoder_lite(q, d), d) for d in stage_a),
        key=lambda x: -x[0],
    )
    return scored, (scored[0][0] if scored else 0.0)


def requery(query: str, ranked: list) -> str:
    """Fail-safe: below gate -> expand query with top-doc signal terms."""
    if not ranked:
        return query
    top = ranked[0][1]
    extras = [t for t in toks(top["title"]) if t not in set(toks(query))][:3]
    return query + " " + " ".join(extras)
