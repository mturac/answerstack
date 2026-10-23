"""Stage 2: hybrid retrieval. LocalCorpus = BM25-lite + ngram overlap.
Swap in SearXNG/Tavily by implementing .search(query, k)."""
import math
import re
from collections import Counter

STOP = set("a an the of for to in on and or what how why when who which is are was were do does did with without into about".split())


def toks(s: str) -> list:
    return [t for t in re.findall(r"[a-z0-9%]+", s.lower()) if t not in STOP]


class LocalCorpus:
    def __init__(self, docs: list):
        self.docs = docs
        self.df = Counter()
        for d in docs:
            for t in set(toks(d["title"] + " " + d["text"])):
                self.df[t] += 1

    def _bm25(self, q: list, doc: dict) -> float:
        text = toks(doc["text"]) + toks(doc["title"])
        tf = Counter(text)
        n = len(self.docs)
        score = 0.0
        for t in set(q):
            idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            score += idf * tf.get(t, 0) * 2.2 / (tf.get(t, 0) + 1.2)
        return score

    def _dense(self, q: list, doc: dict) -> float:
        text = toks(doc["text"]) + toks(doc["title"])
        grams = lambda s: {f"{a}~{b}" for a, b in zip(s, s[1:])}
        if not q:
            return 0.0
        hit = sum(1 for t in q if t in set(text))
        return 0.7 * (hit / len(q)) + 0.3 * len(grams(q) & grams(text)) / max(len(grams(q)), 1)

    def search(self, query: str, k: int = 10) -> list:
        q = toks(query)
        scored = sorted(
            self.docs,
            key=lambda d: 0.6 * self._bm25(q, d) + 4.0 * self._dense(q, d),
            reverse=True,
        )
        return scored[:k]
