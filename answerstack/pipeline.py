"""answerstack: 6-stage answer engine core.
intent -> hybrid retrieve -> cascade rerank (+gate/requery) -> parse -> assemble -> synth
"""
from dataclasses import dataclass, field

from .assemble import assemble
from .intent import classify_intent, rewrite_for_intent
from .parse import parse_doc
from .providers import MockLLM
from .rerank import GATE, requery, rerank_cascade
from .synth import synthesize


@dataclass
class Answer:
    text: str
    citations: dict
    trace: list = field(default_factory=list)


class AnswerStack:
    def __init__(self, retriever, llm=None, max_requeries: int = 1, top_k: int = 5):
        self.retriever, self.llm = retriever, (llm or MockLLM())
        self.max_requeries, self.top_k = max_requeries, top_k

    def run(self, query: str) -> Answer:
        trace = []
        intent = classify_intent(query)
        trace.append(("intent", intent))
        q = rewrite_for_intent(query, intent)

        ranked, gate_score, attempts = [], 0.0, 0
        while attempts <= self.max_requeries:
            candidates = self.retriever.search(q, k=10)
            ranked, gate_score = rerank_cascade(q, candidates)
            if gate_score >= GATE or attempts == self.max_requeries:
                break
            q = requery(q, ranked)
            attempts += 1
        trace.append(("retrieval", {"candidates": len(candidates), "attempts": attempts + 1, "gate": round(gate_score, 2), "passed": gate_score >= GATE}))

        docs = [d for _, d in ranked[: self.top_k]]
        chunks = [c for d in docs for c in parse_doc(d)]
        blocks, cite = assemble(chunks)
        trace.append(("evidence", {"docs": [d.get("id", d["title"]) for d in docs], "chunks": len(chunks)}))

        text = synthesize(query, blocks, self.llm, intent)
        return Answer(text=text, citations=cite, trace=trace)
