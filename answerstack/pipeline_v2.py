"""AnswerStack v2: legacy intent/retrieval/rerank, real evidence engine, same LLM iface."""
from dataclasses import dataclass, field

from .evidence import assemble_evidence, citation_audit, iter_passages
from .intent import classify_intent, rewrite_for_intent
from .pipeline import Answer
from .providers import MockLLM, build_prompt
from .rerank import GATE, requery, rerank_cascade


class AnswerStackV2:
    """Swaps stage 4+5 from string-chopping to evidence assembly. Legacy LLM stays."""

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
        trace.append(("retrieval", {"candidates": len(candidates), "attempts": attempts + 1, "gate": round(gate_score, 2)}))

        docs = [d for _, d in ranked[: self.top_k]]
        passages = [p for d in docs for p in iter_passages(d)]
        blocks, cite, audit = assemble_evidence(query, passages)
        trace.append(("evidence", {"docs": [d.get("id", d["title"]) for d in docs],
                                   "passages": len(passages),
                                   "chosen": len(blocks),
                                   "chars_used": audit["chars_used"]}))

        if not blocks:
            abstain = "Insufficient evidence to answer reliably. No citations."
            return Answer(text=abstain, citations={}, trace=trace + [("synergy", "abstain")])

        text = self.llm(build_prompt(query, blocks))
        gate = citation_audit(text, cite)
        trace.append(("citation_audit", {"unknown": gate["unknown"], "used": gate["used"], "ok": gate["ok"]}))
        return Answer(text=text, citations=cite, trace=trace)
