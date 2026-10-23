"""Citation-faithfulness scorer (heuristic, no LLM judge).
A claim sentence is supported if its content tokens overlap its cited chunk enough."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answerstack.retrieval import toks


def faithfulness(answer_text: str, cite: dict) -> tuple:
    claims, supported = 0, 0
    for sent in re.split(r"(?<=[.!?])\s+", answer_text):
        marks = re.findall(r"\[(\d+)\]", sent)
        if not marks or len(toks(sent)) < 3:
            continue
        claims += 1
        ok = False
        for m in marks:
            ch = cite.get(int(m))
            if not ch:
                continue
            stoks, ctoks = set(toks(sent)), set(toks(ch["text"]))
            if stoks and len(stoks & ctoks) / len(stoks) >= 0.3:
                ok = True
        supported += ok
    return supported, claims
