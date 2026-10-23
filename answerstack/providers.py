"""LLM providers. MockLLM is deterministic: extractive, query-term driven."""
import json
import os
import re
import urllib.request

from .retrieval import toks

TEMPLATE = """Answer the query using ONLY the evidence below. Cite as [n].
Query: {q}

Evidence:
{evidence}

Answer (2-4 sentences, every claim cited [n]):"""


def build_prompt(query: str, blocks: list) -> str:
    return TEMPLATE.format(q=query, evidence="\n\n".join(blocks))


class MockLLM:
    """Deterministic extractive model: ranks evidence sentences by
    query-term overlap, places the citation marker INSIDE the sentence
    so downstream claim-splitters keep it attached. Max 2 claims per doc."""

    def __call__(self, prompt: str) -> str:
        qm = re.search(r"Query: (.+)", prompt)
        query = qm.group(1) if qm else ""
        q = set(toks(query))
        best = []
        for m in re.finditer(r"\[(\d+)\] ([^\n]+)\n([^\n]+)\n(.*?)(?=\n\n\[|$)", prompt, re.S):
            n, body = m.group(1), m.group(4)
            for sent in re.split(r"(?<=[.!?]) ", body):
                terms = set(toks(sent))
                overlap = len(q & terms)
                if overlap:
                    best.append((overlap, int(n), sent.strip()))
        best.sort(key=lambda x: -x[0])
        out, per_doc = [], {}
        for _, n, sent in best:
            if per_doc.get(n, 0) >= 2:
                continue
            if len(out) >= 4:
                break
            core = sent.rstrip(".!;")
            out.append(f"{core} [{n}].")
            per_doc[n] = per_doc.get(n, 0) + 1
        if not out:
            return "No supporting evidence in the corpus. [1]"
        return " ".join(out)


def make_openai(model=None):
    model = model or os.environ.get("AS_MODEL", "gpt-4.1")

    def call(prompt: str) -> str:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode(),
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    return call
