import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answerstack.evidence import (
    DEFAULT_BUDGET,
    assemble_evidence,
    citation_audit,
    iter_passages,
    normalize,
)
from answerstack.intent import classify_intent
from answerstack.parse import parse_doc
from answerstack.pipeline_v2 import AnswerStackV2
from answerstack.providers import MockLLM
from answerstack.retrieval import LocalCorpus


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"id": "doc-1", "title": "A", "url": "https://x/one",
             "text": "Paragraph alpha. \n\n" + "Filler. " * 200 + "\n\nTail evidence: MULTI PASSAGE TARGET."},
            {"id": "doc-2", "title": "B", "url": "https://x/two",
             "text": "Sentence one. Sentence two. Tail token: trust anchor."},
        ]
        self.retriever = LocalCorpus(self.corpus)

    def test_full_coverage_vs_legacy(self):
        doc = self.corpus[0]
        legacy_chunks = parse_doc(doc)
        v2_pass = iter_passages(doc)
        self.assertLess(len(legacy_chunks), len(v2_pass))

    def test_tail_served(self):
        ans = AnswerStackV2(self.retriever, llm=MockLLM()).run("What is trust anchor?")
        self.assertIn("trust anchor", ans.text)
        self.assertTrue(any(a[1].get("passages", 0) > 1 for a in ans.trace if a[0] == "evidence"))

    def test_budget_enforced(self):
        ps = iter_passages(self.corpus[0])
        b, cite, audit = assemble_evidence("MULTI PASSAGE TARGET", ps, budget_chars=100)
        self.assertLessEqual(audit["chars_used"], 100)
        self.assertEqual(len(b), len(cite))

    def test_citation_audit_catches_unknown(self):
        self.assertEqual(citation_audit("Some text [42]", {})["unknown"], [42])
        self.assertEqual(citation_audit("Some text [1]", {1: {}})["unknown"], [])

    def test_abstain_when_no_evidence(self):
        ans = AnswerStackV2(self.retriever, llm=MockLLM()).run("completely unmatchable xyz123")
        self.assertIn("Insufficient evidence", ans.text)
        self.assertEqual(ans.citations, {})


if __name__ == "__main__":
    unittest.main()
