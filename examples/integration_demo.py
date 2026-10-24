"""Compare legacy AnswerStack vs integrated V2 on queries where they materially differ."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from answerstack.intent import classify_intent
from answerstack.pipeline import AnswerStack
from answerstack.pipeline_v2 import AnswerStackV2
from answerstack.providers import MockLLM
from answerstack.retrieval import LocalCorpus


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def show(name, ans):
    print(f"--- {name} ---")
    print(ans.text)
    for stage, data in ans.trace:
        print(f"  {stage}: {data}")
    print()


def main():
    corpus = load(HERE.parent / "eval" / "corpus.jsonl")
    retriever = LocalCorpus(corpus)
    q = "What did Karpathy suggest about ASD-STE100?"
    legacy = AnswerStack(retriever, llm=MockLLM())
    v2 = AnswerStackV2(retriever, llm=MockLLM())
    print(classify_intent(q))
    show("legacy", legacy.run(q))
    show("v2", v2.run(q))

    q2 = "What is the difference between WebSocket and SSE?"
    show("v2 on comparative with same corpus", v2.run(q2))

    q3 = "unanswerable_xyz_qrs"
    a3 = v2.run(q3)
    print("abstention trace:", a3.trace)
    print("abstention says:", a3.text)


if __name__ == "__main__":
    main()
