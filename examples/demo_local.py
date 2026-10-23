import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from answerstack.pipeline import AnswerStack
from answerstack.providers import MockLLM
from answerstack.retrieval import LocalCorpus

corpus = [json.loads(l) for l in (HERE.parent / "eval" / "corpus.jsonl").read_text().splitlines() if l.strip()]
stack = AnswerStack(LocalCorpus(corpus), llm=MockLLM())
ans = stack.run("What did Karpathy suggest about ASD-STE100?")
print("ANSWER\n", ans.text)
print("\nTRACE")
for stage, data in ans.trace:
    print(f"  {stage}: {data}")
print("\nCITATIONS")
for n, ch in ans.citations.items():
    print(f"  [{n}] {ch['title']} <{ch['url']}>")
