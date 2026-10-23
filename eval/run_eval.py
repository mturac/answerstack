import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from answerstack.pipeline import AnswerStack
from answerstack.providers import MockLLM
from answerstack.retrieval import LocalCorpus
from citation_faithfulness import faithfulness


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    corpus = load(HERE / "corpus.jsonl")
    dataset = load(HERE / "dataset.jsonl")
    stack = AnswerStack(LocalCorpus(corpus), llm=MockLLM())
    rows, fails = [], 0
    for item in dataset:
        ans = stack.run(item["q"])
        first_cite = next(iter(ans.citations.values()), {"id": None})
        hit1 = first_cite.get("id") == item["doc_id"]
        has_gold = item["gold"].lower() in ans.text.lower()
        sup, tot = faithfulness(ans.text, ans.citations)
        faith = round(sup / tot, 2) if tot else 0.0
        ok = hit1 and has_gold and faith >= 0.5
        fails += 0 if ok else 1
        rows.append((item["q"][:40], hit1, has_gold, f"{sup}/{tot}", faith, "PASS" if ok else "FAIL"))
    w = max(len(r[0]) for r in rows)
    print(f"{'query':<{w}}  hit@1  gold  cites   faith  result")
    for r in rows:
        print(f"{r[0]:<{w}}  {str(r[1]):5}  {str(r[2]):4}  {r[3]:6}  {r[4]:5}  {r[5]}")
    print(f"\n{len(rows)-fails}/{len(rows)} pass")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
