"""Stage 6: constrained synthesis. Citations pre-embedded in the prompt."""
from .providers import build_prompt


def synthesize(query: str, blocks: list, llm, intent: dict) -> str:
    prompt = build_prompt(query, blocks)
    return llm(prompt)
