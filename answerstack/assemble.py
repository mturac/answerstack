"""Stage 5: citation-embedded evidence assembly."""


def assemble(chunks: list):
    blocks, cite = [], {}
    for i, ch in enumerate(chunks, 1):
        blocks.append(f"[{i}] {ch['title']}\n{ch['url']}\n{ch['text']}")
        cite[i] = ch
    return blocks, cite
