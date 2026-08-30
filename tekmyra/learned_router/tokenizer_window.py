from __future__ import annotations


def tokenize_whitespace(text: str) -> list[str]:
    return text.split()


def head_tail_window(text: str, max_tokens: int, head_tail_ratio: float = 0.5) -> str:
    if max_tokens < 4:
        raise ValueError("max_tokens must be at least 4")
    tokens = tokenize_whitespace(text)
    if len(tokens) <= max_tokens:
        return text
    head_count = max(1, int(max_tokens * head_tail_ratio))
    tail_count = max(1, max_tokens - head_count - 1)
    head = tokens[:head_count]
    tail = tokens[-tail_count:]
    return " ".join(head) + " ... " + " ".join(tail)
