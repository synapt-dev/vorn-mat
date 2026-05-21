"""Shared char-span and offset helpers for prompt-text probes."""

from __future__ import annotations

import re
from typing import Sequence


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sentence_char_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0

    def _flush(end: int) -> None:
        nonlocal start
        left = start
        right = end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end

    for index, char in enumerate(text):
        if char in ".!?":
            _flush(index + 1)
        elif char == "\n":
            _flush(index)
            start = index + 1

    _flush(len(text))
    return tuple(spans)


def word_char_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _WORD_RE.finditer(text))


def sentence_char_span(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    for start, end in sentence_char_spans(text):
        if end > char_start and start < char_end:
            return start, end
    return 0, len(text)


def line_char_span(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, char_start)
    start = 0 if start == -1 else start + 1
    end = text.find("\n", char_end)
    end = len(text) if end == -1 else end
    return start, end


def paragraph_char_span(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, char_start)
    start = 0 if start == -1 else start + 2
    end = text.find("\n\n", char_end)
    end = len(text) if end == -1 else end
    return start, end


def token_span_from_offsets(
    offsets: Sequence[tuple[int, int]],
    *,
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    token_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end and not (start == end == 0)
    ]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1
