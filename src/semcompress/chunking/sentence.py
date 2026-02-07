from __future__ import annotations

import re

# Simple sentence boundary: split after .!? followed by whitespace and a capital letter
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])"
    r"\s+"
    r"(?=[A-Z\"'\(\[])"
)

# Common abbreviations that end with a period but are NOT sentence endings
_ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.",
    "Inc.", "Ltd.", "Corp.", "Vol.", "Dept.", "Est.", "Fig.", "fig.",
    "No.", "St.", "Sgt.", "Gen.", "Gov.", "Pres.", "approx.", "i.e.", "e.g.",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based boundary detection.

    Handles common abbreviations by merging back incorrectly split fragments.
    Falls back to paragraph splitting if no sentence boundaries are found.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings, preserving original text.
    """
    text = text.strip()
    if not text:
        return []

    # Step 1: Split on sentence-like boundaries
    raw_parts = _SENTENCE_SPLIT.split(text)
    raw_parts = [s.strip() for s in raw_parts if s.strip()]

    if not raw_parts:
        return [text]

    # Step 2: Merge back parts that were split after abbreviations
    merged = []
    for part in raw_parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    result = [s for s in merged if s.strip()]

    # Fallback: if regex produced no splits but text has paragraphs
    if len(result) <= 1 and "\n" in text:
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            result = [p.strip() for p in paragraphs if p.strip()]

    return result if result else [text]


def _ends_with_abbreviation(text: str) -> bool:
    """Check if text ends with a known abbreviation."""
    words = text.split()
    if not words:
        return False
    last_word = words[-1]
    return last_word in _ABBREVIATIONS
