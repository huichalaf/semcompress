from __future__ import annotations

import re

# Matches function/class/method definitions at any indentation level
_BLOCK_BOUNDARY = re.compile(
    r"\n(?=\s*(?:def |async def |class |@))",
)

# Detects if text looks like code (heuristic)
_CODE_SIGNALS = re.compile(
    r"(?:"
    r"^\s*(?:def |class |import |from .+ import |async def )"
    r"|^\s*(?:if __name__|@\w+)"
    r"|[{}();]\s*$"
    r"|\bfunction\b|\bconst\b|\blet\b|\bvar\b"
    r"|\bfn\b|\bimpl\b|\bstruct\b|\bpub\b"
    r"|\bfunc\b|\bpackage\b"
    r")",
    re.MULTILINE,
)


def detect_code(text: str) -> bool:
    """Heuristic: return True if text looks like source code."""
    lines = text.strip().splitlines()[:50]  # check first 50 lines
    if not lines:
        return False
    code_lines = sum(1 for line in lines if _CODE_SIGNALS.search(line))
    return code_lines / len(lines) > 0.15


def split_code(text: str) -> list[str]:
    """Split source code into logical blocks.

    Splits on top-level definitions (def, class, async def) and decorators.
    Falls back to double-newline splitting for non-Python code.

    Args:
        text: Source code text.

    Returns:
        List of code block strings.
    """
    text = text.strip()
    if not text:
        return []

    # Try splitting on function/class boundaries
    blocks = _BLOCK_BOUNDARY.split(text)
    blocks = [b.strip() for b in blocks if b.strip()]

    # If we got meaningful splits, return them
    if len(blocks) > 1:
        return blocks

    # Fallback: split on double newlines (works for any language)
    blocks = re.split(r"\n\s*\n", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    return blocks if blocks else [text]
