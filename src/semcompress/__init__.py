"""semcompress — Embedding-based semantic text compaction.

Keep what matters, drop what doesn't. Compresses text by scoring sentence
importance via embedding projections and iteratively removing the least
important chunks.

Usage::

    from semcompress import compact

    result = compact("your long text here...", ratio=0.5)
    print(result.text)
    print(f"Compressed: {result.original_tokens} -> {result.compacted_tokens} tokens")
"""

from __future__ import annotations

from semcompress.compactor import Compactor
from semcompress.models import CompactionResult, CompactorConfig

__all__ = ["compact", "Compactor", "CompactorConfig", "CompactionResult"]

__version__ = "0.1.0"


def compact(
    text: str,
    ratio: float = 0.5,
    tokens: int | None = None,
    model: str = "all-MiniLM-L6-v2",
    method: str = "projection",
    mode: str = "auto",
    **kwargs,
) -> CompactionResult:
    """One-liner convenience function for text compaction.

    Args:
        text: The text to compact.
        ratio: Target compression ratio (0.0-1.0). Keep this fraction of tokens.
            Default 0.5 (keep 50%).
        tokens: Target token budget. If set, overrides ratio.
        model: Sentence-transformer model name. Default "all-MiniLM-L6-v2".
        method: Scoring method — "projection" (default) or "cosine".
        mode: Chunking mode — "auto" (default, detects code vs text),
            "text" (sentence splitting), or "code" (function/class splitting).
        **kwargs: Additional CompactorConfig parameters.

    Returns:
        CompactionResult with .text, .ratio, .original_tokens, .compacted_tokens, etc.

    Example::

        result = compact("long text...", ratio=0.3)
        print(result.text)

        result = compact(source_code, ratio=0.5, mode="code")
        print(result.text)
    """
    config = CompactorConfig(
        target_ratio=ratio,
        target_tokens=tokens,
        model_name=model,
        scoring_method=method,
        mode=mode,
        **kwargs,
    )
    compactor = Compactor(config)
    return compactor.compact(text)
