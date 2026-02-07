from __future__ import annotations


def count_tokens(text: str, method: str = "whitespace",
                 encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using the specified method.

    Args:
        text: The text to count tokens for.
        method: "whitespace" (default, fast, zero deps) or "tiktoken" (precise).
        encoding_name: Tiktoken encoding name (only used if method="tiktoken").

    Returns:
        Number of tokens.
    """
    if not text or not text.strip():
        return 0

    if method == "whitespace":
        return len(text.split())
    elif method == "tiktoken":
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for tiktoken token counting. "
                "Install it with: pip install semcompress[tiktoken]"
            )
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    else:
        raise ValueError(f"Unknown token counting method: {method}")
