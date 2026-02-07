"""MCP server for semcompress — gives Claude Code a context compaction tool.

Run with:
    python -m semcompress.mcp_server

Configure in Claude Code settings:
    "mcpServers": {
        "semcompress": {
            "command": "python",
            "args": ["-m", "semcompress.mcp_server"]
        }
    }
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("semcompress")

# Lazy-loaded compactor (avoid loading model until first call)
_compactor_cache: dict = {}


def _get_compactor(mode: str = "auto", ratio: float = 0.5):
    from semcompress import Compactor, CompactorConfig

    key = (mode, ratio)
    if key not in _compactor_cache:
        config = CompactorConfig(
            target_ratio=ratio,
            mode=mode,
            similarity_threshold=0.3,
            batch_removal_fraction=0.1,
        )
        _compactor_cache[key] = Compactor(config)
    return _compactor_cache[key]


@mcp.tool()
def compact_text(
    text: str,
    ratio: float = 0.5,
    mode: str = "auto",
) -> str:
    """Compress text or code while preserving the most important content.

    Uses embedding-based semantic scoring to identify and remove
    the least important chunks. Ideal for reducing context window usage.

    Args:
        text: The text or code to compress.
        ratio: Target compression ratio (0.0-1.0). 0.5 keeps ~50% of tokens.
        mode: "auto" (detect text vs code), "text", or "code".

    Returns:
        Compressed text with metadata summary.
    """
    compactor = _get_compactor(mode=mode, ratio=ratio)
    result = compactor.compact(text)

    header = (
        f"[semcompress: {result.original_tokens} → {result.compacted_tokens} tokens "
        f"({result.ratio:.0%}) | {result.chunks_removed} blocks removed, "
        f"{result.chunks_kept} kept]\n\n"
    )
    return header + result.text


@mcp.tool()
def compact_files(
    paths: list[str],
    ratio: float = 0.5,
) -> str:
    """Read and compress multiple files into a single compacted context.

    Reads the given file paths, concatenates them, and compresses
    the result to fit within a token budget. Useful for loading
    a codebase into context efficiently.

    Args:
        paths: List of file paths to read and compress.
        ratio: Target compression ratio (0.0-1.0). 0.5 keeps ~50%.

    Returns:
        Compressed content from all files with metadata.
    """
    parts = []
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        parts.append(f"# FILE: {path}\n{content}")

    if not parts:
        return "No valid files found."

    full_text = "\n\n".join(parts)
    compactor = _get_compactor(mode="code", ratio=ratio)
    result = compactor.compact(full_text)

    header = (
        f"[semcompress: {len(paths)} files | "
        f"{result.original_tokens} → {result.compacted_tokens} tokens "
        f"({result.ratio:.0%}) | {result.chunks_removed} blocks removed, "
        f"{result.chunks_kept} kept]\n\n"
    )
    return header + result.text


@mcp.tool()
def compact_directory(
    directory: str,
    extensions: list[str] | None = None,
    ratio: float = 0.5,
    max_tokens: int = 50000,
) -> str:
    """Read and compress all matching files in a directory.

    Walks the directory, reads files matching the given extensions,
    and compresses them into a single compacted context.

    Args:
        directory: Path to the directory to scan.
        extensions: File extensions to include (e.g. [".py", ".ts"]). Default: [".py"].
        ratio: Target compression ratio (0.0-1.0).
        max_tokens: Maximum input tokens before compression.

    Returns:
        Compressed content with metadata.
    """
    from semcompress.token_counting import count_tokens

    if extensions is None:
        extensions = [".py"]

    directory = os.path.expanduser(directory)
    skip_dirs = {
        "__pycache__", ".git", "node_modules", ".venv",
        "venv", "env", ".tox", "dist", "build",
    }

    parts = []
    total_tokens = 0
    budget_exhausted = False

    for root, dirs, fnames in os.walk(directory):
        if budget_exhausted:
            break
        dirs[:] = [d for d in sorted(dirs) if d not in skip_dirs]
        for f in sorted(fnames):
            if not any(f.endswith(ext) for ext in extensions):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, directory)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

            entry = f"# FILE: {rel}\n{content}"
            entry_tokens = count_tokens(entry, "whitespace")

            if total_tokens + entry_tokens > max_tokens and parts:
                budget_exhausted = True
                break
            parts.append(entry)
            total_tokens += entry_tokens

    if not parts:
        return f"No files with extensions {extensions} found in {directory}."

    full_text = "\n\n".join(parts)
    compactor = _get_compactor(mode="code", ratio=ratio)
    result = compactor.compact(full_text)

    header = (
        f"[semcompress: {len(parts)} files from {directory} | "
        f"{result.original_tokens} → {result.compacted_tokens} tokens "
        f"({result.ratio:.0%}) | {result.chunks_removed} blocks removed, "
        f"{result.chunks_kept} kept]\n\n"
    )
    return header + result.text


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
