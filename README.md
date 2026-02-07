# semcompress

**Embedding-based semantic text compaction. Keep what matters, drop what doesn't.**

[![Python](https://img.shields.io/pypi/pyversions/semcompress.svg)](https://pypi.org/project/semcompress/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/huichalaf/semcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/huichalaf/semcompress/actions)

semcompress reduces text and code length while preserving the most semantically important content. Unlike truncation or summarization, it uses embedding projections to score which chunks contribute most to the overall meaning — and removes the ones that don't.

**No API calls. No LLM needed. Runs 100% locally.**

Built for developers building LLM agents, RAG pipelines, and context-constrained applications.

## Before & after

**Input:** 48,769 tokens of FastAPI source code

**Output at 50% compression:**

```
48,769 → 23,989 tokens (49%) in 5.5 seconds
130 blocks removed, 213 kept
```

The compressor keeps core route handlers, middleware, and dependency injection logic. It removes utility helpers, redundant docstrings, and secondary type definitions.

## Benchmarks on real codebases

| Codebase | Tokens | Target | Actual | Kept | Time |
|----------|--------|--------|--------|------|------|
| **FastAPI** (38 files) | 48,769 | 50% | 49.2% | 213 blocks | 5.5s |
| **requests** (12 files) | 8,113 | 50% | 48.7% | 70 blocks | 1.3s |
| **Flask** (4 files) | 7,531 | 50% | 48.7% | 28 blocks | 0.6s |
| **click** (6 files) | 6,586 | 50% | 49.3% | 66 blocks | 2.2s |

Code-mode auto-detects function/class boundaries and preserves structural integrity.

## How it works

```
Text → Semantic Chunks → Embed → Score Importance → Remove Least Important → Compact Text
```

1. **Chunk** — Text is split into semantic sections (topic shifts detected via embedding similarity). Code is split by function/class boundaries.
2. **Embed** — Each chunk gets a vector embedding capturing its meaning (using [sentence-transformers](https://www.sbert.net/)).
3. **Score** — Each chunk is scored by how much it contributes to its section's semantic direction (projection magnitude onto the parent vector).
4. **Remove** — The least important chunks are iteratively removed in batches. After each batch, parent embeddings are recomputed and scores updated.
5. **Reconstruct** — Remaining chunks are joined back into coherent text, preserving original order.

### The math

For a parent section with embedding `v_P` and a child chunk with embedding `v_i`, importance is:

```
importance_i = dot(v_i, v_P) / ||v_P||
```

This is the scalar projection — it measures how much each chunk "points in the same direction" as the overall section. Chunks that strongly align with the section's meaning are kept. Chunks that are orthogonal or opposing are removed first.

## Installation

```bash
pip install semcompress
```

For precise token counting (matching LLM tokenizers):

```bash
pip install semcompress[tiktoken]
```

> **Note:** The first time you run semcompress, it will download the `all-MiniLM-L6-v2` model (~80MB). This happens once and is cached locally.

## Quick start

### Compress text

```python
from semcompress import compact

result = compact("your long text here...", ratio=0.5)

print(result.text)
print(f"Compressed: {result.original_tokens} → {result.compacted_tokens} tokens ({result.ratio:.0%})")
```

### Compress code

```python
from semcompress import compact

source_code = open("my_module.py").read()
result = compact(source_code, ratio=0.5, mode="code")

print(result.text)  # Keeps the most important functions/classes
```

Mode `"auto"` (the default) detects whether input is code or prose automatically.

### With a token budget

```python
result = compact(long_text, tokens=500)
```

### Full control

```python
from semcompress import Compactor, CompactorConfig

config = CompactorConfig(
    target_ratio=0.4,
    mode="auto",                  # "auto", "text", or "code"
    scoring_method="projection",
    batch_removal_fraction=0.05,  # More precise (smaller batches)
    min_chunks_to_keep=1,         # Keep at least 1 chunk per section
)

c = Compactor(config)
result = c.compact(long_text)

print(f"Removed {result.chunks_removed} chunks in {result.iterations} iterations")
print(f"Kept {result.chunks_kept} chunks")
```

### Batch processing

```python
c = Compactor(CompactorConfig(target_ratio=0.5))
results = c.compact_batch([text1, text2, text3])
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"auto"` | `"auto"` (detect code vs text), `"text"`, or `"code"` |
| `model_name` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for embeddings |
| `device` | `"cpu"` | Device for embeddings (`"cpu"` or `"cuda"`) |
| `target_ratio` | `0.5` | Keep this fraction of tokens (0.0–1.0) |
| `target_tokens` | `None` | Target token budget (overrides ratio) |
| `scoring_method` | `"projection"` | `"projection"` or `"cosine"` |
| `similarity_threshold` | `0.5` | Cosine threshold for topic-shift detection |
| `batch_removal_fraction` | `0.1` | Fraction of chunks to remove per iteration |
| `min_chunks_to_keep` | `1` | Minimum chunks to keep per section |
| `max_iterations` | `100` | Maximum removal iterations |
| `token_counter` | `"whitespace"` | `"whitespace"` (fast) or `"tiktoken"` (precise) |
| `tiktoken_encoding` | `"cl100k_base"` | Encoding for tiktoken counter |
| `preserve_order` | `True` | Keep chunks in original order |
| `chunk_separator` | `" "` | Separator between chunks (auto-adjusted for code) |

## Claude Code integration (MCP)

semcompress includes a built-in [MCP server](https://modelcontextprotocol.io/) that gives Claude Code three context-compaction tools: `compact_text`, `compact_files`, and `compact_directory`.

### Setup

**1. Install with MCP support**

```bash
pip install semcompress[mcp]
```

Or from source:

```bash
git clone https://github.com/huichalaf/semcompress.git
cd semcompress
pip install -e ".[mcp]"
```

**2. Verify the server starts**

```bash
python -m semcompress.mcp_server
```

It will hang waiting for input — that's normal (it communicates via stdio). Press `Ctrl+C` to stop.

**3. Configure Claude Code**

Create a `.mcp.json` file in the root of your project:

```json
{
  "mcpServers": {
    "semcompress": {
      "command": "python3",
      "args": ["-m", "semcompress.mcp_server"]
    }
  }
}
```

> **Tip:** If you use a virtual environment, point `command` to the full path of the Python binary:
> ```json
> {
>   "mcpServers": {
>     "semcompress": {
>       "command": "/path/to/your/project/.venv/bin/python",
>       "args": ["-m", "semcompress.mcp_server"]
>     }
>   }
> }
> ```

**4. Restart Claude Code**

Close your current session and reopen it. Claude Code will ask you to approve the MCP server the first time.

**5. Confirm it works**

Run `/context` inside Claude Code. You should see `semcompress` under **MCP Servers** with 3 tools.

### Available tools

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `compact_text` | Compress a text or code string | `text`, `ratio` (0.0-1.0), `mode` ("auto"/"text"/"code") |
| `compact_files` | Read and compress multiple files | `paths` (list of file paths), `ratio` |
| `compact_directory` | Scan and compress an entire directory | `directory`, `extensions` (e.g. `[".py", ".ts"]`), `ratio`, `max_tokens` |

### Example usage inside Claude Code

Once configured, Claude Code can call these tools automatically. You can also ask directly:

- *"Use semcompress to compact the src/ directory at 50%"*
- *"Compress these 3 files with compact_files so they fit in context"*
- *"Compact this code block to keep only the most important parts"*

## Use cases

- **LLM context management** — Fit more information into fixed context windows by compressing retrieved documents.
- **RAG pipelines** — Compress retrieved chunks before passing to the LLM, reducing cost and latency.
- **Code agent context** — Compact entire codebases to give AI coding assistants more context with fewer tokens.
- **Agent memory** — Compact conversation history or tool outputs to keep agents within token budgets.
- **Document preprocessing** — Reduce document length before embedding for vector databases.

## How it compares

| Approach | Preserves meaning? | Needs LLM? | Works on code? | Speed |
|----------|-------------------|------------|----------------|-------|
| **Truncation** | No (loses the end) | No | No | Instant |
| **Summarization** | Yes (rephrased) | Yes | Poorly | Slow |
| **LLMLingua** | Yes (token-level) | Yes (small) | No | Medium |
| **semcompress** | Yes (chunk-level) | No | **Yes** | Fast |

semcompress runs entirely locally with no API calls. It uses a lightweight embedding model (~80MB) and numpy for vector math.

## Development

```bash
git clone https://github.com/huichalaf/semcompress.git
cd semcompress
pip install -e ".[dev,mcp]"
pytest tests/ -m "not slow"     # Fast tests (mock embedder, ~0.05s)
pytest tests/ -m slow           # Integration tests (downloads model)
```

## License

MIT
