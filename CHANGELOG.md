# Changelog

## 0.1.0 (2026-02-06)

Initial release.

- Embedding-based semantic text compaction using sentence-transformers
- Code-aware chunking mode (splits by function/class boundaries)
- Auto-detection of code vs prose input
- Projection and cosine scoring methods
- Iterative batch removal with score recomputation
- Claude Code MCP server integration (`compact_text`, `compact_files`, `compact_directory`)
- Whitespace and tiktoken-based token counting
- Benchmarked on FastAPI, requests, Flask, and click codebases
