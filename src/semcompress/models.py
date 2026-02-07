from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    """A single unit of text (typically a sentence) with its embedding and metadata."""

    text: str
    index: int
    token_count: int = 0
    embedding: np.ndarray | None = field(default=None, repr=False)
    importance_score: float = 0.0
    is_removed: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ChunkNode:
    """A parent chunk containing sub-chunks. Forms the two-level hierarchy."""

    text: str
    index: int
    children: list[Chunk] = field(default_factory=list)
    embedding: np.ndarray | None = field(default=None, repr=False)
    token_count: int = 0

    @property
    def active_children(self) -> list[Chunk]:
        return [c for c in self.children if not c.is_removed]

    @property
    def active_text(self) -> str:
        return " ".join(c.text for c in self.active_children)

    @property
    def active_token_count(self) -> int:
        return sum(c.token_count for c in self.active_children)


@dataclass
class CompactionResult:
    """Result returned after compacting text."""

    text: str
    original_tokens: int
    compacted_tokens: int
    ratio: float
    iterations: int
    chunks_removed: int
    chunks_kept: int
    removed_chunks: list[Chunk] = field(default_factory=list, repr=False)
    kept_chunks: list[Chunk] = field(default_factory=list, repr=False)


@dataclass
class CompactorConfig:
    """Configuration for the compaction process."""

    # Embedding model
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True

    # Mode: "text" (default), "code", or "auto" (auto-detect)
    mode: str = "auto"

    # Chunking
    parent_chunk_target_tokens: int = 256
    min_sub_chunk_tokens: int = 5
    similarity_threshold: float = 0.5

    # Scoring
    scoring_method: str = "projection"  # "projection" | "cosine"

    # Compaction targets (set one)
    target_ratio: float | None = 0.5
    target_tokens: int | None = None

    # Compaction behavior
    batch_removal_fraction: float = 0.1
    min_importance_threshold: float = float("-inf")
    max_iterations: int = 100
    min_chunks_to_keep: int = 1

    # Token counting
    token_counter: str = "whitespace"  # "whitespace" | "tiktoken"
    tiktoken_encoding: str = "cl100k_base"

    # Output
    preserve_order: bool = True
    chunk_separator: str = " "
    parent_separator: str = "\n\n"
