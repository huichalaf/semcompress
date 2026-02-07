from __future__ import annotations

from semcompress.chunking.code import detect_code
from semcompress.chunking.hierarchical import HierarchicalChunker
from semcompress.embeddings.cache import CachedEmbedder
from semcompress.embeddings.sbert import SBERTEmbedder
from semcompress.engine import CompactionEngine
from semcompress.models import CompactionResult, CompactorConfig
from semcompress.scoring.cosine import CosineScorer
from semcompress.scoring.projection import ProjectionScorer
from semcompress.token_counting import count_tokens


class Compactor:
    """Main entry point for text compaction.

    Wires together the embedder, chunker, scorer, and engine to provide
    a simple interface for compacting text.

    Example::

        from semcompress import Compactor, CompactorConfig

        config = CompactorConfig(target_ratio=0.5)
        c = Compactor(config)
        result = c.compact("your long text here...")
        print(result.text)
        print(f"Ratio: {result.ratio:.0%}")
    """

    def __init__(self, config: CompactorConfig | None = None):
        from dataclasses import replace

        self.config = replace(config or CompactorConfig())

        # Auto-adjust separator for explicit code mode
        if self.config.mode == "code" and self.config.chunk_separator == " ":
            self.config.chunk_separator = "\n\n"

        # Initialize embedder with cache
        raw_embedder = SBERTEmbedder(
            model_name=self.config.model_name,
            device=self.config.device,
            normalize=self.config.normalize_embeddings,
        )
        self._embedder = CachedEmbedder(raw_embedder)

        # Initialize chunker
        self._chunker = HierarchicalChunker(self._embedder, self.config)

        # Initialize scorer
        if self.config.scoring_method == "projection":
            scorer = ProjectionScorer()
        elif self.config.scoring_method == "cosine":
            scorer = CosineScorer()
        else:
            raise ValueError(
                f"Unknown scoring method: {self.config.scoring_method}. "
                "Use 'projection' or 'cosine'."
            )

        # Initialize engine
        self._engine = CompactionEngine(self._embedder, scorer, self.config)

    def compact(self, text: str) -> CompactionResult:
        """Compact a single text.

        Args:
            text: The text to compact.

        Returns:
            CompactionResult with compacted text and metadata.
        """
        # Edge case: empty text
        if not text or not text.strip():
            return CompactionResult(
                text="",
                original_tokens=0,
                compacted_tokens=0,
                ratio=1.0,
                iterations=0,
                chunks_removed=0,
                chunks_kept=0,
            )

        original_tokens = count_tokens(
            text, self.config.token_counter, self.config.tiktoken_encoding
        )

        # Determine target
        if self.config.target_tokens is not None:
            target = self.config.target_tokens
        elif self.config.target_ratio is not None:
            target = int(original_tokens * self.config.target_ratio)
        else:
            target = int(original_tokens * 0.5)

        # Edge case: text already within target
        if original_tokens <= target:
            return CompactionResult(
                text=text,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                ratio=1.0,
                iterations=0,
                chunks_removed=0,
                chunks_kept=0,
            )

        # Auto-detect code mode and adjust separator for this call only
        original_separator = self.config.chunk_separator
        if self.config.mode == "auto" and detect_code(text):
            if self.config.chunk_separator == " ":
                self.config.chunk_separator = "\n\n"

        # Normal compaction flow
        nodes = self._chunker.chunk(text)

        if not nodes:
            return CompactionResult(
                text=text,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                ratio=1.0,
                iterations=0,
                chunks_removed=0,
                chunks_kept=0,
            )

        result = self._engine.compact(nodes)

        # Restore separator so config isn't permanently mutated
        self.config.chunk_separator = original_separator

        return result

    def compact_batch(self, texts: list[str]) -> list[CompactionResult]:
        """Compact multiple texts independently.

        Args:
            texts: List of texts to compact.

        Returns:
            List of CompactionResults, one per input text.
        """
        return [self.compact(t) for t in texts]
