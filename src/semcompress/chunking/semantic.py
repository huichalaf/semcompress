from __future__ import annotations

import numpy as np

from semcompress.chunking.code import detect_code, split_code
from semcompress.chunking.sentence import split_sentences
from semcompress.models import CompactorConfig


class SemanticChunker:
    """Groups sentences into semantic sections by detecting topic shifts.

    Uses embedding similarity between consecutive sentences. When similarity
    drops below a threshold, a new section boundary is created.
    """

    def __init__(self, embedder, config: CompactorConfig):
        self._embedder = embedder
        self._config = config

    def _split(self, text: str) -> list[str]:
        """Route to the correct splitter based on mode config."""
        mode = self._config.mode
        if mode == "code":
            return split_code(text)
        if mode == "text":
            return split_sentences(text)
        # auto: detect
        if detect_code(text):
            return split_code(text)
        return split_sentences(text)

    def create_groups(self, text: str) -> list[list[str]]:
        """Split text into groups of semantically related sentences.

        Args:
            text: Input text.

        Returns:
            List of groups, where each group is a list of sentences.
        """
        sentences = self._split(text)

        if len(sentences) <= 1:
            return [sentences] if sentences else []

        # Compute embeddings for all sentences
        embeddings = self._embedder.embed(sentences)

        # Compute cosine similarity between consecutive pairs
        similarities = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(a, b) / (norm_a * norm_b)))

        # Find breakpoints where similarity drops below threshold
        breakpoints = self._find_breakpoints(similarities)

        # Group sentences between breakpoints
        groups = []
        prev = 0
        for bp in breakpoints:
            if prev < bp:
                groups.append(sentences[prev:bp])
            prev = bp
        if prev < len(sentences):
            groups.append(sentences[prev:])

        # Merge very small groups with their neighbors
        groups = self._merge_small_groups(groups)

        return groups

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """Find indices where topic shifts occur."""
        threshold = self._config.similarity_threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Merge groups that are too small (fewer tokens than min_sub_chunk_tokens)."""
        if len(groups) <= 1:
            return groups

        min_tokens = self._config.min_sub_chunk_tokens
        merged = [groups[0]]

        for group in groups[1:]:
            group_text = " ".join(group)
            prev_text = " ".join(merged[-1])

            if len(group_text.split()) < min_tokens:
                merged[-1].extend(group)
            elif len(prev_text.split()) < min_tokens:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged
