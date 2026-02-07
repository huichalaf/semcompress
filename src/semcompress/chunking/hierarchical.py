from __future__ import annotations

from semcompress.chunking.base import BaseChunker
from semcompress.chunking.semantic import SemanticChunker
from semcompress.models import Chunk, ChunkNode, CompactorConfig
from semcompress.token_counting import count_tokens


class HierarchicalChunker(BaseChunker):
    """Builds a two-level chunk hierarchy: parent sections containing sentence sub-chunks.

    Level 1 (parents): Semantic sections detected by topic-shift analysis.
    Level 2 (children): Individual sentences within each section.
    """

    def __init__(self, embedder, config: CompactorConfig):
        self._embedder = embedder
        self._config = config
        self._semantic = SemanticChunker(embedder, config)

    def chunk(self, text: str) -> list[ChunkNode]:
        """Split text into a hierarchical chunk tree.

        Args:
            text: Input text.

        Returns:
            List of ChunkNodes, each containing Chunk children.
        """
        groups = self._semantic.create_groups(text)

        if not groups:
            return []

        nodes = []
        global_idx = 0

        for i, group in enumerate(groups):
            children = []
            for sentence in group:
                token_count = count_tokens(
                    sentence,
                    self._config.token_counter,
                    self._config.tiktoken_encoding,
                )
                children.append(Chunk(
                    text=sentence,
                    index=global_idx,
                    token_count=token_count,
                ))
                global_idx += 1

            parent_text = " ".join(group)
            node = ChunkNode(
                text=parent_text,
                index=i,
                children=children,
                token_count=sum(c.token_count for c in children),
            )
            nodes.append(node)

        return nodes
