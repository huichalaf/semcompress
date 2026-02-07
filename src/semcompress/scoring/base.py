from __future__ import annotations

from abc import ABC, abstractmethod

from semcompress.models import Chunk, ChunkNode


class BaseScorer(ABC):
    """Abstract base class for importance scorers."""

    @abstractmethod
    def score(self, parent: ChunkNode, children: list[Chunk]) -> list[float]:
        """Score each child's importance relative to the parent.

        Args:
            parent: The parent ChunkNode (must have embedding set).
            children: List of child Chunks (must have embeddings set).

        Returns:
            List of float scores, one per child, in the same order.
        """
        ...
