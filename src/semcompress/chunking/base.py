from __future__ import annotations

from abc import ABC, abstractmethod

from semcompress.models import ChunkNode


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str) -> list[ChunkNode]:
        """Split text into a hierarchical list of ChunkNodes."""
        ...
