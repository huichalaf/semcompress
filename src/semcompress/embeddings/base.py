from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (n, dim)."""
        ...

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns array of shape (dim,)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        ...
