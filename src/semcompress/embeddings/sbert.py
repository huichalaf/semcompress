from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from semcompress.embeddings.base import BaseEmbedder


class SBERTEmbedder(BaseEmbedder):
    """Embedding provider using sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        return self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
