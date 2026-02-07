from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from semcompress.embeddings.base import BaseEmbedder


class CachedEmbedder:
    """Wraps a BaseEmbedder with an LRU cache keyed by text hash.

    This avoids recomputing embeddings for unchanged sub-chunks during
    the iterative compaction loop. Only parent embeddings (computed from
    concatenated remaining text) need fresh computation after removals.
    """

    def __init__(self, embedder: BaseEmbedder, maxsize: int = 4096):
        self._embedder = embedder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_single(self, text: str) -> np.ndarray:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        embedding = self._embedder.embed_single(text)
        self._put(key, embedding)
        return embedding

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        keys = [self._key(t) for t in texts]
        uncached_indices = [i for i, k in enumerate(keys) if k not in self._cache]

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            new_embeddings = self._embedder.embed(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                self._put(keys[idx], emb)

        result = []
        for k in keys:
            self._cache.move_to_end(k)
            result.append(self._cache[k])
        return np.array(result)

    def _put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def invalidate(self, text: str) -> None:
        key = self._key(text)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def cache_size(self) -> int:
        return len(self._cache)
