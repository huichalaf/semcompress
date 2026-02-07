import numpy as np

from semcompress.embeddings.cache import CachedEmbedder


class TestCachedEmbedder:
    def test_caches_results(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=100)

        emb1 = cached.embed_single("hello world")
        emb2 = cached.embed_single("hello world")

        np.testing.assert_array_equal(emb1, emb2)
        assert cached.cache_size == 1

    def test_different_texts_different_embeddings(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=100)

        emb1 = cached.embed_single("hello")
        emb2 = cached.embed_single("world")

        assert not np.array_equal(emb1, emb2)
        assert cached.cache_size == 2

    def test_batch_embed_uses_cache(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=100)

        # Pre-cache one text
        cached.embed_single("hello")
        assert cached.cache_size == 1

        # Batch embed including the cached text
        results = cached.embed(["hello", "world"])
        assert results.shape == (2, 384)
        assert cached.cache_size == 2

    def test_lru_eviction(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=2)

        cached.embed_single("first")
        cached.embed_single("second")
        assert cached.cache_size == 2

        cached.embed_single("third")
        assert cached.cache_size == 2  # "first" should be evicted

    def test_invalidate(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=100)

        cached.embed_single("hello")
        assert cached.cache_size == 1

        cached.invalidate("hello")
        assert cached.cache_size == 0

    def test_clear(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder, maxsize=100)

        cached.embed_single("hello")
        cached.embed_single("world")
        assert cached.cache_size == 2

        cached.clear()
        assert cached.cache_size == 0

    def test_dimension(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder)
        assert cached.dimension == 384

    def test_empty_batch(self, mock_embedder):
        cached = CachedEmbedder(mock_embedder)
        result = cached.embed([])
        assert result.shape == (0, 384)
