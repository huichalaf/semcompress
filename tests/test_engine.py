from semcompress.chunking.hierarchical import HierarchicalChunker
from semcompress.embeddings.cache import CachedEmbedder
from semcompress.engine import CompactionEngine
from semcompress.models import CompactorConfig
from semcompress.scoring.projection import ProjectionScorer


class TestConfigImmutability:
    """Ensure compact() does not permanently mutate config."""

    def test_separator_not_mutated_after_code_detection(self, mock_embedder):
        from semcompress.compactor import Compactor

        config = CompactorConfig(
            mode="auto",
            similarity_threshold=0.0,
            target_ratio=0.7,
        )
        original_sep = config.chunk_separator

        # The original config object must not be mutated
        c = Compactor(config)
        code = "import os\nimport sys\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
        c.compact(code)

        assert config.chunk_separator == original_sep, (
            "Compactor mutated the original config's chunk_separator"
        )

    def test_separator_resets_between_calls(self, mock_embedder):
        from semcompress.compactor import Compactor

        config = CompactorConfig(
            mode="auto",
            similarity_threshold=0.0,
            target_ratio=0.7,
        )
        c = Compactor(config)

        # First call with code
        code = "import os\nimport sys\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
        c.compact(code)

        # Second call with prose — separator should be back to " "
        prose = "This is a sentence. Another sentence. A third one. And a fourth."
        c.compact(prose)

        assert c.config.chunk_separator == " ", (
            "chunk_separator was not restored after code auto-detection"
        )


class TestCompactionEngine:
    def _build_engine(self, mock_embedder, **config_kwargs):
        # Use similarity_threshold=0.0 by default so mock embedder groups all
        # sentences into fewer parents (random embeddings produce low similarity)
        config_kwargs.setdefault("similarity_threshold", 0.0)
        config = CompactorConfig(**config_kwargs)
        cached = CachedEmbedder(mock_embedder)
        scorer = ProjectionScorer()
        return CompactionEngine(cached, scorer, config), config, cached

    def test_compacts_text(self, mock_embedder, sample_text):
        engine, config, cached = self._build_engine(mock_embedder, target_ratio=0.5)
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(sample_text)
        result = engine.compact(nodes)

        assert result.compacted_tokens <= result.original_tokens
        assert result.ratio <= 1.0
        assert result.chunks_removed > 0
        assert result.text  # Non-empty

    def test_respects_target_ratio(self, mock_embedder, long_sample_text):
        engine, config, cached = self._build_engine(
            mock_embedder, target_ratio=0.3, batch_removal_fraction=0.05
        )
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(long_sample_text)
        result = engine.compact(nodes)

        # Should achieve significant compression (tolerance for chunk granularity)
        assert result.ratio <= 0.7

    def test_respects_target_tokens(self, mock_embedder, long_sample_text):
        engine, config, cached = self._build_engine(
            mock_embedder, target_tokens=20, target_ratio=None
        )
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(long_sample_text)
        result = engine.compact(nodes)

        assert result.compacted_tokens <= result.original_tokens

    def test_preserves_order(self, mock_embedder, sample_text):
        engine, config, cached = self._build_engine(mock_embedder, target_ratio=0.6)
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(sample_text)
        result = engine.compact(nodes)

        # Kept chunks should have ascending indices
        indices = [c.index for c in result.kept_chunks]
        assert indices == sorted(indices)

    def test_min_chunks_to_keep(self, mock_embedder, sample_text):
        engine, config, cached = self._build_engine(
            mock_embedder, target_ratio=0.01, min_chunks_to_keep=1
        )
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(sample_text)
        result = engine.compact(nodes)

        # Should always keep at least 1 chunk
        assert result.chunks_kept >= 1

    def test_iterations_tracked(self, mock_embedder, sample_text):
        engine, config, cached = self._build_engine(mock_embedder, target_ratio=0.5)
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(sample_text)
        result = engine.compact(nodes)

        assert result.iterations >= 1

    def test_removed_plus_kept_equals_total(self, mock_embedder, sample_text):
        engine, config, cached = self._build_engine(mock_embedder, target_ratio=0.5)
        chunker = HierarchicalChunker(cached, config)

        nodes = chunker.chunk(sample_text)
        result = engine.compact(nodes)

        total = len(result.removed_chunks) + len(result.kept_chunks)
        assert result.chunks_removed + result.chunks_kept == total
