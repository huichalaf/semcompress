from semcompress.chunking.hierarchical import HierarchicalChunker
from semcompress.embeddings.cache import CachedEmbedder
from semcompress.engine import CompactionEngine
from semcompress.models import CompactorConfig
from semcompress.scoring.projection import ProjectionScorer
from semcompress.token_counting import count_tokens


def _compact_with_mock(mock_embedder, text, **config_kwargs):
    """Helper to run full compaction pipeline with mock embedder."""
    config = CompactorConfig(**config_kwargs)
    cached = CachedEmbedder(mock_embedder)
    chunker = HierarchicalChunker(cached, config)
    scorer = ProjectionScorer()
    engine = CompactionEngine(cached, scorer, config)

    nodes = chunker.chunk(text)
    if not nodes:
        return None
    return engine.compact(nodes)


class TestEdgeCases:
    def test_single_sentence(self, mock_embedder):
        result = _compact_with_mock(
            mock_embedder, "Just one single sentence here.", target_ratio=0.5
        )
        # With min_chunks_to_keep=1 and only 1 chunk, nothing can be removed
        assert result is not None
        assert result.chunks_kept >= 1

    def test_two_sentences(self, mock_embedder):
        text = "First sentence here. Second sentence here."
        result = _compact_with_mock(mock_embedder, text, target_ratio=0.5)
        assert result is not None
        assert result.text

    def test_unicode_text(self, mock_embedder):
        text = (
            "La inteligencia artificial avanza rapidamente. "
            "Los modelos de lenguaje son cada vez mas potentes. "
            "El procesamiento del lenguaje natural mejora constantemente."
        )
        result = _compact_with_mock(mock_embedder, text, target_ratio=0.5)
        assert result is not None
        assert result.text

    def test_text_with_newlines(self, mock_embedder):
        text = "First paragraph about AI.\n\nSecond paragraph about ML.\n\nThird about NLP."
        result = _compact_with_mock(mock_embedder, text, target_ratio=0.5)
        assert result is not None

    def test_very_aggressive_compression(self, mock_embedder, long_sample_text):
        result = _compact_with_mock(
            mock_embedder, long_sample_text, target_ratio=0.1
        )
        assert result is not None
        assert result.chunks_kept >= 1  # At least something survives

    def test_no_compression_needed(self, mock_embedder):
        text = "Short text."
        result = _compact_with_mock(mock_embedder, text, target_tokens=1000)
        # Text is already under budget, engine should not remove anything
        if result is not None:
            assert result.chunks_removed == 0


class TestTokenCounting:
    def test_whitespace_counting(self):
        assert count_tokens("hello world", "whitespace") == 2
        assert count_tokens("one two three four", "whitespace") == 4

    def test_empty_string(self):
        assert count_tokens("", "whitespace") == 0
        assert count_tokens("   ", "whitespace") == 0

    def test_invalid_method(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            count_tokens("hello", "invalid")
